"""GPU-resident surrogate environment for the 10x3 half-lattice assignment task.

Experimental settings are defined in spec/protocol.json.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import torch

from .encoding import (N_CELLS, STATE_DIM, to_cnn_input, torch_location_buffer,
                       traversal as traversal_indices)
from .protocol import load_protocol
from .reward import C1_CANONICAL, C2_CANONICAL

_PROTO = load_protocol()
_POLICY = _PROTO["policy"]

STEPS_PER_EPISODE: int = int(_POLICY["steps_per_episode"])
N_PROBES: int = 9

if STEPS_PER_EPISODE != N_CELLS:  # pragma: no cover - protocol invariant
    raise RuntimeError("steps_per_episode must equal the 30 cells")


def load_scaler_tensors(scaler: dict, device) -> tuple:
    """Return ``(mean[9], scale[9])`` as float32 device tensors.

    Accepts the dict produced by ``cmame_rt.data.fit_y_scaler`` (lists), the
    tensors returned by ``surrogate_train.load_surrogate_for_env``, or numpy
    arrays; the scaler is read once and never revisited during a rollout.
    """
    def _to(v):
        if torch.is_tensor(v):
            return v.detach().to(device=device, dtype=torch.float32).reshape(-1)
        return torch.as_tensor(np.asarray(v, dtype=np.float32),
                               dtype=torch.float32, device=device).reshape(-1)

    mean, scale = _to(scaler["mean"]), _to(scaler["scale"])
    if mean.shape != (N_PROBES,) or scale.shape != (N_PROBES,):
        raise ValueError("y scaler must provide nine means and nine scales")
    if bool((scale <= 0).any()):
        raise ValueError("y scaler has a non-positive scale")
    return mean, scale


class SurrogateEnv:
    """Batched assignment environment backed by a frozen CNN surrogate.

    Parameters
    ----------
    surrogate : torch.nn.Module
        Trained CNN, already on ``device``. It is forced to ``eval()`` and all
        parameters get ``requires_grad_(False)``.
    scaler : dict
        ``{'mean': [9], 'scale': [9]}`` fitted on training displacements only.
    traversal : str
        Name of the cell visiting order in ``protocol.encoding``.
    C1, C2 : float
        Training-objective reward coefficients of the case.
    goals : sequence of int
        Goal set of the run; one entry for a target-specific policy, five for a
        goal-conditioned policy.
    device : torch.device or str
    batch : int
        Number of parallel designs (1 during training, 5 for a shared
        evaluation rollout).
    """

    def __init__(self, surrogate, scaler, traversal, C1: float, C2: float,
                 goals: Sequence[int], device, batch: int = 1):
        self.device = torch.device(device)
        self.batch = int(batch)
        self.C1 = float(C1)
        self.C2 = float(C2)
        self.goals = tuple(int(g) for g in goals)
        self.traversal_name = str(traversal)
        order = traversal_indices(self.traversal_name)
        if sorted(order) != list(range(N_CELLS)):
            raise ValueError("traversal %r does not visit 30 cells once each"
                             % self.traversal_name)
        self.order = tuple(int(k) for k in order)

        self.surrogate = surrogate.to(self.device).eval()
        for p in self.surrogate.parameters():
            p.requires_grad_(False)
        for b in self.surrogate.buffers():
            b.requires_grad_(False)

        self.y_mean, self.y_scale = load_scaler_tensors(scaler, self.device)
        self._loc = torch_location_buffer(self.device, torch.float32)

        self.state = torch.zeros((self.batch, STATE_DIM), dtype=torch.float32,
                                 device=self.device)
        self.goal = torch.full((self.batch,), self.goals[0],
                               dtype=torch.long, device=self.device)
        self._t = 0
        self._surrogate_calls = 0

    # ------------------------------------------------------------------ views
    @property
    def t(self) -> int:
        """Number of assignments already made in the current episode."""
        return self._t

    @property
    def surrogate_calls(self) -> int:
        return self._surrogate_calls

    def omega(self) -> torch.Tensor:
        """Goal coordinate ``(g - 5) / 2`` as a ``[B, 1]`` float32 tensor."""
        return ((self.goal.to(torch.float32) - 5.0) / 2.0).unsqueeze(1)

    def actions_row_major(self) -> torch.Tensor:
        """The 30 assigned actions in row-major cell order, ``[B, 30]``."""
        return self.state[:, 0::2].clone()

    # ------------------------------------------------------------------- core
    def reset(self, goal_batch=None) -> torch.Tensor:
        """Clear the lattice and set the episode goals; return the state."""
        self.state.zero_()
        self._t = 0
        if goal_batch is None:
            g = torch.full((self.batch,), self.goals[0], dtype=torch.long,
                           device=self.device)
        else:
            g = torch.as_tensor(goal_batch, dtype=torch.long,
                                device=self.device).reshape(-1)
            if g.numel() == 1 and self.batch > 1:
                g = g.expand(self.batch).clone()
            if g.numel() != self.batch:
                raise ValueError("goal_batch must have %d entries" % self.batch)
        self.goal = g
        return self.state

    def step(self, actions: torch.Tensor):
        """Assign one action per design and advance the traversal.

        Returns ``(next_state, reward[B,1], done, info)``. ``info`` carries
        ``u9`` (mm) and ``actions30`` on the terminal step and ``None``
        otherwise.
        """
        if self._t >= STEPS_PER_EPISODE:
            raise RuntimeError("episode already finished")
        a = actions.reshape(self.batch).to(torch.float32)
        k = self.order[self._t]
        self.state[:, 2 * k] = a
        self.state[:, 2 * k + 1] = self._loc[k]
        self._t += 1
        done = self._t == STEPS_PER_EPISODE
        if not done:
            reward = torch.zeros((self.batch, 1), dtype=torch.float32,
                                 device=self.device)
            return self.state, reward, False, {"u9": None, "actions30": None}
        u9 = self.predict_u9(self.state)
        acts = self.state[:, 0::2]
        reward = self.reward_from(u9, acts, self.goal,
                                  self.C1, self.C2).unsqueeze(1)
        return self.state, reward, True, {"u9": u9, "actions30": acts.clone()}

    # -------------------------------------------------------------- surrogate
    def predict_u9(self, state60: torch.Tensor) -> torch.Tensor:
        """Nine probe displacements in mm for a batch of complete states."""
        with torch.no_grad():
            pred = self.surrogate(to_cnn_input(state60))
        self._surrogate_calls += int(state60.shape[0])
        return pred * self.y_scale + self.y_mean

    @staticmethod
    def reward_from(u9: torch.Tensor, actions30: torch.Tensor,
                    goal: torch.Tensor, C1: float, C2: float) -> torch.Tensor:
        """Terminal reward ``[B]`` for arbitrary goals and coefficients.

        The network and the replay stay float32 (protocol.runtime.precision),
        but this scalar metric is evaluated in float64 and cast back at the end.
        Two terms in it lose precision badly in float32: the numerator
        ``u[g] - C1*(u[g-1]+u[g+1])`` cancels, and ``rho`` is a 30-element mean.
        Accumulating in float64 costs nothing at these sizes and makes the
        result agree with ``spec/reference_math.py`` to one float32 rounding of
        the final cast (relative 6e-8), which is what the math gate checks.
        """
        idx = goal.reshape(-1).to(torch.long) - 1
        u = u9.to(torch.float64)
        a = actions30.to(torch.float64)
        ug = u.gather(1, idx.unsqueeze(1)).squeeze(1)
        ul = u.gather(1, (idx - 1).unsqueeze(1)).squeeze(1)
        ur = u.gather(1, (idx + 1).unsqueeze(1)).squeeze(1)
        rho = ((a + 1.0) / 2.0).mean(dim=1)
        r = (ug - float(C1) * (ul + ur)) / (0.5 + float(C2) * rho)
        return r.to(torch.float32)

    def canonical_reward(self, u9: torch.Tensor, actions30: torch.Tensor,
                         goal: torch.Tensor) -> torch.Tensor:
        """Reward at the canonical reporting coefficients C1=0.5, C2=1.0."""
        return self.reward_from(u9, actions30, goal, C1_CANONICAL,
                                C2_CANONICAL)

    def evaluate_designs(self, actions30: torch.Tensor,
                         goal: torch.Tensor) -> dict:
        """Score complete designs without stepping (gates and direct search)."""
        from .encoding import encode_batch_torch

        a = actions30.to(self.device, torch.float32)
        state = encode_batch_torch(a)
        u9 = self.predict_u9(state)
        g = torch.as_tensor(goal, dtype=torch.long,
                            device=self.device).reshape(-1)
        return {
            "state60": state,
            "u9": u9,
            "training_objective": self.reward_from(u9, a, g, self.C1, self.C2),
            "canonical_reward": self.canonical_reward(u9, a, g),
        }

    # ------------------------------------------------------------------ audit
    def config(self) -> dict:
        """Resolved environment settings for ``resolved_config.json``."""
        return {
            "traversal": self.traversal_name,
            "traversal_order": list(self.order),
            "steps_per_episode": STEPS_PER_EPISODE,
            "state_dim": STATE_DIM,
            "C1": self.C1,
            "C2": self.C2,
            "C1_canonical": C1_CANONICAL,
            "C2_canonical": C2_CANONICAL,
            "goals": list(self.goals),
            "batch": self.batch,
            "device": str(self.device),
            "terminal_rule": _POLICY["termination"],
            "reward_formula": _PROTO["reward"]["formula"],
            "y_scaler_mean": [float(x) for x in self.y_mean.tolist()],
            "y_scaler_scale": [float(x) for x in self.y_scale.tolist()],
            "goal_not_given_to_surrogate": bool(
                _PROTO["encoding"]["goal_not_given_to_surrogate"]),
        }


def relabel_terminal_rewards(u9: torch.Tensor, actions30: torch.Tensor,
                             done: torch.Tensor, goal: torch.Tensor,
                             C1: float, C2: float) -> torch.Tensor:
    """Recompute a minibatch's rewards for freshly drawn goals, shape ``[B,1]``.

    Non-terminal transitions score exactly zero at every goal, so the terminal
    mask is applied after the formula. No surrogate call happens here
    (protocol.policy.shared.relabel_not_new_surrogate_call).
    """
    d = done.reshape(-1)
    g = goal.reshape(-1).to(torch.long)
    safe_u9 = torch.where(d.unsqueeze(1) > 0, u9, torch.ones_like(u9))
    r = SurrogateEnv.reward_from(safe_u9, actions30, g, C1, C2)
    return (r * d).unsqueeze(1)


def physical_thickness_from_actions(actions30, domain: str) -> np.ndarray:
    """Map normalised actions onto the physical thickness bounds of a domain."""
    from .encoding import physical_thickness

    bounds = _PROTO["domains"][domain]["thickness_bounds_mm"]
    a = (actions30.detach().cpu().numpy() if torch.is_tensor(actions30)
         else np.asarray(actions30))
    return physical_thickness(a, float(bounds[0]), float(bounds[1]))


__all__ = ["SurrogateEnv", "STEPS_PER_EPISODE", "load_scaler_tensors",
           "relabel_terminal_rewards",
           "physical_thickness_from_actions"]
