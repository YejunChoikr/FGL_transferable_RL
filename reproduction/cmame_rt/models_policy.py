"""Policy networks (actor and critic) for the CMAME unified rerun.

Experimental settings are defined in spec/protocol.json.
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .protocol import load_protocol

_POLICY = load_protocol()["policy"]
_ENC = load_protocol()["encoding"]

#: Hidden widths, from protocol.policy.hidden_widths.
HIDDEN: tuple = tuple(int(x) for x in _POLICY["hidden_widths"])
LOG_STD_MIN: float = float(_POLICY["sac"]["log_std_bounds"][0])
LOG_STD_MAX: float = float(_POLICY["sac"]["log_std_bounds"][1])
TANH_JACOBIAN_EPS: float = float(_POLICY["sac"]["tanh_jacobian_epsilon"])
STATE_DIM: int = int(_ENC["complete_state_dim"])
ACTION_DIM: int = 1
GOAL_DIM: int = 1
DROPOUT: float = float(_POLICY["dropout"])

_LOG_SQRT_2PI = 0.9189385332046727  # log(sqrt(2*pi)), float64 literal

TRUNK_KEYS: tuple = tuple("fc%d.%s" % (i, p) for i in (1, 2, 3, 4)
                          for p in ("weight", "bias"))
SAC_ACTOR_HEAD_KEYS: tuple = ("mu.weight", "mu.bias",
                              "log_std.weight", "log_std.bias")
DDPG_ACTOR_HEAD_KEYS: tuple = ("mu.weight", "mu.bias")
CRITIC_HEAD_KEYS: tuple = ("q.weight", "q.bias")

if DROPOUT != 0:  # pragma: no cover - protocol invariant
    raise RuntimeError("protocol.policy.dropout must be 0 for these networks")


class Actor(nn.Module):
    """fc1..fc4 ReLU trunk with an algorithm-dependent head.

    Parameters
    ----------
    in_dim : int
        60 for a target-specific actor, 61 for a goal-conditioned actor.
    hidden : sequence of int
        Hidden widths; the protocol fixes (4096, 2048, 1024, 512).
    out : int
        Action dimension (1).
    algo : {'SAC', 'DDPG'}
        'SAC' adds a ``log_std`` head and ``forward`` returns ``(mu, log_std)``;
        'DDPG' returns the bounded action ``tanh(mu)``.
    """

    def __init__(self, in_dim: int, hidden: Sequence[int] = HIDDEN,
                 out: int = ACTION_DIM, algo: str = "SAC",
                 log_std_bounds: Sequence[float] = (LOG_STD_MIN, LOG_STD_MAX)):
        super().__init__()
        algo = str(algo).upper()
        if algo not in ("SAC", "DDPG"):
            raise ValueError("algo must be SAC or DDPG, got %r" % algo)
        h = tuple(int(x) for x in hidden)
        if len(h) != 4:
            raise ValueError("exactly four hidden widths are required")
        self.algo = algo
        self.in_dim = int(in_dim)
        self.out_dim = int(out)
        self.log_std_min = float(log_std_bounds[0])
        self.log_std_max = float(log_std_bounds[1])
        dims = (int(in_dim),) + h
        self.fc1 = nn.Linear(dims[0], dims[1])
        self.fc2 = nn.Linear(dims[1], dims[2])
        self.fc3 = nn.Linear(dims[2], dims[3])
        self.fc4 = nn.Linear(dims[3], dims[4])
        self.mu = nn.Linear(dims[4], int(out))
        if algo == "SAC":
            self.log_std = nn.Linear(dims[4], int(out))

    # -------------------------------------------------------------- internals
    def _trunk(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return F.relu(self.fc4(x))

    @staticmethod
    def _cat(s: torch.Tensor, w) -> torch.Tensor:
        return s if w is None else torch.cat([s, w], dim=-1)

    # ------------------------------------------------------------------- API
    def forward(self, s: torch.Tensor, w=None):
        """SAC: ``(mu, clamped log_std)``.  DDPG: ``tanh(mu)``."""
        h = self._trunk(self._cat(s, w))
        mu = self.mu(h)
        if self.algo == "SAC":
            log_std = torch.clamp(self.log_std(h), self.log_std_min,
                                  self.log_std_max)
            return mu, log_std
        return torch.tanh(mu)

    def deterministic(self, s: torch.Tensor, w=None) -> torch.Tensor:
        """Deterministic action. Forward pass only; consumes no RNG.

        SAC uses ``tanh(mu)`` (protocol.policy.sac.deterministic_action). The
        ``actor.sample()`` route is forbidden because it also draws a
        Gaussian sample and advances the RNG.
        """
        if self.algo == "SAC":
            mu, _ = self.forward(s, w)
            return torch.tanh(mu)
        return self.forward(s, w)

    def rsample(self, s: torch.Tensor, noise: torch.Tensor, w=None):
        """Reparameterised squashed-Gaussian sample from externally drawn noise.

        ``noise`` must be standard normal with the shape of ``mu``. Returns
        ``(action, log_prob)``; ``log_prob`` carries the tanh Jacobian
        correction with epsilon ``protocol.policy.sac.tanh_jacobian_epsilon``
        and is summed over the action dimension.
        """
        if self.algo != "SAC":
            raise RuntimeError("rsample is defined for the SAC actor only")
        mu, log_std = self.forward(s, w)
        std = log_std.exp()
        z = mu + std * noise
        action = torch.tanh(z)
        log_prob = (-0.5 * ((z - mu) / std) ** 2 - log_std - _LOG_SQRT_2PI
                    - torch.log(1.0 - action.pow(2) + TANH_JACOBIAN_EPS))
        return action, log_prob.sum(dim=-1, keepdim=True)


class Critic(nn.Module):
    """fc1..fc4 ReLU trunk with a linear scalar ``q`` head."""

    def __init__(self, in_dim: int, hidden: Sequence[int] = HIDDEN):
        super().__init__()
        h = tuple(int(x) for x in hidden)
        if len(h) != 4:
            raise ValueError("exactly four hidden widths are required")
        self.in_dim = int(in_dim)
        dims = (int(in_dim),) + h
        self.fc1 = nn.Linear(dims[0], dims[1])
        self.fc2 = nn.Linear(dims[1], dims[2])
        self.fc3 = nn.Linear(dims[2], dims[3])
        self.fc4 = nn.Linear(dims[3], dims[4])
        self.q = nn.Linear(dims[4], 1)

    def forward(self, s: torch.Tensor, a: torch.Tensor, w=None) -> torch.Tensor:
        x = torch.cat([s, a] if w is None else [s, a, w], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        return self.q(x)


# ------------------------------------------------------------------ factories
def actor_in_dim(shared: bool, state_dim: int = STATE_DIM) -> int:
    """Actor input width: ``[s60]`` or ``[s60, w1]``."""
    return int(state_dim) + (GOAL_DIM if shared else 0)


def critic_in_dim(shared: bool, state_dim: int = STATE_DIM,
                  action_dim: int = ACTION_DIM) -> int:
    """Critic input width: ``[s60, a1]`` or ``[s60, a1, w1]``."""
    return int(state_dim) + int(action_dim) + (GOAL_DIM if shared else 0)


def make_actor(algo: str, shared: bool, state_dim: int = STATE_DIM) -> Actor:
    """Build an actor with the protocol input order and hidden widths."""
    return Actor(actor_in_dim(shared, state_dim), HIDDEN, ACTION_DIM, algo)


def make_critic(shared: bool, state_dim: int = STATE_DIM,
                action_dim: int = ACTION_DIM) -> Critic:
    """Build a critic with the protocol input order and hidden widths."""
    return Critic(critic_in_dim(shared, state_dim, action_dim), HIDDEN)


def count_params(net: nn.Module) -> int:
    """Number of trainable parameters."""
    return int(sum(p.numel() for p in net.parameters() if p.requires_grad))


def expected_keys(kind: str, algo: str = "SAC") -> tuple:
    """Full state_dict key tuple for ``kind`` in {'actor', 'critic'}."""
    if kind == "actor":
        head = (SAC_ACTOR_HEAD_KEYS if str(algo).upper() == "SAC"
                else DDPG_ACTOR_HEAD_KEYS)
    elif kind == "critic":
        head = CRITIC_HEAD_KEYS
    else:
        raise ValueError("kind must be 'actor' or 'critic'")
    return TRUNK_KEYS + head


def critic_names(algo: str) -> tuple:
    """Online critic names for an algorithm: SAC has two, DDPG has one."""
    return ("q1", "q2") if str(algo).upper() == "SAC" else ("critic",)


__all__ = ["Actor", "Critic", "HIDDEN", "LOG_STD_MIN", "LOG_STD_MAX",
           "TANH_JACOBIAN_EPS", "STATE_DIM", "ACTION_DIM", "GOAL_DIM",
           "TRUNK_KEYS", "SAC_ACTOR_HEAD_KEYS", "DDPG_ACTOR_HEAD_KEYS",
           "CRITIC_HEAD_KEYS", "actor_in_dim", "critic_in_dim", "make_actor",
           "make_critic", "count_params", "expected_keys", "critic_names"]
