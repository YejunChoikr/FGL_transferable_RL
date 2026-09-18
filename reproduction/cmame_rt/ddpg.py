"""Deterministic policy gradient (DDPG) update for the CMAME experiments.

Experimental settings are defined in spec/protocol.json.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from . import models_policy as M
from . import policy_init as PI
from .protocol import load_protocol
from .sac import ADAM_BACKENDS, make_adam

_POLICY = load_protocol()["policy"]
_DDPG = _POLICY["ddpg"]

GAMMA: float = float(_POLICY["gamma"])
TAU: float = float(_DDPG["tau"])
BATCH_SIZE: int = int(_DDPG["batch_size"])
ACTOR_LR: float = float(_DDPG["actor_lr"])
CRITIC_LR: float = float(_DDPG["critic_lr"])
OU_MU: float = float(_DDPG["ou_mu"])
OU_THETA: float = float(_DDPG["ou_theta"])
OU_SIGMA: float = float(_DDPG["ou_sigma"])
OU_DT: float = float(_DDPG["ou_dt"])
OU_SIGMA_DECAY: bool = bool(_DDPG["ou_sigma_decay"])
OU_RESET_EVERY_EPISODE: bool = bool(_DDPG["ou_reset_every_episode"])
UPDATE_ORDER: tuple = tuple(_DDPG["update_order"])

STAT_NAMES: tuple = ("critic_loss", "actor_loss", "q_mean", "target_mean")

if OU_SIGMA_DECAY:  # pragma: no cover - protocol invariant
    raise RuntimeError("protocol.policy.ddpg.ou_sigma_decay must be false")


class OUNoise:
    """Ornstein-Uhlenbeck process on the device, reset at every episode start.

    ``x <- x + theta * (mu - x) * dt + sigma * sqrt(dt) * eps`` with ``dt = 1``
    and ``eps`` supplied by the caller, so the process itself consumes no RNG.
    """

    def __init__(self, size: int = 1, device="cuda"):
        self.size = int(size)
        self.device = torch.device(device)
        self.mu = OU_MU
        self.theta = OU_THETA
        self.sigma = OU_SIGMA
        self.dt = OU_DT
        self.state = torch.full((self.size,), self.mu, dtype=torch.float32,
                                device=self.device)
        self.n_steps = 0
        self.n_resets = 0

    def reset(self) -> None:
        self.state = torch.full((self.size,), self.mu, dtype=torch.float32,
                                device=self.device)
        self.n_resets += 1

    def step(self, eps: torch.Tensor) -> torch.Tensor:
        """Advance the process with a standard-normal ``eps`` of shape ``size``."""
        dx = (self.theta * (self.mu - self.state) * self.dt
              + self.sigma * (self.dt ** 0.5) * eps.reshape(self.size))
        self.state = self.state + dx
        self.n_steps += 1
        return self.state

    def state_dict(self) -> dict:
        return {"state": self.state.detach().cpu().clone(),
                "n_steps": self.n_steps, "n_resets": self.n_resets}

    def load_state_dict(self, sd: dict) -> None:
        self.state = sd["state"].to(self.device)
        self.n_steps = int(sd["n_steps"])
        self.n_resets = int(sd["n_resets"])


class DDPGAgent:
    """Single-critic DDPG with target actor and target critic."""

    def __init__(self, device, shared: bool = False,
                 init_blob: Optional[dict] = None,
                 adam_backend: str = "reference"):
        self.device = torch.device(device)
        self.shared = bool(shared)
        self.adam_backend = str(adam_backend)
        self.gamma = GAMMA
        self.tau = TAU
        self.n_updates = 0

        self.actor = M.make_actor("DDPG", self.shared).to(self.device)
        self.critic = M.make_critic(self.shared).to(self.device)
        if init_blob is not None:
            self.load_init(init_blob)
        self.actor_target = M.make_actor("DDPG", self.shared).to(self.device)
        self.critic_target = M.make_critic(self.shared).to(self.device)
        self.hard_update_targets()
        for p in list(self.actor_target.parameters()) + list(
                self.critic_target.parameters()):
            p.requires_grad_(False)

        self.actor_opt = make_adam(self.actor.parameters(), ACTOR_LR,
                                   self.adam_backend)
        self.critic_opt = make_adam(self.critic.parameters(), CRITIC_LR,
                                    self.adam_backend)
        self.noise = OUNoise(1, self.device)

        self._critic_params = list(self.critic.parameters())
        self._actor_params = list(self.actor.parameters())
        self._actor_t_params = list(self.actor_target.parameters())
        self._critic_t_params = list(self.critic_target.parameters())

    # ------------------------------------------------------------------ setup
    def load_init(self, blob: dict) -> None:
        def prep(sd):
            return (PI.widen_fc1_with_zero_goal_column(sd) if self.shared
                    else {k: v.detach().clone() for k, v in sd.items()})

        self.actor.load_state_dict(
            {k: v.to(self.device) for k, v in prep(blob["actor"]).items()})
        self.critic.load_state_dict(
            {k: v.to(self.device) for k, v in prep(blob["critic"]).items()})

    def hard_update_targets(self) -> None:
        """Copy the online actor and critic onto their targets."""
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

    def apply_transfer(self, source: dict, copy_actor: bool = True,
                       copy_critics: bool = True) -> dict:
        """Copy ``fc1..fc4`` from a source checkpoint; heads stay scratch."""
        manifest = {"recipe": "fc1-fc4 actor and critic",
                    "copy_actor": bool(copy_actor),
                    "copy_critics": bool(copy_critics), "tensors": {}}
        heads_before = {
            "actor": PI.head_hashes(self.actor, "actor", "DDPG"),
            "critic": PI.head_hashes(self.critic, "critic", "DDPG"),
        }
        if copy_actor:
            manifest["tensors"]["actor"] = PI.copy_trunk_(
                self.actor, source["actor"], "actor")
        if copy_critics:
            manifest["tensors"]["critic"] = PI.copy_trunk_(
                self.critic, source["critic"], "critic")
        self.hard_update_targets()
        heads_after = {
            "actor": PI.head_hashes(self.actor, "actor", "DDPG"),
            "critic": PI.head_hashes(self.critic, "critic", "DDPG"),
        }
        manifest["n_tensors"] = sum(len(v) for v in
                                    manifest["tensors"].values())
        manifest["heads_unchanged"] = bool(heads_before == heads_after)
        manifest["head_sha256"] = heads_after
        manifest["targets"] = ("hard-copied from the online networks after "
                               "load, target actor included")
        manifest["optimizer_state_transferred"] = False
        manifest["replay_transferred"] = False
        manifest["ou_state_transferred"] = False
        manifest["rng_transferred"] = False
        return manifest

    # -------------------------------------------------------------- behaviour
    def act(self, state: torch.Tensor, eps: torch.Tensor,
            w=None) -> torch.Tensor:
        """Training action ``clip(tanh(actor) + OU, -1, 1)``."""
        with torch.no_grad():
            a = self.actor(state, w)
        n = self.noise.step(eps).reshape(1, -1).expand_as(a)
        return torch.clamp(a + n, -1.0, 1.0)

    def deterministic(self, state: torch.Tensor, w=None) -> torch.Tensor:
        """Evaluation action: no OU noise (protocol.policy.ddpg.evaluation_noise)."""
        with torch.no_grad():
            return self.actor(state, w)

    # ------------------------------------------------------------------ learn
    def update(self, batch: dict) -> torch.Tensor:
        """One locked DDPG update; returns the ``STAT_NAMES`` stats on device."""
        s, a, r, s2, d = (batch["s"], batch["a"], batch["r"], batch["s2"],
                          batch["done"])
        w = batch.get("w")

        with torch.no_grad():
            q_next = self.critic_target(s2, self.actor_target(s2, w), w)
            y = r + self.gamma * (1.0 - d) * q_next
        q = self.critic(s, a, w)
        critic_loss = F.mse_loss(q, y)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()

        for p in self._critic_params:
            p.requires_grad_(False)
        try:
            actor_loss = -self.critic(s, self.actor(s, w), w).mean()
            self.actor_opt.zero_grad(set_to_none=True)
            actor_loss.backward()
            self.actor_opt.step()
        finally:
            for p in self._critic_params:
                p.requires_grad_(True)

        self.soft_update_targets()
        self.n_updates += 1
        return torch.stack([critic_loss.detach(), actor_loss.detach(),
                            q.mean().detach(), y.mean().detach()])

    @torch.no_grad()
    def soft_update_targets(self) -> None:
        for src, tgt in ((self._actor_params, self._actor_t_params),
                         (self._critic_params, self._critic_t_params)):
            torch._foreach_mul_(tgt, 1.0 - self.tau)
            torch._foreach_add_(tgt, src, alpha=self.tau)

    # ------------------------------------------------------------------- audit
    def lr_report(self) -> dict:
        return {"actor": [g["lr"] for g in self.actor_opt.param_groups],
                "critic": [g["lr"] for g in self.critic_opt.param_groups],
                "expected": {"actor": ACTOR_LR, "critic": CRITIC_LR}}

    def optimizer_report(self) -> dict:
        def groups(opt):
            g = opt.param_groups[0]
            return {"lr": g["lr"], "betas": list(g["betas"]), "eps": g["eps"],
                    "weight_decay": g["weight_decay"],
                    "amsgrad": bool(g["amsgrad"]),
                    "foreach": g.get("foreach"), "fused": g.get("fused"),
                    "n_tensors": sum(len(gr["params"])
                                     for gr in opt.param_groups),
                    "n_state_entries": len(opt.state)}

        return {"backend": self.adam_backend, "actor": groups(self.actor_opt),
                "critic": groups(self.critic_opt), "n_updates": self.n_updates,
                "ou": {"mu": OU_MU, "theta": OU_THETA, "sigma": OU_SIGMA,
                       "dt": OU_DT, "sigma_decay": OU_SIGMA_DECAY,
                       "reset_every_episode": OU_RESET_EVERY_EPISODE,
                       "n_steps": self.noise.n_steps,
                       "n_resets": self.noise.n_resets},
                "trainable": {
                    "actor": [n for n, p in self.actor.named_parameters()
                              if p.requires_grad],
                    "critic": [n for n, p in self.critic.named_parameters()
                               if p.requires_grad]}}

    def networks(self) -> dict:
        return {"actor": self.actor, "critic": self.critic,
                "actor_target": self.actor_target,
                "critic_target": self.critic_target}

    def cpu_state(self) -> dict:
        return {name: {k: v.detach().cpu().clone()
                       for k, v in net.state_dict().items()}
                for name, net in self.networks().items()}

    def actor_state(self) -> dict:
        return {k: v.detach().cpu().clone()
                for k, v in self.actor.state_dict().items()}

    def critic_states(self) -> dict:
        return {"critic": {k: v.detach().cpu().clone()
                           for k, v in self.critic.state_dict().items()},
                "actor_target": {k: v.detach().cpu().clone()
                                 for k, v in
                                 self.actor_target.state_dict().items()},
                "critic_target": {k: v.detach().cpu().clone()
                                  for k, v in
                                  self.critic_target.state_dict().items()}}

    def optimizer_state(self) -> dict:
        return {"actor_opt": self.actor_opt.state_dict(),
                "critic_opt": self.critic_opt.state_dict(),
                "ou": self.noise.state_dict(),
                "n_updates": self.n_updates}

    def load_optimizer_state(self, sd: dict) -> None:
        self.actor_opt.load_state_dict(sd["actor_opt"])
        self.critic_opt.load_state_dict(sd["critic_opt"])
        self.noise.load_state_dict(sd["ou"])
        self.n_updates = int(sd["n_updates"])

    def load_cpu_state(self, blob: dict) -> None:
        for name, net in self.networks().items():
            net.load_state_dict({k: v.to(self.device)
                                 for k, v in blob[name].items()})


__all__ = ["DDPGAgent", "OUNoise", "STAT_NAMES", "GAMMA", "TAU", "BATCH_SIZE",
           "ACTOR_LR", "CRITIC_LR", "OU_MU", "OU_THETA", "OU_SIGMA", "OU_DT",
           "OU_SIGMA_DECAY", "OU_RESET_EVERY_EPISODE", "UPDATE_ORDER",
           "ADAM_BACKENDS"]
