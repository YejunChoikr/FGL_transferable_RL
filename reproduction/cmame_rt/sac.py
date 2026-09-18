"""Soft actor-critic update for the CMAME experiments.

Experimental settings are defined in spec/protocol.json.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F

from . import models_policy as M
from . import policy_init as PI
from .protocol import load_protocol

_POLICY = load_protocol()["policy"]
_SAC = _POLICY["sac"]
_OPT = _POLICY["optimizer"]

GAMMA: float = float(_POLICY["gamma"])
TAU: float = float(_SAC["tau"])
BATCH_SIZE: int = int(_SAC["batch_size"])
ACTOR_LR: float = float(_SAC["actor_lr"])
CRITIC_LR: float = float(_SAC["critic_lr"])
ALPHA_LR: float = float(_SAC["alpha_lr"])
INITIAL_ALPHA: float = float(_SAC["initial_alpha"])
TARGET_ENTROPY: float = float(_SAC["target_entropy"])
REWARD_SCALE: float = float(_SAC["reward_scale"])
UPDATE_ORDER: tuple = tuple(_SAC["update_order"])

#: Adam backend flag sets compared in preflight (SPEED_AND_FINAL_CHECK_ko.md).
ADAM_BACKENDS: dict = {
    "reference": {"foreach": False, "fused": False},
    "foreach": {"foreach": True, "fused": False},
    "fused": {"foreach": False, "fused": True},
}

STAT_NAMES: tuple = ("q1_loss", "q2_loss", "actor_loss", "alpha_loss", "alpha",
                     "entropy", "q1_mean", "target_mean")


def make_adam(params, lr: float, backend: str = "reference"):
    """Adam with the locked hyper-parameters and one selected backend.

    Only the ``foreach``/``fused`` execution flags vary; lr, betas, eps,
    weight decay, amsgrad and the update order are fixed by the protocol
    (protocol.runtime.adam_backend_release.not_algorithm_changes).
    """
    if backend not in ADAM_BACKENDS:
        raise ValueError("unknown adam backend %r; known: %s"
                         % (backend, sorted(ADAM_BACKENDS)))
    flags = ADAM_BACKENDS[backend]
    return torch.optim.Adam(
        params,
        lr=float(lr),
        betas=tuple(float(b) for b in _OPT["betas"]),
        eps=float(_OPT["eps"]),
        weight_decay=float(_OPT["weight_decay"]),
        amsgrad=bool(_OPT["amsgrad"]),
        foreach=flags["foreach"],
        fused=flags["fused"],
    )


def _hard_copy(src: torch.nn.Module, dst: torch.nn.Module) -> None:
    dst.load_state_dict(src.state_dict())


class SACAgent:
    """Twin-critic SAC with an optional appended goal column."""

    def __init__(self, device, shared: bool = False,
                 init_blob: Optional[dict] = None,
                 adam_backend: str = "reference"):
        self.device = torch.device(device)
        self.shared = bool(shared)
        self.adam_backend = str(adam_backend)
        self.gamma = GAMMA
        self.tau = TAU
        self.reward_scale = REWARD_SCALE
        self.target_entropy = TARGET_ENTROPY
        self.n_updates = 0

        self.actor = M.make_actor("SAC", self.shared).to(self.device)
        self.q1 = M.make_critic(self.shared).to(self.device)
        self.q2 = M.make_critic(self.shared).to(self.device)
        if init_blob is not None:
            self.load_init(init_blob)
        self.q1_target = M.make_critic(self.shared).to(self.device)
        self.q2_target = M.make_critic(self.shared).to(self.device)
        self.hard_update_targets()
        for p in list(self.q1_target.parameters()) + list(
                self.q2_target.parameters()):
            p.requires_grad_(False)

        self.actor_opt = make_adam(self.actor.parameters(), ACTOR_LR,
                                   self.adam_backend)
        self.q1_opt = make_adam(self.q1.parameters(), CRITIC_LR,
                                self.adam_backend)
        self.q2_opt = make_adam(self.q2.parameters(), CRITIC_LR,
                                self.adam_backend)
        self.log_alpha = torch.tensor(math.log(INITIAL_ALPHA),
                                      dtype=torch.float32, device=self.device,
                                      requires_grad=True)
        self.alpha_opt = make_adam([self.log_alpha], ALPHA_LR,
                                   self.adam_backend)

        self._critic_params = (list(self.q1.parameters())
                               + list(self.q2.parameters()))
        self._q1t_params = list(self.q1_target.parameters())
        self._q2t_params = list(self.q2_target.parameters())
        self._q1_params = list(self.q1.parameters())
        self._q2_params = list(self.q2.parameters())

    # ------------------------------------------------------------------ setup
    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def load_init(self, blob: dict) -> None:
        """Load the paired scratch tensors, widening them when goal-conditioned."""
        def prep(sd):
            return (PI.widen_fc1_with_zero_goal_column(sd) if self.shared
                    else {k: v.detach().clone() for k, v in sd.items()})

        self.actor.load_state_dict(
            {k: v.to(self.device) for k, v in prep(blob["actor"]).items()})
        self.q1.load_state_dict(
            {k: v.to(self.device) for k, v in prep(blob["q1"]).items()})
        self.q2.load_state_dict(
            {k: v.to(self.device) for k, v in prep(blob["q2"]).items()})

    def hard_update_targets(self) -> None:
        """Copy both online critics onto their targets."""
        _hard_copy(self.q1, self.q1_target)
        _hard_copy(self.q2, self.q2_target)

    def apply_transfer(self, source: dict, copy_actor: bool = True,
                       copy_critics: bool = True) -> dict:
        """Copy ``fc1..fc4`` from a source checkpoint (arms ST_PA / ST_PAC).

        Heads keep their paired-scratch values, targets are hard-copied after
        the load, and optimizer state, replay, log_alpha and RNG are untouched.
        """
        manifest = {"recipe": "A4-C4-S0", "copy_actor": bool(copy_actor),
                    "copy_critics": bool(copy_critics), "tensors": {}}
        heads_before = {
            "actor": PI.head_hashes(self.actor, "actor", "SAC"),
            "q1": PI.head_hashes(self.q1, "critic", "SAC"),
            "q2": PI.head_hashes(self.q2, "critic", "SAC"),
        }
        if copy_actor:
            manifest["tensors"]["actor"] = PI.copy_trunk_(
                self.actor, source["actor"], "actor")
        if copy_critics:
            manifest["tensors"]["q1"] = PI.copy_trunk_(self.q1, source["q1"],
                                                       "q1")
            manifest["tensors"]["q2"] = PI.copy_trunk_(self.q2, source["q2"],
                                                       "q2")
        self.hard_update_targets()
        heads_after = {
            "actor": PI.head_hashes(self.actor, "actor", "SAC"),
            "q1": PI.head_hashes(self.q1, "critic", "SAC"),
            "q2": PI.head_hashes(self.q2, "critic", "SAC"),
        }
        manifest["n_tensors"] = sum(len(v) for v in
                                    manifest["tensors"].values())
        manifest["heads_unchanged"] = bool(heads_before == heads_after)
        manifest["head_sha256"] = heads_after
        manifest["targets"] = "hard-copied from the online critics after load"
        manifest["optimizer_state_transferred"] = False
        manifest["replay_transferred"] = False
        manifest["log_alpha_transferred"] = False
        manifest["rng_transferred"] = False
        manifest["actor_warmup_updates"] = 0
        return manifest

    # -------------------------------------------------------------- behaviour
    def act(self, state: torch.Tensor, noise: torch.Tensor,
            w=None) -> torch.Tensor:
        """Stochastic rollout action from externally drawn Gaussian noise."""
        with torch.no_grad():
            a, _ = self.actor.rsample(state, noise, w)
        return a

    def deterministic(self, state: torch.Tensor, w=None) -> torch.Tensor:
        """``tanh(mu)`` forward only; no RNG is consumed (gate 4)."""
        with torch.no_grad():
            return self.actor.deterministic(state, w)

    # ------------------------------------------------------------------ learn
    def update(self, batch: dict, noise_target: torch.Tensor,
               noise_actor: torch.Tensor) -> torch.Tensor:
        """One locked SAC update; returns the ``STAT_NAMES`` stats on device.

        ``batch`` carries ``s, a, r, s2, done`` and, for a goal-conditioned run,
        ``w`` (the goal column shared by ``s`` and ``s2`` of that transition).
        Both Gaussian noise tensors are supplied by the caller.
        """
        s, a, r, s2, d = (batch["s"], batch["a"], batch["r"], batch["s2"],
                          batch["done"])
        w = batch.get("w")
        r = r * self.reward_scale

        # 1) twin critic targets ------------------------------------------------
        with torch.no_grad():
            na, nlogp = self.actor.rsample(s2, noise_target, w)
            min_qt = torch.min(self.q1_target(s2, na, w),
                               self.q2_target(s2, na, w)) - self.alpha * nlogp
            y = r + (1.0 - d) * self.gamma * min_qt

        q1 = self.q1(s, a, w)
        q2 = self.q2(s, a, w)
        q1_loss = F.mse_loss(q1, y)
        q2_loss = F.mse_loss(q2, y)
        self.q1_opt.zero_grad(set_to_none=True)
        q1_loss.backward()
        self.q1_opt.step()
        self.q2_opt.zero_grad(set_to_none=True)
        q2_loss.backward()
        self.q2_opt.step()

        # 2) actor --------------------------------------------------------------
        for p in self._critic_params:
            p.requires_grad_(False)
        try:
            a_new, logp = self.actor.rsample(s, noise_actor, w)
            min_q_pi = torch.min(self.q1(s, a_new, w), self.q2(s, a_new, w))
            actor_loss = (self.alpha.detach() * logp - min_q_pi).mean()
            self.actor_opt.zero_grad(set_to_none=True)
            actor_loss.backward()
            self.actor_opt.step()
        finally:
            for p in self._critic_params:
                p.requires_grad_(True)

        # 3) temperature --------------------------------------------------------
        alpha_loss = -(self.log_alpha
                       * (logp + self.target_entropy).detach()).mean()
        self.alpha_opt.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_opt.step()

        # 4) polyak target critics ---------------------------------------------
        self.soft_update_targets()
        self.n_updates += 1

        return torch.stack([
            q1_loss.detach(), q2_loss.detach(), actor_loss.detach(),
            alpha_loss.detach(), self.alpha.detach().reshape(()),
            (-logp).mean().detach(), q1.mean().detach(), y.mean().detach(),
        ])

    @torch.no_grad()
    def soft_update_targets(self) -> None:
        """``theta_target <- (1 - tau) * theta_target + tau * theta``."""
        for src, tgt in ((self._q1_params, self._q1t_params),
                         (self._q2_params, self._q2t_params)):
            torch._foreach_mul_(tgt, 1.0 - self.tau)
            torch._foreach_add_(tgt, src, alpha=self.tau)

    # ------------------------------------------------------------------- audit
    def lr_report(self) -> dict:
        """Actual optimizer param-group learning rates (gate 4, start-up check)."""
        return {
            "actor": [g["lr"] for g in self.actor_opt.param_groups],
            "q1": [g["lr"] for g in self.q1_opt.param_groups],
            "q2": [g["lr"] for g in self.q2_opt.param_groups],
            "alpha": [g["lr"] for g in self.alpha_opt.param_groups],
            "expected": {"actor": ACTOR_LR, "critic": CRITIC_LR,
                         "alpha": ALPHA_LR},
        }

    def optimizer_report(self) -> dict:
        """Optimizer flags, trainable-parameter list and step counters."""
        def groups(opt):
            g = opt.param_groups[0]
            return {"lr": g["lr"], "betas": list(g["betas"]), "eps": g["eps"],
                    "weight_decay": g["weight_decay"],
                    "amsgrad": bool(g["amsgrad"]),
                    "foreach": g.get("foreach"), "fused": g.get("fused"),
                    "n_tensors": sum(len(gr["params"])
                                     for gr in opt.param_groups),
                    "n_state_entries": len(opt.state)}

        return {
            "backend": self.adam_backend,
            "actor": groups(self.actor_opt),
            "q1": groups(self.q1_opt),
            "q2": groups(self.q2_opt),
            "alpha": groups(self.alpha_opt),
            "n_updates": self.n_updates,
            "trainable": {
                "actor": [n for n, p in self.actor.named_parameters()
                          if p.requires_grad],
                "q1": [n for n, p in self.q1.named_parameters()
                       if p.requires_grad],
                "q2": [n for n, p in self.q2.named_parameters()
                       if p.requires_grad],
            },
        }

    def networks(self) -> dict:
        return {"actor": self.actor, "q1": self.q1, "q2": self.q2,
                "q1_target": self.q1_target, "q2_target": self.q2_target}

    def cpu_state(self) -> dict:
        """Immutable CPU clone of every online and target network plus alpha."""
        out = {}
        for name, net in self.networks().items():
            out[name] = {k: v.detach().cpu().clone()
                         for k, v in net.state_dict().items()}
        out["log_alpha"] = self.log_alpha.detach().cpu().clone()
        return out

    def actor_state(self) -> dict:
        return {k: v.detach().cpu().clone()
                for k, v in self.actor.state_dict().items()}

    def critic_states(self) -> dict:
        return {
            "q1": {k: v.detach().cpu().clone()
                   for k, v in self.q1.state_dict().items()},
            "q2": {k: v.detach().cpu().clone()
                   for k, v in self.q2.state_dict().items()},
            "q1_target": {k: v.detach().cpu().clone()
                          for k, v in self.q1_target.state_dict().items()},
            "q2_target": {k: v.detach().cpu().clone()
                          for k, v in self.q2_target.state_dict().items()},
            "log_alpha": self.log_alpha.detach().cpu().clone(),
        }

    def optimizer_state(self) -> dict:
        return {"actor_opt": self.actor_opt.state_dict(),
                "q1_opt": self.q1_opt.state_dict(),
                "q2_opt": self.q2_opt.state_dict(),
                "alpha_opt": self.alpha_opt.state_dict(),
                "n_updates": self.n_updates}

    def load_optimizer_state(self, sd: dict) -> None:
        self.actor_opt.load_state_dict(sd["actor_opt"])
        self.q1_opt.load_state_dict(sd["q1_opt"])
        self.q2_opt.load_state_dict(sd["q2_opt"])
        self.alpha_opt.load_state_dict(sd["alpha_opt"])
        self.n_updates = int(sd["n_updates"])

    def load_cpu_state(self, blob: dict) -> None:
        for name, net in self.networks().items():
            net.load_state_dict({k: v.to(self.device)
                                 for k, v in blob[name].items()})
        with torch.no_grad():
            self.log_alpha.copy_(blob["log_alpha"].to(self.device))


__all__ = ["SACAgent", "make_adam", "ADAM_BACKENDS", "STAT_NAMES", "GAMMA",
           "TAU", "BATCH_SIZE", "ACTOR_LR", "CRITIC_LR", "ALPHA_LR",
           "INITIAL_ALPHA", "TARGET_ENTROPY", "REWARD_SCALE", "UPDATE_ORDER"]
