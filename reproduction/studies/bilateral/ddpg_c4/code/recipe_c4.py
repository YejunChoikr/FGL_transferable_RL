"""A4_C4_NATIVE_OPT: the single method change, isolated in one file.

``ddpg_agent.py`` in this project is a BYTE-IDENTICAL copy of the native DDPG
authority's agent. That is deliberate. The authority's module defines

    ACTOR_TRANSFER  = ("fc1", "fc2", "fc3", "fc4")
    CRITIC_TRANSFER = ("fc1", "fc2", "fc3")            # fc4, fc5 stay scratch

and editing it in place would make "did anything else change?" a judgement call
about a diff. Instead the new mask lives here, so the audit statement is exact:
the agent file's SHA-256 equals the authority's, and the one behavioural
difference in this project is the CRITIC_TRANSFER tuple below.

    A4_C3_NATIVE_OPT : actor fc1-fc4 + critic fc1-fc3 = 14 tensors
    A4_C4_NATIVE_OPT : actor fc1-fc4 + critic fc1-fc4 = 16 tensors

Optimizer settings are the NATIVE ones (actor 1e-4, critic 1e-3, WD 0), NOT the
SAC-matched 3e-4/3e-4. Nothing here touches learning rate, batch size or tau.
"""
from __future__ import annotations

import torch

ACTOR_TRANSFER = ("fc1", "fc2", "fc3", "fc4")     # unchanged from A4_C3
CRITIC_TRANSFER = ("fc1", "fc2", "fc3", "fc4")    # <-- THE ONLY CHANGE (was fc1-fc3)
ACTOR_FRESH = ("fc5",)
CRITIC_FRESH = ("fc5",)
N_TRANSFERRED_TENSORS = 2 * len(ACTOR_TRANSFER) + 2 * len(CRITIC_TRANSFER)   # 16

RECIPE_ID = "A4_C4_NATIVE_OPT"
BASELINE_RECIPE_ID = "A4_C3_NATIVE_OPT"


def _load_masked(dst_module, src_sd, layers, tag, moved):
    dst = dst_module.state_dict()
    for k in list(dst):
        if k.split(".")[0] in layers:
            if k not in src_sd:
                raise KeyError(f"source {tag} missing {k}")
            if dst[k].shape != src_sd[k].shape:
                raise ValueError(f"{tag} shape mismatch on {k}: "
                                 f"{tuple(dst[k].shape)} vs {tuple(src_sd[k].shape)}")
            dst[k] = src_sd[k].clone()
            moved.append(f"{tag}:{k}")
    dst_module.load_state_dict(dst)


def apply_a4c4_native(agent, source_sd):
    """Load actor fc1-fc4 and critic fc1-fc4, then hard-copy online -> target.

    Returns the audit record. Optimizer state, replay, OU state, observations,
    rewards and normalization state are never touched -- this function has no
    access to them by construction.
    """
    before = {f"actor:{k}": v.detach().clone()
              for k, v in agent.actor.state_dict().items()}
    before.update({f"critic:{k}": v.detach().clone()
                   for k, v in agent.critic.state_dict().items()})

    moved = []
    _load_masked(agent.actor, source_sd["actor"], ACTOR_TRANSFER, "actor", moved)
    _load_masked(agent.critic, source_sd["critic"], CRITIC_TRANSFER, "critic", moved)
    if len(moved) != N_TRANSFERRED_TENSORS:
        raise AssertionError(f"{RECIPE_ID} moved {len(moved)} tensors, "
                             f"expected {N_TRANSFERRED_TENSORS}: {moved}")

    agent.hard_update()          # targets re-synced AFTER assembly, never before

    changed, unchanged = [], []
    for k, b in before.items():
        net, key = k.split(":", 1)
        now = (agent.actor if net == "actor" else agent.critic).state_dict()[key]
        (changed if not torch.equal(b.cpu(), now.cpu()) else unchanged).append(k)

    heads = ("actor:fc5.weight", "actor:fc5.bias",
             "critic:fc5.weight", "critic:fc5.bias")
    heads_scratch = all(
        torch.equal(before[h].cpu(),
                    (agent.actor if h.startswith("actor") else agent.critic)
                    .state_dict()[h.split(":", 1)[1]].cpu())
        for h in heads)
    targets_equal = (
        all(torch.equal(p.cpu(), q.cpu())
            for p, q in zip(agent.actor.parameters(), agent.actor_target.parameters()))
        and all(torch.equal(p.cpu(), q.cpu())
                for p, q in zip(agent.critic.parameters(), agent.critic_target.parameters())))

    return {
        "recipe_id": RECIPE_ID,
        "baseline_recipe_id": BASELINE_RECIPE_ID,
        "actor_load": list(ACTOR_TRANSFER), "actor_fresh": list(ACTOR_FRESH),
        "critic_load": list(CRITIC_TRANSFER), "critic_fresh": list(CRITIC_FRESH),
        "moved": sorted(moved),
        "n_transferred_tensors": len(moved),
        "expected_n_transferred_tensors": N_TRANSFERRED_TENSORS,
        "tensors_changed": sorted(changed),
        "tensors_unchanged_scratch": sorted(unchanged),
        "fc5_heads_bitwise_unchanged": bool(heads_scratch),
        "targets_bitwise_equal_to_assembled_online": bool(targets_equal),
        "actor_lr": float(agent.actor_opt.param_groups[0]["lr"]),
        "critic_lr": float(agent.critic_opt.param_groups[0]["lr"]),
        "actor_weight_decay": float(agent.actor_opt.param_groups[0]["weight_decay"]),
        "critic_weight_decay": float(agent.critic_opt.param_groups[0]["weight_decay"]),
        "batch_size": int(agent.batch_size),
        "tau": float(agent.tau),
        "optimizer_state_transferred": False,
        "replay_transferred": False,
        "ou_state_transferred": False,
        "replay_len_after_transfer": len(agent.memory),
        "warmup": "none", "freezing": "none",
    }


def optimizer_audit(agent):
    """Prove the NATIVE optimizer settings survived, and that nothing is frozen."""
    def groups(opt):
        return [{"lr": float(g["lr"]), "weight_decay": float(g["weight_decay"]),
                 "n_tensors": len(g["params"]),
                 "n_params": sum(p.numel() for p in g["params"])}
                for g in opt.param_groups]
    return {
        "actor": groups(agent.actor_opt),
        "critic": groups(agent.critic_opt),
        "distinct_actor_lrs": sorted({float(g["lr"]) for g in agent.actor_opt.param_groups}),
        "distinct_critic_lrs": sorted({float(g["lr"]) for g in agent.critic_opt.param_groups}),
        "expected_actor_lr": 1e-4, "expected_critic_lr": 1e-3,
        "expected_batch_size": 64, "expected_tau": 0.001,
        "actual_batch_size": int(agent.batch_size), "actual_tau": float(agent.tau),
        "any_frozen_parameter": any(not p.requires_grad
                                    for p in list(agent.actor.parameters()) +
                                    list(agent.critic.parameters())),
        "sac_matched_values_present": bool(
            any(abs(float(g["lr"]) - 3e-4) < 1e-12
                for g in list(agent.actor_opt.param_groups) +
                list(agent.critic_opt.param_groups))
            or int(agent.batch_size) == 256 or abs(float(agent.tau) - 0.005) < 1e-12),
    }
