"""Scheduled deterministic evaluation, checkpoint selection and early AUC.

Experimental settings are defined in spec/protocol.json.
"""
from __future__ import annotations

import random
from typing import Optional, Sequence

import numpy as np
import torch

from .protocol import load_protocol
from .reward import (C1_CANONICAL, C2_CANONICAL, bilateral_margin, kappa,
                     peak_success, material_weight)

_PROTO = load_protocol()
_EVAL = _PROTO["evaluation"]

#: Scheduled evaluation episodes, 0, 25, ..., 1500.
EVAL_EPISODES: tuple = tuple(int(e) for e in _EVAL["episodes"])
N_EVALS: int = len(EVAL_EPISODES)
REPEATS_PER_GOAL: int = int(_EVAL["repeats_per_goal_per_checkpoint"])
EARLY_AUC_MAX_EPISODE: int = 300
SELECTION_RULE: str = str(_EVAL["checkpoint_selection"])

if EVAL_EPISODES != tuple(range(0, 1501, 25)):  # pragma: no cover
    raise RuntimeError("protocol.evaluation.episodes is not 0,25,...,1500")


# --------------------------------------------------------------- RNG auditing
def rng_snapshot(generators: Optional[dict] = None) -> dict:
    """Capture every RNG state that training depends on.

    ``generators`` maps a stream name to a ``torch.Generator`` owned by the run;
    their states are captured alongside the global Python, NumPy and torch
    states so that a missing generator cannot hide a leak.
    """
    np_state = np.random.get_state()
    snap = {
        "python": repr(random.getstate()),
        "numpy": (np_state[0], np_state[1].tobytes(), np_state[2],
                  np_state[3], np_state[4]),
        "torch_cpu": torch.get_rng_state().clone(),
        "torch_cuda": ([s.clone() for s in torch.cuda.get_rng_state_all()]
                       if torch.cuda.is_available() else []),
        "generators": {},
    }
    for name, g in (generators or {}).items():
        snap["generators"][name] = g.get_state().clone()
    return snap


def rng_equal(a: dict, b: dict) -> dict:
    """Field-by-field comparison of two RNG snapshots."""
    out = {
        "python": a["python"] == b["python"],
        "numpy": a["numpy"] == b["numpy"],
        "torch_cpu": bool(torch.equal(a["torch_cpu"], b["torch_cpu"])),
        "torch_cuda": (len(a["torch_cuda"]) == len(b["torch_cuda"])
                       and all(bool(torch.equal(x, y)) for x, y
                               in zip(a["torch_cuda"], b["torch_cuda"]))),
    }
    gens = {}
    for name in sorted(set(a["generators"]) | set(b["generators"])):
        ga, gb = a["generators"].get(name), b["generators"].get(name)
        gens[name] = bool(ga is not None and gb is not None
                          and torch.equal(ga, gb))
    out["generators"] = gens
    out["all_equal"] = bool(out["python"] and out["numpy"]
                            and out["torch_cpu"] and out["torch_cuda"]
                            and all(gens.values()))
    return out


def training_fingerprint(agent, replay=None) -> dict:
    """Quantities an evaluation must not change: optimizer, alpha, replay, BN."""
    fp = {
        "n_updates": int(getattr(agent, "n_updates", 0)),
        "optimizer_steps": {},
        "log_alpha": None,
        "replay_size": None if replay is None else int(len(replay)),
        "replay_pos": None if replay is None else int(replay.position),
        "module_training_flags": {},
    }
    for name in ("actor_opt", "q1_opt", "q2_opt", "alpha_opt", "critic_opt"):
        opt = getattr(agent, name, None)
        if opt is None:
            continue
        steps = []
        for group in opt.param_groups:
            for p in group["params"]:
                st = opt.state.get(p, {})
                if "step" in st:
                    s = st["step"]
                    steps.append(float(s.item()) if torch.is_tensor(s)
                                 else float(s))
        fp["optimizer_steps"][name] = steps
    if hasattr(agent, "log_alpha"):
        fp["log_alpha"] = float(agent.log_alpha.detach().cpu())
    for name, net in agent.networks().items():
        fp["module_training_flags"][name] = bool(net.training)
    return fp


def assert_unchanged(before: dict, after: dict, label: str = "evaluation"
                     ) -> dict:
    """Raise when an evaluation changed any training-side quantity."""
    problems = [k for k in before if before[k] != after[k]]
    if problems:
        raise AssertionError("%s changed %s" % (label, problems))
    return {"unchanged": True, "checked": sorted(before)}


# ------------------------------------------------------------------- rollouts
@torch.no_grad()
def deterministic_rollout(agent, env, goals: Sequence[int]) -> dict:
    """One deterministic rollout per goal from the empty lattice.

    ``env.batch`` must equal ``len(goals)``; the goals are laid out along the
    batch dimension so the five shared rollouts run as one batched pass
    (SPEED_AND_FINAL_CHECK_ko.md, "shared 평가 batching").
    """
    g = list(int(x) for x in goals)
    if env.batch != len(g):
        raise ValueError("eval env batch %d does not match %d goals"
                         % (env.batch, len(g)))
    state = env.reset(g)
    shared = agent.shared
    for _ in range(env_steps(env)):
        w = env.omega() if shared else None
        action = agent.deterministic(state, w)
        state, reward, done, info = env.step(action)
    return {"goals": g, "u9": info["u9"], "actions30": info["actions30"],
            "terminal_reward": reward}


def env_steps(env) -> int:
    from .env import STEPS_PER_EPISODE

    return STEPS_PER_EPISODE


def evaluate_checkpoint(agent, env, goals: Sequence[int], episode: int,
                        C1: float, C2: float,
                        objective: str = "arithmetic") -> list:
    """Run the scheduled evaluation and build one record per goal."""
    roll = deterministic_rollout(agent, env, goals)
    u9 = roll["u9"].detach().cpu().numpy().astype(np.float64)
    acts = roll["actions30"].detach().cpu().numpy().astype(np.float64)
    records = []
    for i, g in enumerate(roll["goals"]):
        u, a = u9[i], acts[i]
        rho = float(np.mean((a + 1) / 2))
        j = g - 1
        if objective == "bilateral":
            train_obj = (float(min(u[j] - u[j - 1], u[j] - u[j + 1]))
                         / (0.5 + rho))
        elif objective == "arithmetic":
            train_obj = (float(u[j] - C1 * (u[j - 1] + u[j + 1]))
                         / material_weight(C2, rho))
        else:
            raise ValueError("objective must be 'arithmetic' or 'bilateral'")
        canon = (float(u[j] - C1_CANONICAL * (u[j - 1] + u[j + 1]))
                 / (0.5 + C2_CANONICAL * rho))
        records.append({
            "episode": int(episode),
            "goal": int(g),
            "actions30": [float(x) for x in a],
            "u9_pred": [float(x) for x in u],
            "training_objective": train_obj,
            "selection_score": train_obj,
            "canonical_reward": canon,
            "peak_success": bool(peak_success(u, g)),
            "bilateral_margin": float(bilateral_margin(u, g)),
            "kappa": kappa(u, g),
            "normalized_thickness": rho,
        })
    return records


# ------------------------------------------------------------------- analysis
def early_auc(episodes: Sequence[int], rewards: Sequence[float]) -> float:
    """Trapezoidal integral of the canonical reward over 0..300, divided by 300.

    Same definition as ``spec/reference_math.py:early_auc``: the observations
    0, 25, ..., 300 must each appear exactly once. Training-reward averages,
    smoothing, a 25-300 window and a denominator of 299 are all wrong.
    """
    e = np.asarray(episodes, dtype=np.int64)
    r = np.asarray(rewards, dtype=np.float64)
    if e.shape != r.shape or not np.isfinite(r).all():
        raise ValueError("invalid curve")
    mask = (e >= 0) & (e <= EARLY_AUC_MAX_EPISODE)
    if not np.array_equal(e[mask], np.arange(0, EARLY_AUC_MAX_EPISODE + 1, 25)):
        raise ValueError("required observations: 0,25,...,300, exactly once")
    return float(np.sum(np.diff(e[mask])
                        * (r[mask][:-1] + r[mask][1:]) / 2)
                 / EARLY_AUC_MAX_EPISODE)


def select_checkpoint(episodes: Sequence[int],
                      selection_scores, strict: bool = True) -> int:
    """Episode with the maximum mean training objective; earliest exact tie.

    ``selection_scores`` is ``[n_episodes]`` or ``[n_episodes, n_goals]``; a
    goal-conditioned run selects ONE checkpoint from the five-goal mean.
    Episode 0 is a legitimate winner.

    ``strict=False`` accepts a shortened episode grid and exists only for the
    ``pilot/`` smoke runs; every production run keeps the full 0..1500 schedule.
    """
    e = np.asarray(episodes, dtype=np.int64)
    s = np.asarray(selection_scores, dtype=np.float64)
    if strict and not np.array_equal(e, np.asarray(EVAL_EPISODES,
                                                   dtype=np.int64)):
        raise ValueError("required observations: 0,25,...,1500")
    if not strict and (e.ndim != 1 or e.size == 0
                       or not np.array_equal(e, np.sort(e))
                       or np.unique(e).size != e.size):
        raise ValueError("episodes must be strictly increasing and unique")
    if s.ndim == 1:
        s = s[:, None]
    if s.ndim != 2 or s.shape[0] != len(e) or not np.isfinite(s).all():
        raise ValueError("invalid checkpoint scores")
    k = int(np.argmax(np.mean(s, axis=1)))
    return int(e[k])


def curve_from_records(records: Sequence[dict], field: str) -> dict:
    """Collapse per-goal records into ``{episode: mean over goals}`` plus goals."""
    by_ep: dict = {}
    for r in records:
        by_ep.setdefault(int(r["episode"]), {})[int(r["goal"])] = float(r[field])
    episodes = sorted(by_ep)
    goals = sorted({int(r["goal"]) for r in records})
    mean = [float(np.mean([by_ep[e][g] for g in goals])) for e in episodes]
    per_goal = {g: [by_ep[e][g] for e in episodes] for g in goals}
    return {"episodes": episodes, "mean": mean, "per_goal": per_goal,
            "goals": goals}


def early_auc_report(records: Sequence[dict]) -> dict:
    """Early AUC of the canonical reward: mean curve plus one value per goal."""
    c = curve_from_records(records, "canonical_reward")
    out = {"definition": _EVAL["early_auc"],
           "window": [0, EARLY_AUC_MAX_EPISODE],
           "mean_curve": early_auc(c["episodes"], c["mean"]),
           "per_goal": {str(g): early_auc(c["episodes"], c["per_goal"][g])
                        for g in c["goals"]}}
    return out


def selection_report(records: Sequence[dict], strict: bool = True) -> dict:
    """Apply the checkpoint rule to the per-goal selection scores."""
    c = curve_from_records(records, "selection_score")
    scores = np.asarray([[c["per_goal"][g][i] for g in c["goals"]]
                         for i in range(len(c["episodes"]))], dtype=np.float64)
    chosen = select_checkpoint(c["episodes"], scores, strict=strict)
    return {"selected_episode": int(chosen),
            "strict_schedule": bool(strict),
            "rule": "max mean training-objective over goal set, earliest tie",
            "protocol_rule": SELECTION_RULE,
            "goals": [int(g) for g in c["goals"]],
            "episodes": [int(e) for e in c["episodes"]],
            "scores": scores.tolist(),
            "mean_scores": np.mean(scores, axis=1).tolist()}


__all__ = ["EVAL_EPISODES", "N_EVALS", "REPEATS_PER_GOAL", "SELECTION_RULE",
           "rng_snapshot", "rng_equal", "training_fingerprint",
           "assert_unchanged", "deterministic_rollout", "evaluate_checkpoint",
           "early_auc", "select_checkpoint", "curve_from_records",
           "early_auc_report", "selection_report"]
