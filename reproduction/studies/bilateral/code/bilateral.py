"""Locked bilateral hard-min objective and complete design diagnostics."""
from __future__ import annotations

import hashlib
from typing import Iterable

import numpy as np

from original_reward import full_metrics as original_full_metrics

TASK_INDICES = {
    "FGL5": {"target": 4, "left": 3, "right": 5},
    "FGL6": {"target": 5, "left": 4, "right": 6},
    "FGL7": {"target": 6, "left": 5, "right": 7},
}


def _raw(u: Iterable[float], task: str) -> float:
    values = np.asarray(list(u), dtype=np.float64).reshape(-1)
    ix = TASK_INDICES[task]
    left = values[ix["target"]] - values[ix["left"]]
    right = values[ix["target"]] - values[ix["right"]]
    return float(min(left, right))


def target_u5(*u):
    return _raw(u, "FGL5")


def target_u6(*u):
    return _raw(u, "FGL6")


def target_u7(*u):
    return _raw(u, "FGL7")


REWARDS = {"FGL5": target_u5, "FGL6": target_u6, "FGL7": target_u7}


def density_code(actions: Iterable[float]) -> float:
    a = np.asarray(list(actions), dtype=np.float64).reshape(-1)
    if a.size != 30:
        raise ValueError(f"expected 30 actions, received {a.size}")
    return float(np.mean((a + 2.0) / 2.0))


def action_sha256(actions: Iterable[float]) -> str:
    a = np.asarray(list(actions), dtype=np.float64).reshape(-1)
    return hashlib.sha256("|".join(f"{x:.12g}" for x in a).encode()).hexdigest()


def metrics(u: Iterable[float], actions: Iterable[float], task: str) -> dict:
    """Return every locked bilateral and original-reward diagnostic."""
    values = np.asarray(list(u), dtype=np.float64).reshape(-1)
    a = np.asarray(list(actions), dtype=np.float64).reshape(-1)
    if values.size != 9 or a.size != 30:
        raise ValueError(f"expected u=9/actions=30, got {values.size}/{a.size}")
    ix = TASK_INDICES[task]
    ui = float(values[ix["target"]])
    ul = float(values[ix["left"]])
    ur = float(values[ix["right"]])
    left = ui - ul
    right = ui - ur
    bilateral = min(left, right)
    d = density_code(a)
    original = original_full_metrics(values, a, task)
    global_ix = int(np.argmax(values))
    other = np.delete(values, ix["target"])
    neighbor_mean = 0.5 * (ul + ur)
    return {
        "u": [float(x) for x in values],
        "u_target": ui,
        "u_left": ul,
        "u_right": ur,
        "left_margin": float(left),
        "right_margin": float(right),
        "bilateral_margin": float(bilateral),
        "bilateral_reward": float(bilateral / d),
        "original_reward": float(original["reward"]),
        "kappa": float(ui / neighbor_mean) if neighbor_mean != 0 else None,
        "rho_A": float(d - 0.5),
        "density_code": float(d),
        "local_peak": bool(left > 0 and right > 0),
        "global_peak": bool(ui >= np.max(other)),
        "global_peak_index": global_ix + 1,
        "global_margin": float(ui - np.max(other)),
        "left_right_asymmetry": float(abs(left - right)),
        "left_minus_right": float(left - right),
        "action_sha256": action_sha256(a),
    }
