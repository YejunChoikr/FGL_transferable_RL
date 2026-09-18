"""Paper original average-contrast reward + margin diagnostics.

    C1 = 0.5 ; C2 = 1.0
    m_lower   = u_i - u_(i-1)
    m_upper   = u_i - u_(i+1)
    numerator = u_i - 0.5*(u_(i-1) + u_(i+1)) = 0.5*(m_lower + m_upper)
    density_code = mean((action + 2)/2) = 0.5 + rho_A
    reward = numerator / density_code

No hard-min, soft-min, clipping, reward scaling, global-profile penalty or extra
regulariser enters training or the BO objective. ``m_bilateral`` and the global
margin below are DIAGNOSTICS computed after the fact (used for the FEA physical
comparison) — they are never optimised.
"""
from __future__ import annotations

import numpy as np

C1 = 0.5
C2 = 1.0

TASK_INDICES = {
    "FGL5": {"target": 4, "lower": 3, "upper": 5},
    "FGL6": {"target": 5, "lower": 4, "upper": 6},
    "FGL7": {"target": 6, "lower": 5, "upper": 7},
}


def target_u5(u1, u2, u3, u4, u5, u6, u7, u8, u9):
    return C2 * u5 - C1 * (u4 + u6)


def target_u6(u1, u2, u3, u4, u5, u6, u7, u8, u9):
    return C2 * u6 - C1 * (u5 + u7)


def target_u7(u1, u2, u3, u4, u5, u6, u7, u8, u9):
    return C2 * u7 - C1 * (u6 + u8)


REWARDS = {"target_u5": target_u5, "target_u6": target_u6, "target_u7": target_u7}
TASK_TO_REWARD = {"FGL5": "target_u5", "FGL6": "target_u6", "FGL7": "target_u7"}


def density_code(actions) -> float:
    a = np.asarray(actions, dtype=np.float64).reshape(-1)
    return float(np.sum((a + 2.0) / 2.0) / 30.0)


def rho_a(actions) -> float:
    return density_code(actions) - 0.5


def margins(u, task) -> dict:
    """All margin diagnostics for one u1..u9 profile and one task."""
    u = np.asarray(u, dtype=np.float64).reshape(-1)
    ix = TASK_INDICES[task]
    ui, ul, uu = float(u[ix["target"]]), float(u[ix["lower"]]), float(u[ix["upper"]])
    m_lower = ui - ul
    m_upper = ui - uu
    numerator = ui - C1 * (ul + uu)
    others = np.delete(u, ix["target"])
    return {
        "u_target": ui, "u_lower": ul, "u_upper": uu,
        "m_lower": m_lower, "m_upper": m_upper,
        "m_bilateral": float(min(m_lower, m_upper)),
        "numerator": numerator,
        "global_margin": float(ui - np.max(others)),
        "target_is_global_max": bool(ui >= np.max(others)),
    }


def full_metrics(u, actions, task) -> dict:
    """Reward + margins + density for one design, from a u1..u9 profile."""
    d = density_code(actions)
    m = margins(u, task)
    m.update({
        "density_code": d, "rho_A": d - 0.5,
        "reward": m["numerator"] / d,
        "m_bilateral_density_normalized": m["m_bilateral"] / d,
    })
    return m
