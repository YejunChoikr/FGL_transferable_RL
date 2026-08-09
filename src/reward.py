"""Terminal reward and the margin quantities reported in the paper.

The reward is assigned only at the terminal step of an episode:

    numerator = u_i - C1 * (u_{i-1} + u_{i+1})
    W         = 1                       for C2 = 0
              = 0.5 + C2 * t_bar_star   otherwise
    reward    = numerator / W

where u_i is the horizontal displacement at the target probe, u_{i-1} and
u_{i+1} those at the adjacent probes, and t_bar_star the normalized mean member
thickness of the design. No smoothing, clipping or additional penalty is
applied.
"""
from __future__ import annotations

import numpy as np

C1 = 0.5
C2 = 1.0

N_CELLS = 30

TASK_INDICES = {
    "FGL5": {"target": 4, "lower": 3, "upper": 5},
    "FGL6": {"target": 5, "lower": 4, "upper": 6},
    "FGL7": {"target": 6, "lower": 5, "upper": 7},
}


def target_u5(u1, u2, u3, u4, u5, u6, u7, u8, u9):
    return u5 - C1 * (u4 + u6)


def target_u6(u1, u2, u3, u4, u5, u6, u7, u8, u9):
    return u6 - C1 * (u5 + u7)


def target_u7(u1, u2, u3, u4, u5, u6, u7, u8, u9):
    return u7 - C1 * (u6 + u8)


REWARDS = {"target_u5": target_u5, "target_u6": target_u6, "target_u7": target_u7}
TASK_TO_REWARD = {"FGL5": "target_u5", "FGL6": "target_u6", "FGL7": "target_u7"}


def normalized_thickness(actions) -> float:
    """t_bar_star: the mean cell thickness mapped onto [0, 1] over the domain.

    Actions lie on [-1, 1] and map linearly onto the admissible thickness
    range, so (a + 1) / 2 is the normalized thickness of one cell.
    """
    a = np.asarray(actions, dtype=np.float64).reshape(-1)
    return float(np.sum((a + 1.0) / 2.0) / N_CELLS)


def weighting_factor(actions) -> float:
    """W of the reward denominator."""
    if C2 == 0:
        return 1.0
    return 0.5 + C2 * normalized_thickness(actions)


def margins(u, task) -> dict:
    """Margin quantities for one u1..u9 profile and one task."""
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
    """Reward, margins and normalized thickness for one design."""
    w = weighting_factor(actions)
    m = margins(u, task)
    m.update({
        "t_bar_star": normalized_thickness(actions),
        "W": w,
        "reward": m["numerator"] / w,
        "m_bilateral_normalized": m["m_bilateral"] / w,
    })
    return m
