"""Manuscript Eqs. (3)-(4) with the coefficients C1, C2 left free.

    numerator = u_i - C1*(u_(i-1) + u_(i+1))

    if C2 == 0:  W = 1                 <-- SEPARATE no-density branch
    else:        W = 0.5 + C2*rho_A

    reward = numerator / W
    rho_A  = mean((action + 1)/2)

``C2 == 0`` is NOT ``0.5 + 0*rho_A``. The manuscript defines it as a distinct
no-density reference with ``W = 1``; substituting zero would give ``W = 0.5`` and
double every reward. The branch is pinned by a unit test.

At ``(C1, C2) = (0.5, 1.0)`` this reproduces the original locked reward exactly,
because ``0.5 + 1.0*rho_A == mean((action + 2)/2)`` (verified numerically).

Because the reward FORMULA and its SCALE change with the coefficients, raw reward
is never used to compare coefficient cells. It is only the learning signal and the
checkpoint-selection criterion inside one run. Cross-cell comparison uses the
scale-free ``kappa`` and the physical margins below.
"""
from __future__ import annotations

import numpy as np

TASK_INDICES = {
    "FGL5": {"target": 4, "lower": 3, "upper": 5},
    "FGL6": {"target": 5, "lower": 4, "upper": 6},
    "FGL7": {"target": 6, "lower": 5, "upper": 7},
}
ORIGINAL_C1 = 0.5
ORIGINAL_C2 = 1.0


def rho_a(action) -> float:
    """rho_A = mean((action + 1)/2). Equals density_code - 0.5."""
    a = np.asarray(action, dtype=np.float64).reshape(-1)
    return float(np.mean((a + 1.0) / 2.0))


def weight_W(c2: float, rho: float) -> float:
    """W = 1 when C2 == 0 (no-density reference); else 0.5 + C2*rho_A."""
    return 1.0 if float(c2) == 0.0 else 0.5 + float(c2) * float(rho)


def numerator(u, task: str, c1: float) -> float:
    ix = TASK_INDICES[task]
    u = np.asarray(u, dtype=np.float64).reshape(-1)
    return float(u[ix["target"]] - float(c1) * (u[ix["lower"]] + u[ix["upper"]]))


def reward(u, action, task: str, c1: float, c2: float) -> float:
    return numerator(u, task, c1) / weight_W(c2, rho_a(action))


def kappa(u, task: str) -> float:
    """Scale-free localisation ratio u_i / (0.5*(u_(i-1)+u_(i+1))).

    Independent of C1 and C2, so it is comparable ACROSS coefficient cells and is
    the primary selection metric.
    """
    ix = TASK_INDICES[task]
    u = np.asarray(u, dtype=np.float64).reshape(-1)
    den = 0.5 * (u[ix["lower"]] + u[ix["upper"]])
    return float(u[ix["target"]] / den) if den != 0 else float("nan")


def make_reward_fn(task: str, c1: float, c2: float):
    """Return f(u1..u9) -> numerator, for the Environment's reward_function slot.

    The environment divides by its own W, so this supplies only the numerator.
    """
    ix = TASK_INDICES[task]
    c1 = float(c1)

    def f(*u):
        return u[ix["target"]] - c1 * (u[ix["lower"]] + u[ix["upper"]])
    return f


def physical_metrics(u, action, task: str, c1: float, c2: float) -> dict:
    """Every per-design quantity the spec requires, for one terminal design."""
    u = np.asarray(u, dtype=np.float64).reshape(-1)
    a = np.asarray(action, dtype=np.float64).reshape(-1)
    ix = TASK_INDICES[task]
    ui, ul, uu = float(u[ix["target"]]), float(u[ix["lower"]]), float(u[ix["upper"]])
    rho = rho_a(a)
    m_lower, m_upper = ui - ul, ui - uu
    bilat = float(min(m_lower, m_upper))
    others = np.delete(u, ix["target"])
    return {
        "u": u.tolist(),
        "rho_A": rho,
        "kappa": kappa(u, task),
        "m_lower": m_lower,
        "m_upper": m_upper,
        "bilateral_raw": bilat,
        "bilateral_normalized": bilat / (0.5 + rho),
        "global_margin": float(ui - np.max(others)),
        "target_is_global_peak": bool(ui >= np.max(others)),
        "own_objective_reward": reward(u, a, task, c1, c2),
        "original_locked_reward_at_C1_0.5_C2_1.0":
            reward(u, a, task, ORIGINAL_C1, ORIGINAL_C2),
        "W": weight_W(c2, rho),
        "numerator": numerator(u, task, c1),
        "action": a.tolist(),
    }


# fields stored per episode in physical_metric_history.npz (float32, no smoothing,
# no episode skipped)
HISTORY_SCALARS = ["rho_A", "kappa", "m_lower", "m_upper", "bilateral_raw",
                   "bilateral_normalized", "global_margin", "target_is_global_peak",
                   "own_objective_reward", "original_locked_reward_at_C1_0.5_C2_1.0",
                   "W", "numerator"]


def recompute_at(u, action, task: str, c1: float, c2: float) -> float:
    """Re-score any stored design under a different coefficient pair.

    Used for `common_locked_reward`: every candidate rescored under the FINAL
    (C1*, C2*) so that cells become comparable on one common scale.
    """
    return reward(u, action, task, c1, c2)
