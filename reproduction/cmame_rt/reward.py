"""Canonical reward and design metrics (NumPy reference plus a torch batch).

Experimental settings are defined in spec/protocol.json.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from .protocol import reward_cfg

_R = reward_cfg()

#: Canonical reporting coefficients.
C1_CANONICAL: float = float(_R["C1"])
C2_CANONICAL: float = float(_R["C2"])
N_PROBES: int = 9
N_CELLS: int = 30

#: Goals admissible for the reward (1-based, as in the manuscript).
VALID_GOALS = tuple(range(2, 9))


def _check(u: np.ndarray, a: np.ndarray, goal: int, C2: float) -> None:
    if u.shape != (N_PROBES,) or a.shape != (N_CELLS,):
        raise ValueError("invalid displacement/actions")
    if not np.isfinite(u).all() or not np.isfinite(a).all():
        raise ValueError("invalid displacement/actions")
    if goal not in VALID_GOALS or C2 < 0 or (np.abs(a) > 1).any():
        raise ValueError("invalid goal/coefficient/actions")


def normalized_thickness(actions: Sequence) -> float:
    """Return ``rho = mean((a + 1) / 2)``, the normalised thickness fraction."""
    a = np.asarray(actions, dtype=np.float64)
    return float(np.mean((a + 1) / 2))


def kappa(displacements: Sequence, goal: int):
    """Goal displacement divided by the arithmetic mean of its two neighbours.

    Returns ``None`` when the denominator is exactly zero; no epsilon is added
    (protocol.evaluation.kappa).
    """
    u = np.asarray(displacements, dtype=np.float64)
    i = goal - 1
    neighbor_mean = float((u[i - 1] + u[i + 1]) / 2)
    return float(u[i] / neighbor_mean) if neighbor_mean != 0 else None


def peak_success(displacements: Sequence, goal: int) -> bool:
    """True when the goal probe strictly exceeds all eight others (tie fails)."""
    u = np.asarray(displacements, dtype=np.float64)
    i = goal - 1
    return bool(u[i] > np.max(np.delete(u, i)))


def bilateral_margin(displacements: Sequence, goal: int) -> float:
    """Minimum of the two neighbour displacement differences, mm."""
    u = np.asarray(displacements, dtype=np.float64)
    i = goal - 1
    return float(min(u[i] - u[i - 1], u[i] - u[i + 1]))


def contrast_mm(displacements: Sequence, goal: int) -> float:
    """Goal displacement minus the mean of its two neighbours, mm."""
    u = np.asarray(displacements, dtype=np.float64)
    i = goal - 1
    return float(u[i] - (u[i - 1] + u[i + 1]) / 2)


def reward(displacements: Sequence, actions: Sequence, goal: int,
           C1: float = C1_CANONICAL, C2: float = C2_CANONICAL) -> float:
    """Scalar terminal reward in float64 (see module docstring)."""
    u = np.asarray(displacements, dtype=np.float64)
    a = np.asarray(actions, dtype=np.float64)
    _check(u, a, goal, C2)
    i = goal - 1
    rho = float(np.mean((a + 1) / 2))
    return float(u[i] - C1 * (u[i - 1] + u[i + 1])) / (0.5 + C2 * rho)


def design_metrics(displacements: Sequence, actions: Sequence, goal: int,
                   C1: float = C1_CANONICAL,
                   C2: float = C2_CANONICAL) -> dict:
    """Full metric record for one completed design.

    Bit-for-bit equivalent to ``spec/reference_math.py:design_metrics`` and
    verified against it in the surrogate gates.
    """
    u = np.asarray(displacements, dtype=np.float64)
    a = np.asarray(actions, dtype=np.float64)
    _check(u, a, goal, C2)
    i = goal - 1
    rho = float(np.mean((a + 1) / 2))
    contrast = float(u[i] - (u[i - 1] + u[i + 1]) / 2)
    numerator = float(u[i] - C1 * (u[i - 1] + u[i + 1]))
    denom = 0.5 + C2 * rho
    neighbor_mean = float((u[i - 1] + u[i + 1]) / 2)
    return {
        "reward": numerator / denom,
        "contrast_mm": contrast,
        "kappa": float(u[i] / neighbor_mean) if neighbor_mean != 0 else None,
        "normalized_thickness": rho,
        "bilateral_margin_mm": float(min(u[i] - u[i - 1], u[i] - u[i + 1])),
        "peak_success": bool(u[i] > np.max(np.delete(u, i))),
    }


def reward_torch(u9, actions, goal: Any, C1: float = C1_CANONICAL,
                 C2: float = C2_CANONICAL):
    """Batched float32 terminal reward, shape ``[B]``.

    ``u9`` is ``[B,9]`` in mm, ``actions`` is ``[B,30]`` in [-1,1] and ``goal``
    is either a python int or an integer tensor ``[B]`` of 1-based goals. The
    result agrees with :func:`reward` to rtol 1e-6 (checked in the gates).
    """
    import torch

    if u9.ndim != 2 or u9.shape[1] != N_PROBES:
        raise ValueError("u9 must have shape (B,9)")
    if actions.ndim != 2 or actions.shape[1] != N_CELLS:
        raise ValueError("actions must have shape (B,30)")
    if u9.shape[0] != actions.shape[0]:
        raise ValueError("batch mismatch between u9 and actions")
    u = u9.to(torch.float32)
    a = actions.to(torch.float32)
    b = u.shape[0]
    if isinstance(goal, int):
        idx = torch.full((b,), goal - 1, dtype=torch.long, device=u.device)
    else:
        idx = goal.to(torch.long).reshape(-1) - 1
        if idx.shape[0] != b:
            raise ValueError("goal tensor must have shape (B,)")
    if bool(((idx < 1) | (idx > N_PROBES - 2)).any()):
        raise ValueError("invalid goal/coefficient/actions")
    ug = u.gather(1, idx.unsqueeze(1)).squeeze(1)
    ul = u.gather(1, (idx - 1).unsqueeze(1)).squeeze(1)
    ur = u.gather(1, (idx + 1).unsqueeze(1)).squeeze(1)
    rho = ((a + 1) / 2).mean(dim=1)
    return (ug - C1 * (ul + ur)) / (0.5 + C2 * rho)


def peak_success_torch(u9, goal: Any):
    """Batched strict-peak indicator, shape ``[B]`` bool."""
    import torch

    u = u9.to(torch.float32)
    b = u.shape[0]
    if isinstance(goal, int):
        idx = torch.full((b,), goal - 1, dtype=torch.long, device=u.device)
    else:
        idx = goal.to(torch.long).reshape(-1) - 1
    ug = u.gather(1, idx.unsqueeze(1))
    masked = u.masked_fill(
        torch.nn.functional.one_hot(idx, N_PROBES).to(torch.bool),
        float("-inf"))
    return (ug.squeeze(1) > masked.max(dim=1).values)


def bilateral_margin_torch(u9, goal: Any):
    """Batched minimum neighbour difference, shape ``[B]`` float32 (mm)."""
    import torch

    u = u9.to(torch.float32)
    b = u.shape[0]
    if isinstance(goal, int):
        idx = torch.full((b,), goal - 1, dtype=torch.long, device=u.device)
    else:
        idx = goal.to(torch.long).reshape(-1) - 1
    ug = u.gather(1, idx.unsqueeze(1)).squeeze(1)
    ul = u.gather(1, (idx - 1).unsqueeze(1)).squeeze(1)
    ur = u.gather(1, (idx + 1).unsqueeze(1)).squeeze(1)
    return torch.minimum(ug - ul, ug - ur)


def goal_coordinate(goal: Any):
    """Goal coordinate ``omega = (g - 5) / 2`` used by the shared policy."""
    if isinstance(goal, (int, np.integer)):
        return (float(goal) - 5.0) / 2.0
    import torch

    if isinstance(goal, torch.Tensor):
        return (goal.to(torch.float32) - 5.0) / 2.0
    return (np.asarray(goal, dtype=np.float64) - 5.0) / 2.0


__all__ = ["C1_CANONICAL", "C2_CANONICAL", "VALID_GOALS", "reward",
           "design_metrics", "kappa", "peak_success", "bilateral_margin",
           "contrast_mm", "normalized_thickness", "reward_torch",
           "peak_success_torch", "bilateral_margin_torch", "goal_coordinate"]
