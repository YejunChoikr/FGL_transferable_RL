"""Single shared design encoder (60-dim interleaved action/location state).

Experimental settings are defined in spec/protocol.json.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any, Sequence

import numpy as np

from .protocol import load_location_golden, load_protocol

_GOLDEN = load_location_golden()
_ENC = load_protocol()["encoding"]

#: Golden dimensionless location vector, loaded (not recomputed).
LOCATION: np.ndarray = np.asarray(_GOLDEN["values"], dtype=np.float32)

ROWS: int = int(_ENC["rows"])
COLS: int = int(_ENC["independent_columns"])
N_CELLS: int = ROWS * COLS
STATE_DIM: int = int(_ENC["complete_state_dim"])
CNN_INPUT_SHAPE = tuple(_ENC["cnn_input_shape"][1:])

if LOCATION.shape != (N_CELLS,) or LOCATION.dtype != np.float32:
    raise RuntimeError("location_golden.json does not hold 30 float32 values")
if not bool((LOCATION > 0).all()) or float(LOCATION[-1]) != 1.0:
    raise RuntimeError("golden location vector failed its invariants")

_TRAVERSALS = {k: list(v) for k, v in _ENC["traversal_indices"].items()}
DEFAULT_TRAVERSAL: str = _ENC["default_traversal"]


def traversal(name: Any = None) -> list:
    """Return the cell visiting order registered under ``name``."""
    key = DEFAULT_TRAVERSAL if name is None else name
    try:
        order = _TRAVERSALS[key]
    except KeyError as exc:
        raise KeyError("unknown traversal %r; known: %s"
                       % (key, sorted(_TRAVERSALS))) from exc
    return list(order)


def traversal_names() -> list:
    """Return every traversal name defined by the protocol."""
    return sorted(_TRAVERSALS)


def encode_design(actions: Sequence, assigned: Any = None) -> np.ndarray:
    """Encode normalised actions into the 60-dim state (reference semantics).

    Parameters mirror ``spec/reference_math.py:encode_design``: ``actions`` is a
    finite length-30 vector in [-1, 1] and ``assigned`` marks the cells already
    visited. Unassigned entries contribute ``(0, 0)``.
    """
    a = np.asarray(actions, dtype=np.float32)
    if a.shape != (N_CELLS,) or not np.isfinite(a).all() or (np.abs(a) > 1).any():
        raise ValueError("actions must be a finite length-30 vector in [-1,1]")
    m = (np.ones(N_CELLS, dtype=bool) if assigned is None
         else np.asarray(assigned, dtype=bool))
    if m.shape != (N_CELLS,):
        raise ValueError("assigned must have shape (30,)")
    return np.stack((np.where(m, a, 0), np.where(m, LOCATION, 0)),
                    axis=1).ravel().astype(np.float32)


def encode_batch_numpy(actions: np.ndarray, assigned: Any = None) -> np.ndarray:
    """Vectorised ``encode_design`` over a batch of designs, shape ``(B, 60)``."""
    a = np.asarray(actions, dtype=np.float32)
    if a.ndim != 2 or a.shape[1] != N_CELLS:
        raise ValueError("actions must have shape (B,30)")
    m = (np.ones_like(a, dtype=bool) if assigned is None
         else np.asarray(assigned, dtype=bool))
    if m.shape != a.shape:
        raise ValueError("assigned must have shape (B,30)")
    out = np.empty((a.shape[0], STATE_DIM), dtype=np.float32)
    out[:, 0::2] = np.where(m, a, 0)
    out[:, 1::2] = np.where(m, LOCATION[None, :], 0)
    return out


def physical_thickness(actions: Sequence, t_min: float,
                       t_max: float) -> np.ndarray:
    """Map normalised actions in [-1,1] onto physical thickness in mm."""
    a = np.asarray(actions, dtype=np.float64)
    if not t_min < t_max:
        raise ValueError("invalid bounds")
    return t_min + (a + 1) * (t_max - t_min) / 2


def normalized_action(thickness: Sequence, t_min: float,
                      t_max: float) -> np.ndarray:
    """Inverse of :func:`physical_thickness` (mm to normalised action)."""
    t = np.asarray(thickness, dtype=np.float64)
    if not t_min < t_max:
        raise ValueError("invalid bounds")
    return 2 * (t - t_min) / (t_max - t_min) - 1


@lru_cache(maxsize=8)
def _location_buffer(device_str: str, dtype_str: str):
    import torch

    dtype = getattr(torch, dtype_str)
    return torch.as_tensor(LOCATION.copy(), dtype=dtype,
                           device=torch.device(device_str))


def torch_location_buffer(device: Any = "cpu", dtype: Any = None):
    """Return the golden location vector as a torch tensor on ``device``."""
    import torch

    dtype = torch.float32 if dtype is None else dtype
    name = str(dtype).replace("torch.", "")
    return _location_buffer(str(torch.device(device)), name)


def encode_batch_torch(actions, assigned=None):
    """Encode a torch batch ``actions[B,30]`` into states ``[B,60]`` float32.

    ``assigned`` is a bool tensor of the same shape; ``None`` means every cell
    is assigned. The location buffer follows the device of ``actions``.
    """
    import torch

    if actions.ndim != 2 or actions.shape[1] != N_CELLS:
        raise ValueError("actions must have shape (B,30)")
    a = actions.to(torch.float32)
    loc = torch_location_buffer(a.device, torch.float32).expand_as(a)
    if assigned is None:
        a_eff, loc_eff = a, loc
    else:
        if assigned.shape != a.shape:
            raise ValueError("assigned must have shape (B,30)")
        m = assigned.to(torch.bool)
        zero = torch.zeros((), dtype=torch.float32, device=a.device)
        a_eff = torch.where(m, a, zero)
        loc_eff = torch.where(m, loc, zero)
    out = torch.empty((a.shape[0], STATE_DIM), dtype=torch.float32,
                      device=a.device)
    out[:, 0::2] = a_eff
    out[:, 1::2] = loc_eff
    return out


def to_cnn_input(x60):
    """Reshape ``[B,60]`` states to the CNN input ``[B,1,10,6]`` (reshape only)."""
    if x60.shape[-1] != STATE_DIM:
        raise ValueError("expected a trailing dimension of 60")
    return x60.reshape(-1, 1, ROWS, 2 * COLS)


__all__ = ["LOCATION", "ROWS", "COLS", "N_CELLS", "STATE_DIM",
           "CNN_INPUT_SHAPE", "DEFAULT_TRAVERSAL", "traversal",
           "traversal_names", "encode_design", "encode_batch_numpy",
           "physical_thickness", "normalized_action", "torch_location_buffer",
           "encode_batch_torch", "to_cnn_input"]
