"""Deterministic sub-seed derivation for every random stream.

Experimental settings are defined in spec/protocol.json.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .protocol import runtime_cfg

#: Fixed entropy root shared by every stream.
ROOT_ENTROPY: int = 20260913

#: Stream identifiers, read from protocol.runtime.rng_stream_ids.
STREAM_IDS: dict = dict(runtime_cfg()["rng_stream_ids"])


def stream_id(name: str) -> int:
    """Return the protocol stream identifier registered under ``name``."""
    try:
        return int(STREAM_IDS[name])
    except KeyError as exc:
        raise KeyError("unknown rng stream: %r; known: %s"
                       % (name, sorted(STREAM_IDS))) from exc


def subseed(run_seed: int, stream: Any) -> int:
    """Derive the 32-bit sub-seed for ``run_seed`` on one stream.

    ``stream`` accepts either the numeric protocol identifier or its name.
    """
    sid = stream_id(stream) if isinstance(stream, str) else int(stream)
    ss = np.random.SeedSequence([ROOT_ENTROPY, int(run_seed), sid])
    return int(ss.generate_state(1, dtype=np.uint32)[0])


def numpy_gen(seed: int) -> np.random.Generator:
    """Return a NumPy PCG64 generator seeded with ``seed``."""
    return np.random.Generator(np.random.PCG64(int(seed)))


def torch_gen(seed: int, device: Any = "cpu"):
    """Return a torch.Generator on ``device`` seeded with ``seed``."""
    import torch

    g = torch.Generator(device=torch.device(device))
    g.manual_seed(int(seed))
    return g


def stream_subseeds(run_seed: int) -> dict:
    """Return every named stream sub-seed for one run seed (manifest use)."""
    return {name: subseed(run_seed, name) for name in sorted(STREAM_IDS)}


__all__ = ["ROOT_ENTROPY", "STREAM_IDS", "stream_id", "subseed", "numpy_gen",
           "torch_gen", "stream_subseeds"]
