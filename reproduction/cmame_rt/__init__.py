"""cmame_rt: runtime package for the CMAME unified rerun (2026-09-13).

Experimental settings are defined in spec/protocol.json.
"""
from __future__ import annotations

from . import determinism, encoding, paths, protocol, reward, rng
from .protocol import (PROTOCOL_FILE_SHA256, PROTOCOL_HASH, case_by_id,
                       load_protocol)
from .rng import STREAM_IDS, subseed

__version__ = "0.1.0"

__all__ = ["paths", "protocol", "rng", "encoding", "reward", "determinism",
           "PROTOCOL_HASH", "PROTOCOL_FILE_SHA256", "load_protocol",
           "case_by_id", "subseed", "STREAM_IDS", "__version__"]
