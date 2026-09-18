"""Filesystem anchors for the cmame_rt runtime package.

Experimental settings are defined in spec/protocol.json.
"""
from __future__ import annotations

from pathlib import Path

ROOT: Path = Path(__file__).resolve().parent.parent
PKG: Path = ROOT / "cmame_rt"

SPEC: Path = ROOT / "spec"
DATA: Path = ROOT / "data"
LOCKS: Path = ROOT / "locks"
INIT_BANK: Path = ROOT / "init_bank"
RUNS: Path = ROOT / "runs"
ACCEPTED: Path = ROOT / "accepted"
FEA: Path = ROOT / "fea"
RESULTS: Path = ROOT / "results"
PILOT: Path = ROOT / "pilot"
LOGS: Path = ROOT / "logs"
STATUS: Path = ROOT / "status"
LEDGER: Path = ROOT / "ledger"
TESTS: Path = ROOT / "tests"

# Directories that the runtime writes into; spec/ stays read-only.
_WRITABLE = (DATA, LOCKS, INIT_BANK, RUNS, ACCEPTED, FEA, RESULTS, PILOT, LOGS,
             STATUS, LEDGER)


def ensure_dirs() -> None:
    """Create the writable runtime directories if they are absent."""
    for d in _WRITABLE:
        d.mkdir(parents=True, exist_ok=True)


ensure_dirs()

__all__ = ["ROOT", "PKG", "SPEC", "DATA", "LOCKS", "INIT_BANK", "RUNS",
           "ACCEPTED", "FEA", "RESULTS", "PILOT", "LOGS", "STATUS", "LEDGER",
           "TESTS", "ensure_dirs"]
