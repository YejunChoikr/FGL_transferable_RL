"""Domain data access for the CMAME experiments.

Experimental settings are defined in spec/protocol.json.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np

try:  # cmame_rt.paths is owned by SURR; fall back if it is not present yet
    from .paths import ROOT as _ROOT
    ROOT = Path(_ROOT)
except Exception:  # pragma: no cover
    ROOT = Path(__file__).resolve().parents[1]

DATA = ROOT / "data"
CACHE_DIR = DATA / "cache"
SCALER_DIR = DATA / "scalers"
LOCK_PATH = ROOT / "locks" / "DATA_LOCK.json"

DOMAINS = ("source", "upper", "lower", "ar08")
DOMAIN_INDEX = {"source": 0, "upper": 1, "lower": 2, "ar08": 3}
N_TOTAL = {"source": 30000, "upper": 6000, "lower": 6000, "ar08": 6000}
THICKNESS_BOUNDS_MM = {"source": (1.2, 1.8), "upper": (1.2, 2.4),
                       "lower": (0.9, 1.8), "ar08": (1.2, 1.8)}
CELL_SIZE_MM = {"source": (10.0, 10.0), "upper": (10.0, 10.0),
                "lower": (10.0, 10.0), "ar08": (8.0, 10.0)}
BUDGETS = {"source": (30000,), "upper": (500, 1000, 2000, 4000, 6000),
           "lower": (6000,), "ar08": (6000,)}

BLOCK = 10
TRAIN_IN_BLOCK = 8      # positions 0..7 of every block
VAL_POSITION = 8        # position 8
TEST_POSITION = 9       # position 9
ROLES = ("train", "validation", "protocol_test")

_cache: Dict[str, "DomainData"] = {}


@dataclass(frozen=True)
class DomainData:
    """One domain's frozen records, in record_id-ascending row order."""

    domain: str
    X60: np.ndarray          # (N,60) float32 encode_design(actions), golden location
    A30: np.ndarray          # (N,30) float32 normalised actions in [-1,1]
    T30: np.ndarray          # (N,30) float64 physical thickness, mm
    U9: np.ndarray           # (N,9)  float64 probe displacements u1..u9, mm
    record_id: np.ndarray    # (N,)   int64 stable identifier of the raw record
    role: np.ndarray         # (N,)   str in {train, validation, protocol_test}
    block: np.ndarray        # (N,)   int64 = order_index // 10
    order_index: np.ndarray  # (N,)   int64 position in the frozen permutation
    thickness_bounds_mm: tuple
    cell_size_mm: tuple
    n_total: int

    def __len__(self) -> int:
        return int(self.X60.shape[0])


def sha256_file(path) -> str:
    """Hex sha256 of a file's bytes."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_array(a) -> str:
    """Hex sha256 of an array's C-contiguous buffer (order sensitive)."""
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def cache_path(domain: str) -> Path:
    """Path of the frozen cache for ``domain``."""
    return CACHE_DIR / f"{domain}.npz"


def y_scaler_path(domain: str, n_total: int) -> Path:
    """Path of the target scaler shared by every arm at ``domain`` x ``n_total``."""
    return SCALER_DIR / f"{domain}_N{int(n_total)}.json"


def load_domain(domain: str) -> DomainData:
    """Load one domain's frozen cache (memoised per process)."""
    if domain not in DOMAINS:
        raise KeyError(f"unknown domain {domain!r}; expected one of {DOMAINS}")
    if domain in _cache:
        return _cache[domain]
    with np.load(cache_path(domain), allow_pickle=False) as d:
        dd = DomainData(
            domain=domain,
            X60=np.ascontiguousarray(d["X60"], dtype=np.float32),
            A30=np.ascontiguousarray(d["A30"], dtype=np.float32),
            T30=np.ascontiguousarray(d["T30"], dtype=np.float64),
            U9=np.ascontiguousarray(d["U9"], dtype=np.float64),
            record_id=np.ascontiguousarray(d["record_id"], dtype=np.int64),
            role=np.asarray(d["role"]).astype(str),
            block=np.ascontiguousarray(d["block"], dtype=np.int64),
            order_index=np.ascontiguousarray(d["order_index"], dtype=np.int64),
            thickness_bounds_mm=THICKNESS_BOUNDS_MM[domain],
            cell_size_mm=CELL_SIZE_MM[domain],
            n_total=N_TOTAL[domain],
        )
    _cache[domain] = dd
    return dd


def budget_split(domain: str, n_total: int) -> dict:
    """Row indices for one data budget, ordered by the frozen permutation.

    ``n_total`` must be a positive multiple of ten and at most the domain's
    frozen total.  A budget is the prefix of whole blocks of that permutation,
    so train/validation/protocol_test sets are nested across budgets and no
    record changes role.  ``common_test_idx`` is always the full-size
    protocol_test set (600 for a target domain, 3000 for source).
    """
    dd = load_domain(domain)
    n_total = int(n_total)
    if n_total <= 0 or n_total % BLOCK != 0:
        raise ValueError(f"n_total must be a positive multiple of {BLOCK}, got {n_total}")
    if n_total > dd.n_total:
        raise ValueError(f"n_total {n_total} exceeds the frozen total {dd.n_total} "
                         f"for domain {domain}")
    order = dd.order_index
    sel = order < n_total
    pos = order % BLOCK
    train_idx = np.flatnonzero(sel & (pos < TRAIN_IN_BLOCK))
    val_idx = np.flatnonzero(sel & (pos == VAL_POSITION))
    test_idx = np.flatnonzero(sel & (pos == TEST_POSITION))
    common_idx = np.flatnonzero(dd.role == "protocol_test")
    return {
        "train_idx": train_idx[np.argsort(order[train_idx], kind="stable")],
        "val_idx": val_idx[np.argsort(order[val_idx], kind="stable")],
        "test_idx": test_idx[np.argsort(order[test_idx], kind="stable")],
        "common_test_idx": common_idx[np.argsort(order[common_idx], kind="stable")],
    }


def fit_y_scaler(U9: np.ndarray) -> dict:
    """StandardScaler statistics (ddof=0) over the nine displacement columns.

    Matches ``sklearn.preprocessing.StandardScaler``: the population standard
    deviation is used and a zero-variance column keeps a scale of one.
    """
    y = np.asarray(U9, dtype=np.float64)
    if y.ndim != 2 or y.shape[1] != 9:
        raise ValueError(f"expected (n,9) displacements, got {y.shape}")
    if not np.isfinite(y).all():
        raise ValueError("non-finite displacement in the scaler fit set")
    mean = y.mean(axis=0)
    scale = y.std(axis=0, ddof=0)
    scale = np.where(scale == 0.0, 1.0, scale)
    return {"mean": mean.tolist(), "scale": scale.tolist(), "n": int(y.shape[0])}


def load_y_scaler(domain: str, n_total: int) -> dict:
    """Load the frozen target scaler for ``domain`` at budget ``n_total``."""
    p = y_scaler_path(domain, n_total)
    if not p.exists():
        raise FileNotFoundError(f"scaler not found: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def scaler_arrays(scaler: dict):
    """Return ``(mean[9], scale[9])`` float64 arrays for a loaded scaler."""
    return (np.asarray(scaler["mean"], dtype=np.float64),
            np.asarray(scaler["scale"], dtype=np.float64))


def standardize(U9: np.ndarray, scaler: dict) -> np.ndarray:
    """Apply a loaded scaler to displacements in mm."""
    mean, scale = scaler_arrays(scaler)
    return (np.asarray(U9, dtype=np.float64) - mean) / scale


def inverse_standardize(Z: np.ndarray, scaler: dict) -> np.ndarray:
    """Invert :func:`standardize`, returning displacements in mm."""
    mean, scale = scaler_arrays(scaler)
    return np.asarray(Z, dtype=np.float64) * scale + mean


def load_data_lock() -> dict:
    """Read ``locks/DATA_LOCK.json``."""
    return json.loads(LOCK_PATH.read_text(encoding="utf-8"))


def verify_data_lock() -> dict:
    """Re-hash every cache and scaler file and compare with DATA_LOCK.json."""
    lock = load_data_lock()
    report = {"lock_path": str(LOCK_PATH), "domains": {}, "ok": True}
    for domain, entry in lock["domains"].items():
        actual_cache = sha256_file(cache_path(domain))
        expected_cache = entry["cache"]["sha256"]
        scalers = {}
        for name, want in entry["scalers"].items():
            p = SCALER_DIR / name
            got = sha256_file(p) if p.exists() else None
            scalers[name] = {"expected": want["sha256"], "actual": got,
                             "match": got == want["sha256"]}
        ok = (actual_cache == expected_cache) and all(v["match"] for v in scalers.values())
        report["domains"][domain] = {
            "cache": {"expected": expected_cache, "actual": actual_cache,
                      "match": actual_cache == expected_cache},
            "scalers": scalers,
            "ok": ok,
        }
        report["ok"] = report["ok"] and ok
    return report


__all__ = ["DomainData", "DOMAINS", "DOMAIN_INDEX", "N_TOTAL", "BUDGETS",
           "THICKNESS_BOUNDS_MM", "CELL_SIZE_MM", "BLOCK", "ROLES",
           "load_domain", "budget_split", "fit_y_scaler", "load_y_scaler",
           "y_scaler_path", "scaler_arrays", "standardize",
           "inverse_standardize", "verify_data_lock", "load_data_lock",
           "cache_path", "sha256_file", "sha256_array"]
