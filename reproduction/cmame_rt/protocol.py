"""Authoritative access to spec/protocol.json and the frozen case registry.

Experimental settings are defined in spec/protocol.json.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from functools import lru_cache
from pathlib import Path
from types import ModuleType
from typing import Any

from .paths import SPEC

PROTOCOL_PATH: Path = SPEC / "protocol.json"
CASES_PATH: Path = SPEC / "cases.json"
ALIASES_PATH: Path = SPEC / "case_aliases.json"
FEA_REQUESTS_PATH: Path = SPEC / "fea_requests.json"
RESULT_VIEWS_PATH: Path = SPEC / "result_views.json"
LOCATION_GOLDEN_PATH: Path = SPEC / "location_golden.json"
REFERENCE_MATH_PATH: Path = SPEC / "reference_math.py"

#: Expected canonical hash, as stamped into every row of cases.json.
EXPECTED_PROTOCOL_HASH = "d58f0bd1b6bcb5708c0d75a1e3c8e25370d3cb993fa4674c431434a449ef552a"


def canonical_digest(obj: Any) -> str:
    """Return the spec's canonical sha256 of a JSON-serialisable object.

    Identical to ``digest`` in spec/build_manifest.py: the object is dumped with
    sorted keys and compact separators before hashing.
    """
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: str | Path) -> str:
    """Return the sha256 digest of the raw bytes of ``path``."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_json(path: Path) -> Any:
    with open(path, "rb") as fh:
        return json.loads(fh.read())


@lru_cache(maxsize=1)
def load_protocol() -> dict:
    """Load spec/protocol.json, the sole source of experimental settings."""
    return _read_json(PROTOCOL_PATH)


@lru_cache(maxsize=1)
def load_cases() -> list:
    """Load the registered case registry."""
    return _read_json(CASES_PATH)


@lru_cache(maxsize=1)
def load_aliases() -> Any:
    """Load the alias table; aliases never define new jobs."""
    return _read_json(ALIASES_PATH)


@lru_cache(maxsize=1)
def load_fea_requests() -> Any:
    """Load the frozen FEA request registry."""
    return _read_json(FEA_REQUESTS_PATH)


@lru_cache(maxsize=1)
def load_result_views() -> Any:
    """Load the canonical result-view definitions."""
    return _read_json(RESULT_VIEWS_PATH)


@lru_cache(maxsize=1)
def load_location_golden() -> dict:
    """Load the golden dimensionless location vector record."""
    return _read_json(LOCATION_GOLDEN_PATH)


@lru_cache(maxsize=1)
def _cases_by_id() -> dict:
    return {c["case_id"]: c for c in load_cases()}


def case_by_id(case_id: str) -> dict:
    """Return the frozen case row for ``case_id``."""
    try:
        return _cases_by_id()[case_id]
    except KeyError as exc:  # pragma: no cover - defensive
        raise KeyError(f"unknown case_id: {case_id}") from exc


def cases_of_kind(kind: str) -> list:
    """Return every case row whose ``kind`` equals the argument."""
    return [c for c in load_cases() if c.get("kind") == kind]


@lru_cache(maxsize=1)
def load_reference_math() -> ModuleType:
    """Import spec/reference_math.py as a module (gates and tests only)."""
    spec = importlib.util.spec_from_file_location(
        "cmame_reference_math", REFERENCE_MATH_PATH)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise ImportError(f"cannot load {REFERENCE_MATH_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PROTOCOL_HASH: str = canonical_digest(load_protocol())
PROTOCOL_FILE_SHA256: str = file_sha256(PROTOCOL_PATH)
PROTOCOL_HASH_MATCHES_SPEC: bool = PROTOCOL_HASH == EXPECTED_PROTOCOL_HASH


def hash_report() -> dict:
    """Both protocol digests plus the agreement flag, for lock files."""
    return {
        "protocol_hash": PROTOCOL_HASH,
        "protocol_hash_expected": EXPECTED_PROTOCOL_HASH,
        "protocol_hash_matches_spec": PROTOCOL_HASH_MATCHES_SPEC,
        "protocol_file_sha256": PROTOCOL_FILE_SHA256,
        "protocol_hash_rule": ("sha256(json.dumps(obj, sort_keys=True, "
                               "separators=(',',':')).encode())"),
    }


def surrogate_cfg() -> dict:
    """Shortcut to ``protocol['surrogate']``."""
    return load_protocol()["surrogate"]


def runtime_cfg() -> dict:
    """Shortcut to ``protocol['runtime']``."""
    return load_protocol()["runtime"]


def domain_cfg(domain: str) -> dict:
    """Return the physical bounds record for one domain."""
    return load_protocol()["domains"][domain]


def reward_cfg() -> dict:
    """Shortcut to ``protocol['reward']``."""
    return load_protocol()["reward"]


__all__ = ["PROTOCOL_HASH", "PROTOCOL_FILE_SHA256", "EXPECTED_PROTOCOL_HASH",
           "PROTOCOL_HASH_MATCHES_SPEC", "canonical_digest", "file_sha256",
           "load_protocol", "load_cases", "load_aliases", "load_fea_requests",
           "load_result_views", "load_location_golden", "load_reference_math",
           "case_by_id", "cases_of_kind", "hash_report", "surrogate_cfg",
           "runtime_cfg", "domain_cfg", "reward_cfg"]
