"""Frozen CPU initial tensors for the surrogate architectures.

Experimental settings are defined in spec/protocol.json.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .models_surrogate import (EXPECTED_PARAMS, assert_param_count,
                               build_model, count_params)
from .paths import INIT_BANK, LOCKS
from .protocol import canonical_digest, file_sha256, hash_report
from .rng import STREAM_IDS, subseed

ARCH_STREAM = {"cnn": "cnn_init", "mlp": "mlp_init"}
SEEDS = (0, 1, 2, 3, 4)
INIT_LOCK_PATH: Path = LOCKS / "INIT_LOCK.json"


def init_name(arch: str, seed: int) -> str:
    """Canonical file name for one initial surrogate state."""
    return "%s_s%d.pt" % (arch, int(seed))


def init_path(arch: str, seed: int) -> Path:
    """Absolute path of one initial surrogate state."""
    return INIT_BANK / init_name(arch, seed)


def tensor_digest(state_dict: dict) -> str:
    """Container-independent sha256 over sorted (key, dtype, shape, bytes)."""
    h = hashlib.sha256()
    for k in sorted(state_dict):
        t = state_dict[k]
        arr = t.detach().cpu().numpy()
        h.update(k.encode())
        h.update(str(arr.dtype).encode())
        h.update(str(arr.shape).encode())
        h.update(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()


def build_surrogate_init(seed: int, arch: str) -> dict:
    """Build the frozen CPU initial ``state_dict`` for ``(seed, arch)``."""
    if arch not in ARCH_STREAM:
        raise KeyError("unknown architecture %r" % (arch,))
    sub = subseed(seed, ARCH_STREAM[arch])
    with torch.device("cpu"):
        torch.manual_seed(sub)
        model = build_model(arch)
    assert_param_count(model, arch)
    sd = model.state_dict()
    return {k: v.detach().clone().cpu() for k, v in sd.items()}


def save_init(state_dict: dict, arch: str, seed: int,
              overwrite: bool = False) -> Path:
    """Write one initial state to ``init_bank/<arch>_s<seed>.pt``."""
    path = init_path(arch, seed)
    if path.exists() and not overwrite:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".pt.tmp")
    torch.save(state_dict, tmp)
    os.replace(tmp, path)
    return path


def load_init(name: str) -> dict:
    """Load an initial state by file name (``cnn_s0.pt``) or by path."""
    p = Path(name)
    if not p.is_absolute() and p.parent == Path("."):
        p = INIT_BANK / name
    if p.suffix != ".pt":
        p = p.with_suffix(".pt")
    return torch.load(p, map_location="cpu", weights_only=True)


def load_surrogate_init(arch: str, seed: int) -> dict:
    """Load the frozen initial state for ``(seed, arch)``."""
    return load_init(init_name(arch, seed))


def build_all(seeds=SEEDS, archs=("cnn", "mlp"), overwrite: bool = False) -> dict:
    """Create every surrogate initial tensor and return the manifest record."""
    entries = {}
    for arch in archs:
        for seed in seeds:
            sd = build_surrogate_init(seed, arch)
            path = save_init(sd, arch, seed, overwrite=overwrite)
            reloaded = load_init(path.name)
            if tensor_digest(reloaded) != tensor_digest(sd):
                raise AssertionError("round-trip mismatch for %s" % path.name)
            model = build_model(arch)
            model.load_state_dict(reloaded, strict=True)
            entries[path.name] = {
                "architecture": arch,
                "seed": int(seed),
                "rng_stream": ARCH_STREAM[arch],
                "rng_stream_id": int(STREAM_IDS[ARCH_STREAM[arch]]),
                "subseed": int(subseed(seed, ARCH_STREAM[arch])),
                "file_sha256": file_sha256(path),
                "tensor_digest": tensor_digest(reloaded),
                "trainable_params": count_params(model),
                "expected_trainable_params": EXPECTED_PARAMS[arch],
                "n_state_keys": len(reloaded),
                "keys": sorted(reloaded),
            }
    return entries


def surrogate_lock_record(entries: dict) -> dict:
    """Assemble the ``surrogate`` section of INIT_LOCK.json."""
    record = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "host": platform.node(),
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "device": "cpu",
        "rule": {
            "seeding": ("torch.manual_seed(subseed(seed, cnn_init|mlp_init)) "
                        "on CPU, then module construction with pinned "
                        "nn.Linear/nn.Conv2d reset_parameters defaults"),
            "subseed": ("int(SeedSequence([20260913, run_seed, stream_id])."
                        "generate_state(1, dtype=uint32)[0])"),
            "batchnorm": ("affine weight=1, bias=0; running_mean=0, "
                          "running_var=1, num_batches_tracked=0"),
            "reuse": ("one state per (seed, architecture), shared across "
                      "domains, budgets and transfer arms"),
            "tensor_digest": ("sha256 over sorted key, dtype, shape and "
                              "contiguous little-endian bytes"),
        },
        "expected_trainable_params": dict(EXPECTED_PARAMS),
        "protocol": hash_report(),
        "files": entries,
    }
    record["record_digest"] = canonical_digest(
        {k: v for k, v in record.items() if k != "generated_at"})
    return record


def write_init_lock(entries: dict, path: Path = INIT_LOCK_PATH) -> dict:
    """Merge the surrogate section into INIT_LOCK.json, preserving other keys.

    Only the ``surrogate`` key is written; the policy section belongs to the
    POLICY agent and is left untouched.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    doc: dict = {}
    for _ in range(5):
        try:
            if path.exists():
                doc = json.loads(path.read_text(encoding="utf-8"))
            break
        except (json.JSONDecodeError, OSError):
            time.sleep(0.2)
    doc["surrogate"] = surrogate_lock_record(entries)
    payload = json.dumps(doc, ensure_ascii=False, indent=2) + "\n"
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False,
                                     dir=str(path.parent),
                                     suffix=".tmp") as fh:
        fh.write(payload)
        tmp = Path(fh.name)
    os.replace(tmp, path)
    return doc["surrogate"]


def verify_against_lock(path: Path = INIT_LOCK_PATH) -> dict:
    """Re-hash every stored initial tensor and compare with INIT_LOCK.json."""
    doc = json.loads(path.read_text(encoding="utf-8"))
    files = doc["surrogate"]["files"]
    problems = []
    for name, rec in files.items():
        p = INIT_BANK / name
        if not p.exists():
            problems.append("%s missing" % name)
            continue
        sd = load_init(name)
        if tensor_digest(sd) != rec["tensor_digest"]:
            problems.append("%s tensor digest mismatch" % name)
        if file_sha256(p) != rec["file_sha256"]:
            problems.append("%s file sha256 mismatch" % name)
        rebuilt = build_surrogate_init(rec["seed"], rec["architecture"])
        if tensor_digest(rebuilt) != rec["tensor_digest"]:
            problems.append("%s is not reproducible from its subseed" % name)
    return {"n_files": len(files), "problems": problems,
            "pass": not problems}


def main(argv: Any = None) -> int:  # pragma: no cover - CLI helper
    import argparse

    ap = argparse.ArgumentParser(description="build surrogate init bank")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--verify-only", action="store_true")
    args = ap.parse_args(argv)
    if args.verify_only:
        print(json.dumps(verify_against_lock(), indent=2))
        return 0
    entries = build_all(overwrite=args.overwrite)
    rec = write_init_lock(entries)
    print(json.dumps({"files": sorted(entries),
                      "record_digest": rec["record_digest"]}, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["ARCH_STREAM", "SEEDS", "INIT_LOCK_PATH", "init_name", "init_path",
           "tensor_digest", "build_surrogate_init", "save_init", "load_init",
           "load_surrogate_init", "build_all", "surrogate_lock_record",
           "write_init_lock", "verify_against_lock"]
