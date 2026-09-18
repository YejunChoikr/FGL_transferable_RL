"""Supervised surrogate training runner (scratch and transfer arms).

Experimental settings are defined in spec/protocol.json.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from . import models_surrogate as MS
from .determinism import apply_runtime_flags
from .encoding import STATE_DIM, encode_batch_numpy
from .init_bank import (init_name, init_path, load_surrogate_init,
                        tensor_digest)
from .paths import ACCEPTED, DATA, PILOT, ROOT
from .protocol import (PROTOCOL_HASH, file_sha256, hash_report, surrogate_cfg,
                       load_protocol)
from .rng import subseed

_S = surrogate_cfg()
N_PROBES = 9

ADAM_BACKENDS = {
    "reference": {"foreach": False, "fused": False},
    "foreach": {"foreach": True, "fused": False},
    "fused": {"foreach": False, "fused": True},
}


# ----------------------------------------------------------------------
# small helpers
# ----------------------------------------------------------------------
def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _sha256_file(p: Path) -> str:
    return file_sha256(p)


def _json_write(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2,
                              default=_default) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _default(o: Any):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(repr(o))


def resolve_adam_backend(adam_backend: Any) -> dict:
    """Normalise an Adam backend selector into ``{'foreach':…, 'fused':…}``."""
    if adam_backend is None:
        return dict(ADAM_BACKENDS["reference"])
    if isinstance(adam_backend, str):
        try:
            return dict(ADAM_BACKENDS[adam_backend])
        except KeyError as exc:
            raise KeyError("unknown adam backend %r; known: %s"
                           % (adam_backend, sorted(ADAM_BACKENDS))) from exc
    return {"foreach": bool(adam_backend.get("foreach", False)),
            "fused": bool(adam_backend.get("fused", False))}


def expected_initial_lr(case: dict) -> float:
    """Protocol initial LR for a case, independent of the case row value."""
    init = case["init"]
    if init == "transfer":
        return float(_S["transfer_lr"])
    if case.get("domain") == "source":
        return float(_S["source_lr"])
    return float(_S["scratch_lr"])


def assert_initial_lr(optimizer, expected: float) -> None:
    """Refuse to start when any param group LR deviates from the protocol."""
    seen = [float(g["lr"]) for g in optimizer.param_groups]
    if any(lr != expected for lr in seen):
        raise AssertionError(
            "optimizer initial LR %s does not match the protocol value %s; "
            "the run is refused" % (seen, expected))


# ----------------------------------------------------------------------
# data access (DATA agent owns cmame_rt/data.py)
# ----------------------------------------------------------------------
class DataUnavailable(RuntimeError):
    """Raised when cmame_rt.data or its cache is not ready yet."""


def _import_data_module():
    try:
        from . import data as data_mod  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on DATA agent
        raise DataUnavailable("cmame_rt.data is not available: %s" % exc)
    return data_mod


def _y_scaler(data_mod, domain: str, n_total: int) -> dict:
    """Fetch the shared output StandardScaler for one domain and budget."""
    for name in ("load_y_scaler", "get_y_scaler"):
        fn = getattr(data_mod, name, None)
        if fn is not None:
            return fn(domain, n_total)
    path_fn = getattr(data_mod, "y_scaler_path", None)
    if path_fn is not None:
        p = Path(path_fn(domain, n_total))
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    raise DataUnavailable("no y scaler accessor in cmame_rt.data")


def load_case_data(case: dict, allow_synthetic: bool = False) -> dict:
    """Return the tensors and index sets required for one supervised case."""
    domain = case["domain"]
    n_total = int(case["n_total"])
    try:
        data_mod = _import_data_module()
        dom = data_mod.load_domain(domain)
        split = data_mod.budget_split(domain, n_total)
        scaler = _y_scaler(data_mod, domain, n_total)
        x60 = np.asarray(dom.X60, dtype=np.float32)
        u9 = np.asarray(dom.U9, dtype=np.float64)
        record_id = np.asarray(dom.record_id)
        provider = "cmame_rt.data"
        cache_path = DATA / "cache" / ("%s.npz" % domain)
        cache_sha = _sha256_file(cache_path) if cache_path.exists() else None
    except (DataUnavailable, AttributeError, FileNotFoundError, OSError) as exc:
        if not allow_synthetic:
            raise DataUnavailable(
                "real data unavailable (%s); synthetic data is only permitted "
                "for pilot smoke runs" % exc)
        x60, u9, record_id, split, scaler = _synthetic_dataset(case)
        provider = "synthetic_pilot"
        cache_sha = None
    return {
        "X60": x60, "U9": u9, "record_id": record_id, "split": split,
        "scaler": scaler, "provider": provider, "cache_sha256": cache_sha,
    }


def _synthetic_dataset(case: dict):
    """Deterministic stand-in dataset used only to exercise the runner."""
    n_total = int(case["n_total"])
    n_common = 3000 if case["domain"] == "source" else 600
    n = n_total + n_common
    rng = np.random.default_rng(subseed(int(case["seed"]), "train_shuffle"))
    a30 = rng.uniform(-1, 1, size=(n, 30)).astype(np.float32)
    x60 = encode_batch_numpy(a30)
    w = np.linspace(0.2, 1.0, 30)[None, :]
    base = (a30 * w) @ np.linspace(-1, 1, 30).reshape(30, 1)
    u9 = (np.tanh(base) * np.linspace(1, 5, N_PROBES)[None, :]
          + 0.05 * rng.standard_normal((n, N_PROBES))).astype(np.float64)
    record_id = np.arange(n)
    n_train = int(round(0.8 * n_total))
    n_val = int(round(0.1 * n_total))
    idx = np.arange(n)
    split = {
        "train_idx": idx[:n_train],
        "val_idx": idx[n_train:n_train + n_val],
        "test_idx": idx[n_train + n_val:n_total],
        "common_test_idx": idx[n_total:],
    }
    tr = u9[split["train_idx"]]
    scaler = {"mean": tr.mean(axis=0).tolist(),
              "scale": tr.std(axis=0, ddof=0).tolist()}
    return x60, u9, record_id, split, scaler


def _as_idx(split: dict, key: str) -> np.ndarray:
    for k in (key, key.replace("_idx", ""), key.replace("_idx", "_indices")):
        if k in split:
            return np.asarray(split[k], dtype=np.int64)
    raise KeyError("split has no %r (keys: %s)" % (key, sorted(split)))


# ----------------------------------------------------------------------
# model construction
# ----------------------------------------------------------------------
def resolve_source_best(case: dict, source_best_path: Any = None) -> Path:
    """Locate the accepted source ``best.pt`` for a transfer case."""
    if source_best_path is not None:
        p = Path(source_best_path)
        if p.is_dir():
            p = p / "best.pt"
        if not p.exists():
            raise FileNotFoundError("source checkpoint not found: %s" % p)
        return p
    deps = list(case.get("dependencies") or [])
    if len(deps) != 1:
        raise ValueError("transfer case %s must declare exactly one "
                         "dependency, found %s" % (case["case_id"], deps))
    dep = deps[0]
    candidates = [ACCEPTED / dep / "best.pt"]
    candidates += sorted((ACCEPTED / dep).glob("attempt_*/best.pt"))
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        "accepted source checkpoint for %s is missing; a missing checkpoint is "
        "never substituted by another seed" % dep)


def build_initial_model(case: dict, source_best_path: Any = None):
    """Build the model at its protocol-defined initial state.

    Returns ``(model, transfer_manifest_or_None)``. Scratch loads the frozen
    init-bank tensors; transfer copies convolution / hidden FC / BN affine
    parameters from the paired-seed source and keeps the paired scratch output
    layer, with BatchNorm running statistics reset.
    """
    arch = case["architecture"]
    seed = int(case["seed"])
    model = MS.build_model(arch)
    MS.assert_param_count(model, arch)
    scratch_state = load_surrogate_init(arch, seed)
    model.load_state_dict(scratch_state, strict=True)
    if case["init"] != "transfer":
        return model, None

    src_path = resolve_source_best(case, source_best_path)
    src_state = torch.load(src_path, map_location="cpu", weights_only=True)
    if isinstance(src_state, dict) and "state_dict" in src_state:
        src_state = src_state["state_dict"]
    own = model.state_dict()
    if set(src_state) != set(own):
        raise AssertionError("source state keys differ from the target model")

    copy_keys = MS.transfer_copy_keys(arch, own)
    reset_keys = list(MS.OUTPUT_KEYS[arch])
    copied = {}
    for k in copy_keys:
        if src_state[k].shape != own[k].shape:
            raise AssertionError("shape mismatch on transfer key %s" % k)
        own[k] = src_state[k].detach().clone()
        copied[k] = {"shape": list(own[k].shape),
                     "sha256": tensor_digest({k: own[k]})}
    for k in reset_keys:
        own[k] = scratch_state[k].detach().clone()
    model.load_state_dict(own, strict=True)
    MS.reset_bn_running_stats(model)

    after = model.state_dict()
    bn_buffer_keys = sorted(k for k in after if k.rsplit(".", 1)[-1]
                            in tuple(_S["transfer"]["reset_bn_buffers"]))
    leftover = sorted(set(own) - set(copy_keys) - set(reset_keys))
    if leftover != bn_buffer_keys:
        raise AssertionError("keys neither copied nor reset: %s"
                             % sorted(set(leftover) - set(bn_buffer_keys)))
    manifest = {
        "source_checkpoint": str(src_path),
        "source_checkpoint_sha256": _sha256_file(src_path),
        "scratch_init_file": init_name(arch, seed),
        "scratch_init_sha256": _sha256_file(init_path(arch, seed)),
        "copied_keys": sorted(copied),
        "copied_key_count": len(copied),
        "copied_detail": copied,
        "reset_output_keys": reset_keys,
        "reset_output_equals_paired_scratch": all(
            torch.equal(after[k], scratch_state[k]) for k in reset_keys),
        "bn_buffers_reset": bn_buffer_keys,
        "all_layers_trainable": all(p.requires_grad
                                    for p in model.parameters()),
        "keys_unaccounted_for": [],
        "key_accounting": {
            "copied_from_source": len(copy_keys),
            "paired_scratch_output": len(reset_keys),
            "bn_buffers_reset": len(bn_buffer_keys),
            "total_state_keys": len(own),
        },
    }
    return model, manifest


# ----------------------------------------------------------------------
# evaluation helpers
# ----------------------------------------------------------------------
@torch.no_grad()
def _predict_mm(model, x: torch.Tensor, mean_t, scale_t, batch: int):
    model.eval()
    out = torch.empty((x.shape[0], N_PROBES), dtype=torch.float32,
                      device=x.device)
    for s in range(0, x.shape[0], batch):
        e = min(s + batch, x.shape[0])
        out[s:e] = model(x[s:e]) * scale_t + mean_t
    return out


def _mse_mm2(pred_mm: torch.Tensor, true_mm: torch.Tensor) -> float:
    d = (pred_mm.double() - true_mm.double())
    return float(torch.mean(d * d).item())


def _regression_metrics(pred_mm: np.ndarray, true_mm: np.ndarray) -> dict:
    err = pred_mm.astype(np.float64) - true_mm.astype(np.float64)
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err * err)))
    ss_res = np.sum(err * err, axis=0)
    ss_tot = np.sum((true_mm - true_mm.mean(axis=0)) ** 2, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        r2 = 1.0 - ss_res / ss_tot
    r2 = np.where(ss_tot == 0, np.nan, r2)
    return {"mae_mm": mae, "rmse_mm": rmse,
            "per_probe_r2": [float(v) for v in r2],
            "mean_probe_r2": float(np.nanmean(r2)),
            "n": int(pred_mm.shape[0])}


# ----------------------------------------------------------------------
# main runner
# ----------------------------------------------------------------------
def run_supervised(case: dict, attempt_dir: Any, device: Any = "cuda:0",
                   adam_backend: Any = None, source_best_path: Any = None,
                   epochs: Any = None, allow_synthetic: bool = False) -> dict:
    """Train one supervised surrogate case and write the INTERFACE artifacts.

    ``epochs`` may only be reduced for pilot smoke runs written under
    ``pilot/``; production runs always use the 300 epochs fixed by the protocol.
    """
    started = _now()
    t0 = time.perf_counter()
    attempt_dir = Path(attempt_dir)
    attempt_dir.mkdir(parents=True, exist_ok=True)
    is_pilot = PILOT.resolve() in attempt_dir.resolve().parents or \
        attempt_dir.resolve() == PILOT.resolve()

    protocol_epochs = int(_S["epochs"])
    n_epochs = protocol_epochs if epochs is None else int(epochs)
    if n_epochs != protocol_epochs:
        if not is_pilot:
            raise ValueError("reduced epoch counts are only allowed under "
                             "pilot/; production uses %d" % protocol_epochs)
        if n_epochs > 3:
            raise ValueError("pilot runs are capped at 3 epochs")
    if allow_synthetic and not is_pilot:
        raise ValueError("synthetic data is only allowed under pilot/")

    runtime_state = apply_runtime_flags()
    dev = torch.device(device if torch.cuda.is_available()
                       or str(device).startswith("cpu") else "cpu")

    arch = case["architecture"]
    seed = int(case["seed"])
    backend = resolve_adam_backend(adam_backend)
    if backend["fused"] and dev.type != "cuda":
        raise ValueError("the fused Adam backend requires a CUDA device")

    # ---- data -------------------------------------------------------
    bundle = load_case_data(case, allow_synthetic=allow_synthetic)
    split = bundle["split"]
    tr_idx = _as_idx(split, "train_idx")
    va_idx = _as_idx(split, "val_idx")
    te_idx = _as_idx(split, "test_idx")
    ct_idx = _as_idx(split, "common_test_idx")
    scaler = bundle["scaler"]
    mean = np.asarray(scaler["mean"], dtype=np.float64)
    scale = np.asarray(scaler["scale"], dtype=np.float64)
    if mean.shape != (N_PROBES,) or scale.shape != (N_PROBES,):
        raise ValueError("y scaler must carry nine means and nine scales")
    if not np.all(scale > 0):
        raise ValueError("y scaler has a non-positive scale")

    x_all = torch.as_tensor(bundle["X60"], dtype=torch.float32, device=dev)
    if x_all.shape[1] != STATE_DIM:
        raise ValueError("X60 must have 60 columns")
    u_all = torch.as_tensor(bundle["U9"], dtype=torch.float64, device=dev)
    mean_t = torch.as_tensor(mean, dtype=torch.float32, device=dev)
    scale_t = torch.as_tensor(scale, dtype=torch.float32, device=dev)
    y_std_all = ((u_all - torch.as_tensor(mean, device=dev))
                 / torch.as_tensor(scale, device=dev)).to(torch.float32)

    ti = torch.as_tensor(tr_idx, dtype=torch.long, device=dev)
    vi = torch.as_tensor(va_idx, dtype=torch.long, device=dev)
    x_tr, y_tr = x_all[ti], y_std_all[ti]
    x_va = x_all[vi]
    u_va_mm = u_all[vi].to(torch.float32)
    n_train = int(x_tr.shape[0])
    t_load_s = time.perf_counter() - t0

    # ---- model / optimizer -----------------------------------------
    model, transfer_manifest = build_initial_model(case, source_best_path)
    model.to(dev)
    opt_cfg = _S["optimizer"]
    lr0 = expected_initial_lr(case)
    case_lr = float(case.get("initial_lr", lr0))
    if case_lr != lr0:
        raise AssertionError("case initial_lr %s disagrees with the protocol "
                             "value %s" % (case_lr, lr0))
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr0,
        betas=tuple(float(b) for b in opt_cfg["betas"]),
        eps=float(opt_cfg["eps"]), weight_decay=float(opt_cfg["weight_decay"]),
        amsgrad=bool(opt_cfg["amsgrad"]), foreach=backend["foreach"],
        fused=backend["fused"])
    assert_initial_lr(optimizer, lr0)
    n_opt_params = sum(p.numel() for g in optimizer.param_groups
                       for p in g["params"])
    n_model_params = MS.count_params(model)
    if n_opt_params != n_model_params:
        raise AssertionError("optimizer covers %d of %d trainable parameters"
                             % (n_opt_params, n_model_params))

    sch_cfg = _S["scheduler"]
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode=sch_cfg["mode"], factor=float(sch_cfg["factor"]),
        patience=int(sch_cfg["patience"]), threshold=float(sch_cfg["threshold"]),
        threshold_mode=sch_cfg["threshold_mode"],
        cooldown=int(sch_cfg["cooldown"]), min_lr=float(sch_cfg["min_lr"]),
        eps=float(sch_cfg["eps"]))

    batch = int(_S["batch_size"])
    drop_last = bool(_S["drop_last"])
    eval_batch = int(_S["validation_batch_size"])
    test_batch = int(_S["test_batch_size"])

    # ---- resolved config -------------------------------------------
    resolved = {
        "case": dict(case),
        "protocol": hash_report(),
        "protocol_surrogate": _S,
        "runtime": load_protocol()["runtime"],
        "runtime_observed": runtime_state,
        "adam_backend": backend,
        "adam_backend_name": (adam_backend if isinstance(adam_backend, str)
                              else "reference" if adam_backend is None
                              else "custom"),
        "device": str(dev),
        "epochs": n_epochs,
        "epochs_protocol": protocol_epochs,
        "pilot": bool(is_pilot),
        "initial_lr": lr0,
        "batch_size": batch,
        "drop_last": drop_last,
        "validation_batch_size": eval_batch,
        "data": {
            "provider": bundle["provider"],
            "cache_sha256": bundle["cache_sha256"],
            "n_train": n_train, "n_validation": int(len(va_idx)),
            "n_protocol_test": int(len(te_idx)),
            "n_common_test": int(len(ct_idx)),
        },
        "y_scaler": {"mean": mean.tolist(), "scale": scale.tolist(),
                     "sha256": hashlib.sha256(
                         json.dumps({"mean": mean.tolist(),
                                     "scale": scale.tolist()},
                                    sort_keys=True,
                                    separators=(",", ":")).encode()
                     ).hexdigest()},
        "init": {
            "file": init_name(arch, seed),
            "sha256": _sha256_file(init_path(arch, seed)),
            "subseed_cnn_or_mlp": subseed(seed, "%s_init" % arch),
            "train_shuffle_subseed": subseed(seed, "train_shuffle"),
            "dropout_subseed": subseed(seed, "surrogate_dropout"),
        },
        "transfer": transfer_manifest,
        "host": platform.node(),
        "started_at": started,
    }
    _json_write(attempt_dir / "resolved_config.json", resolved)
    _json_write(attempt_dir / "scaler.json",
                {"mean": mean.tolist(), "scale": scale.tolist(),
                 "ddof": 0, "source": bundle["provider"]})
    if transfer_manifest is not None:
        _json_write(attempt_dir / "transfer_manifest.json", transfer_manifest)

    # ---- training ---------------------------------------------------
    shuffle_gen = torch.Generator(device="cpu")
    shuffle_gen.manual_seed(subseed(seed, "train_shuffle"))
    torch.manual_seed(subseed(seed, "surrogate_dropout"))

    loss_fn = nn.MSELoss()
    best_val = float("inf")
    best_epoch = -1
    best_state = None
    nonfinite = False
    train_rows = []
    lr_rows = []

    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    t_train0 = time.perf_counter()

    for epoch in range(1, n_epochs + 1):
        lr_now = float(optimizer.param_groups[0]["lr"])
        lr_rows.append((epoch, lr_now))
        model.train()
        perm = torch.randperm(n_train, generator=shuffle_gen).to(dev)
        total = torch.zeros((), dtype=torch.float64, device=dev)
        nb = 0
        limit = (n_train - n_train % batch) if drop_last else n_train
        for s in range(0, limit, batch):
            sel = perm[s:s + batch]
            xb, yb = x_tr[sel], y_tr[sel]
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
            total += loss.detach().double()
            nb += 1
        train_loss = float((total / max(nb, 1)).item())

        pred_va = _predict_mm(model, x_va, mean_t, scale_t, eval_batch)
        val_mse = _mse_mm2(pred_va, u_va_mm)
        if not np.isfinite(train_loss) or not np.isfinite(val_mse):
            nonfinite = True
        train_rows.append((epoch, train_loss, val_mse, lr_now))

        if val_mse < best_val:
            best_val = val_mse
            best_epoch = epoch
            best_state = {k: v.detach().clone()
                          for k, v in model.state_dict().items()}
        scheduler.step(val_mse)

    if best_state is None:
        raise RuntimeError("no epoch completed; nothing to checkpoint")
    torch.save(best_state, attempt_dir / "best.pt")
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    t_train_s = time.perf_counter() - t_train0

    torch.save({k: v.detach().clone() for k, v in model.state_dict().items()},
               attempt_dir / "last.pt")

    with open(attempt_dir / "lr_log.csv", "w", encoding="utf-8",
              newline="") as fh:
        fh.write("epoch,lr\n")
        for e, lr in lr_rows:
            fh.write("%d,%.12g\n" % (e, lr))
    with open(attempt_dir / "train_log.csv", "w", encoding="utf-8",
              newline="") as fh:
        fh.write("epoch,train_loss_std,val_mse_mm2,lr\n")
        for e, tl, vm, lr in train_rows:
            fh.write("%d,%.12g,%.12g,%.12g\n" % (e, tl, vm, lr))

    # ---- test metrics on the best checkpoint ------------------------
    eval_model = MS.build_model(arch).to(dev)
    eval_model.load_state_dict(best_state, strict=True)
    t_eval0 = time.perf_counter()

    ct = torch.as_tensor(ct_idx, dtype=torch.long, device=dev)
    pred_ct = _predict_mm(eval_model, x_all[ct], mean_t, scale_t,
                          test_batch).cpu().numpy()
    true_ct = u_all[ct].cpu().numpy()
    common = _regression_metrics(pred_ct, true_ct)

    pt = torch.as_tensor(te_idx, dtype=torch.long, device=dev)
    if len(te_idx):
        pred_pt = _predict_mm(eval_model, x_all[pt], mean_t, scale_t,
                              test_batch).cpu().numpy()
        protocol_test = _regression_metrics(pred_pt, u_all[pt].cpu().numpy())
    else:
        protocol_test = {"mae_mm": None, "rmse_mm": None, "per_probe_r2": [],
                         "mean_probe_r2": None, "n": 0}
    t_eval_s = time.perf_counter() - t_eval0

    rid = np.asarray(bundle["record_id"])[ct_idx]
    np.savez(attempt_dir / "predictions_common_test.npz",
             pred_mm=pred_ct.astype(np.float64),
             true_mm=true_ct.astype(np.float64),
             record_id=rid)

    if not np.isfinite(pred_ct).all():
        nonfinite = True

    _json_write(attempt_dir / "metrics.json", {
        "common_test": common,
        "protocol_test_log_only": protocol_test,
        "best_epoch": best_epoch,
        "best_val_mse_mm2": best_val,
    })
    _json_write(attempt_dir / "timing.json", {
        "t_load_s": t_load_s, "t_train_s": t_train_s, "t_eval_s": t_eval_s,
        "epochs": n_epochs, "n_train": n_train, "batch_size": batch,
        "device": str(dev),
        "gpu": (torch.cuda.get_device_name(dev) if dev.type == "cuda"
                else None),
    })

    artifacts = {}
    for name in ("resolved_config.json", "lr_log.csv", "train_log.csv",
                 "best.pt", "last.pt", "scaler.json", "metrics.json",
                 "predictions_common_test.npz", "timing.json",
                 "transfer_manifest.json"):
        p = attempt_dir / name
        if p.exists():
            artifacts[name] = _sha256_file(p)

    result = {
        "complete": True,
        "case_id": case["case_id"],
        "attempt": int(case.get("attempt", 1)),
        "kind": "supervised",
        "epochs_done": n_epochs,
        "best_epoch": best_epoch,
        "best_val_mse_mm2": best_val,
        "nonfinite": bool(nonfinite),
        "protocol_hash": PROTOCOL_HASH,
        "artifact_sha256": artifacts,
        "t_train_s": t_train_s,
        "t_load_s": t_load_s,
        "host": platform.node(),
        "device": str(dev),
        "data_provider": bundle["provider"],
        "pilot": bool(is_pilot),
        "started_at": started,
        "finished_at": _now(),
    }
    _json_write(attempt_dir / "result.json", result)
    return result


# ----------------------------------------------------------------------
# policy-environment loader
# ----------------------------------------------------------------------
def load_surrogate_for_env(case_id_or_path: Any, device: Any = "cuda:0",
                           arch: Any = None):
    """Load an accepted surrogate for the RL environment.

    Returns ``(model.eval(), {'mean': tensor[9], 'scale': tensor[9]})`` with
    ``requires_grad=False`` on every parameter, so the environment never
    updates BatchNorm or dropout state (protocol.surrogate
    .policy_environment_mode).
    """
    p = Path(case_id_or_path)
    if not p.exists():
        p = ACCEPTED / str(case_id_or_path)
    ckpt = p if p.is_file() else p / "best.pt"
    if not ckpt.exists():
        cands = sorted(p.glob("attempt_*/best.pt"))
        if not cands:
            raise FileNotFoundError("no accepted best.pt under %s" % p)
        ckpt = cands[0]
    base = ckpt.parent

    if arch is None:
        cfg_path = base / "resolved_config.json"
        if cfg_path.exists():
            arch = json.loads(cfg_path.read_text(
                encoding="utf-8"))["case"]["architecture"]
        else:
            arch = "cnn"

    dev = torch.device(device)
    model = MS.build_model(arch).to(dev)
    state = torch.load(ckpt, map_location=dev, weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    for prm in model.parameters():
        prm.requires_grad_(False)

    scaler_path = base / "scaler.json"
    if not scaler_path.exists():
        raise FileNotFoundError("scaler.json missing next to %s" % ckpt)
    sc = json.loads(scaler_path.read_text(encoding="utf-8"))
    scaler = {
        "mean": torch.as_tensor(sc["mean"], dtype=torch.float32, device=dev),
        "scale": torch.as_tensor(sc["scale"], dtype=torch.float32, device=dev),
    }
    return model, scaler


__all__ = ["run_supervised", "load_surrogate_for_env", "build_initial_model",
           "resolve_source_best", "resolve_adam_backend",
           "expected_initial_lr", "assert_initial_lr", "load_case_data",
           "ADAM_BACKENDS", "DataUnavailable"]
