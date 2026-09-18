"""Settings, objectives, checksums and paths for the S10 DDPG transfer experiment."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from pathlib import Path

import numpy as np
import yaml

CODE = Path(__file__).resolve().parent
ROOT = CODE.parent
CFG = yaml.safe_load((ROOT / "configs" / "experiment.yaml").read_text(encoding="utf-8"))

TASKS = list(CFG["tasks"])
SEEDS = [int(s) for s in CFG["seeds"]]
D = CFG["ddpg"]

# Files defining the DDPG transfer experiment.
LEARNING_CODE = ["code/ddpg_agent.py", "code/recipe_c4.py", "code/env60.py",
                 "code/run_one.py", "configs/experiment.yaml"]
ORCHESTRATION_CODE = ["code/common_c4.py"]

TASK_IX = {k: v for k, v in CFG["task_indices"].items()}


def sha256(path):
    p = Path(path)
    if not p.is_file():
        return None
    h = hashlib.sha256()
    with p.open("rb") as f:
        for c in iter(lambda: f.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()


def sha256_text(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def atomic_write_bytes(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    with os.fdopen(fd, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    for attempt in range(20):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(0.25)


def atomic_write_text(path, text):
    atomic_write_bytes(path, text.encode("utf-8"))


def atomic_write_json(path, obj):
    atomic_write_text(path, json.dumps(obj, indent=2, ensure_ascii=False,
                                       default=_default))


def _default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, Path):
        return str(o)
    return str(o)


def jload(path, default=None):
    p = Path(path)
    if not p.is_file():
        return default
    try:
        return json.loads(p.read_text(encoding="utf-8-sig"))
    except Exception:                                             # noqa: BLE001
        return default


def code_hashes():
    return {rel: sha256(ROOT / rel) for rel in LEARNING_CODE + ORCHESTRATION_CODE}


def learning_code_hashes():
    return {rel: sha256(ROOT / rel) for rel in LEARNING_CODE}


def verify_assets():
    bad = []
    for rel, want in CFG["required_asset_sha256"].items():
        got = sha256(ROOT / rel)
        if got != want:
            bad.append({"asset": rel, "want": want, "got": got})
    if bad:
        raise SystemExit(f"[assets] SHA-256 mismatch: {bad}")
    return {rel: sha256(ROOT / rel) for rel in CFG["required_asset_sha256"]}


def det_schedule():
    every = int(D["deterministic_eval_interval"])
    n = int(D["episodes"])
    return [0] + list(range(every, n + 1, every))


# ------------------------------------------------------------------ objectives
# env60.Environment computes  reward = reward_function(*u) / mean((s+2)/2),
# and mean((s+2)/2) == 0.5 + rho_A exactly. The bilateral objective is expressed by
# supplying the NUMERATOR only; the shared density denominator is already the
# environment denominator.
def numerator_bilateral(task):
    ix = TASK_IX[task]

    def f(*u):
        return min(u[ix["target"]] - u[ix["lower"]],
                   u[ix["target"]] - u[ix["upper"]])
    return f


def make_numerator(objective, task):
    if objective == "bilateral":
        return numerator_bilateral(task)
    raise ValueError(f"unknown objective {objective!r}")


def rho_a(actions):
    a = np.asarray(actions, dtype=np.float64).reshape(-1)
    return float(np.mean((a + 1.0) / 2.0))


def physical_metrics(u, actions, task):
    """Every physical quantity the protocol asks to store, for one design."""
    u = np.asarray(u, dtype=np.float64).reshape(-1)
    a = np.asarray(actions, dtype=np.float64).reshape(-1)
    ix = TASK_IX[task]
    ui, ul, uu = float(u[ix["target"]]), float(u[ix["lower"]]), float(u[ix["upper"]])
    r = rho_a(a)
    den = 0.5 + r
    others = np.delete(u, ix["target"])
    left, right = ui - ul, ui - uu
    kap = float("nan") if (ul + uu) == 0 else ui / (0.5 * (ul + uu))
    return {
        "u": u.tolist(), "u_target": ui, "u_lower": ul, "u_upper": uu,
        "rho_A": r, "density_code": den, "kappa": kap,
        "left_margin": left, "right_margin": right,
        "bilateral_margin": float(min(left, right)),
        "bilateral_margin_normalized": float(min(left, right)) / den,
        "global_margin": float(ui - np.max(others)),
        "target_is_global_max": bool(ui >= np.max(others)),
        "reward_original": (ui - 0.5 * (ul + uu)) / den,
        "reward_bilateral": float(min(left, right)) / den,
    }


def action_to_thickness(actions, t_min, t_max):
    a = np.asarray(actions, dtype=np.float64).reshape(-1)
    mid = 0.5 * (float(t_min) + float(t_max))
    half = 0.5 * (float(t_max) - float(t_min))
    return mid + half * a


def run_dir(spec):
    return (ROOT / "runs" / spec["objective"] / spec["domain"] / spec["task"] /
            f"seed_{spec['seed']}")


def source_path(bank, task, seed):
    return (Path(os.environ.get("FGL_BILATERAL_OUTPUT", ROOT.parent / "generated"))
            / "source_bank" / "ddpg" / task / f"seed_{seed}.pt")
