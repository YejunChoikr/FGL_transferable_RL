"""Config, hashing and the structurally-inherited calibration H1 protocol."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
from pathlib import Path

import numpy as np
import yaml

CODE = Path(__file__).resolve().parent
ROOT = CODE.parent

CFG = yaml.safe_load((ROOT / "configs" / "experiment.yaml").read_text(encoding="utf-8"))
TASKS = ["FGL5", "FGL6", "FGL7"]
SEEDS = list(CFG["seeds"])
STATE_DIM = int(CFG["env"]["state_dim"])
CRITIC_DIM = int(CFG["env"]["critic_input_dim"])
DET_EVERY = int(CFG["train"]["deterministic_eval_interval"])
DET_EXTRA = list(CFG["train"]["deterministic_eval_episodes_extra"])
C1_GRID = list(CFG["stages"]["stage1_C1"]["C1_grid"])
C2_GRID = list(CFG["stages"]["stage2_C2"]["C2_grid"])

HASHED_FILES = ["configs/experiment.yaml", "configs/CALIBRATION_PROTOCOL_LOCK.yaml",
                "code/sac_agent.py", "code/env_cc.py", "code/reward_cc.py",
                "code/common.py", "code/run_reward.py", "code/unit_tests.py"]
HASHED_ASSETS = [CFG["assets"]["surrogate"], CFG["assets"]["scaler_Y"]]

_LOCK = None


def sha256(path):
    p = Path(path)
    if not p.exists():
        return None
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def code_hashes(root=ROOT):
    return {r: sha256(Path(root) / r) for r in HASHED_FILES}


def asset_hashes(root=ROOT):
    return {r: sha256(Path(root) / r) for r in HASHED_ASSETS}


def bundle_hash(root=ROOT):
    m = {**code_hashes(root), **asset_hashes(root)}
    return hashlib.sha256(json.dumps(m, sort_keys=True).encode()).hexdigest()


def lock(root=ROOT) -> dict:
    """Parse the calibration H1 lock structurally; never re-write its values by hand."""
    global _LOCK
    if _LOCK is not None:
        return _LOCK
    c = CFG["calibration"]
    p = Path(root) / c["lock_file"]
    if not p.exists():
        raise SystemExit(f"[lock] missing {p}")
    y = yaml.safe_load(p.read_text(encoding="utf-8"))
    if str(y.get("status")) != c["required_status"]:
        raise SystemExit(f"[lock] status {y.get('status')} != {c['required_status']}")
    if y.get("selected_config") != c["required_config"]:
        raise SystemExit(f"[lock] selected {y.get('selected_config')} != {c['required_config']}")
    ss = y.get("state_schema") or {}
    if int(ss.get("state_dim", -1)) != int(c["required_state_dim"]):
        raise SystemExit(f"[lock] state_dim {ss.get('state_dim')}")
    if int(ss.get("critic_input_dim", -1)) != int(c["required_critic_input_dim"]):
        raise SystemExit(f"[lock] critic_input_dim {ss.get('critic_input_dim')}")
    _LOCK = {"selected_config": y["selected_config"], "params": y["sac_and_train_parameters"],
             "network": y.get("network"), "state_schema": ss, "raw": y,
             "lock_sha256": sha256(p)}
    return _LOCK


def agent_cfg(root=ROOT) -> dict:
    p = lock(root)["params"]
    return {
        "sac": {"log_std_min": float(p["log_std_min"]), "log_std_max": float(p["log_std_max"]),
                "actor_lr": float(p["actor_lr"]), "critic_lr": float(p["critic_lr"]),
                "alpha_lr": float(p["alpha_lr"]), "init_alpha": float(p["initial_alpha"]),
                "target_entropy": float(p["target_entropy"]),
                "weight_decay": float(p["weight_decay"]),
                "reward_scale": float(p["reward_scale"]),
                "autotune_alpha": bool(p["automatic_entropy_tuning"])},
        "train": {"gamma": float(p["gamma"]), "tau": float(p["tau"]),
                  "memory_capacity": int(p["replay_capacity"]),
                  "batch_size": int(p["batch_size"])},
    }


def episodes(root=ROOT):
    return int(lock(root)["params"]["episodes"])


def fill_episodes(root=ROOT):
    return int(lock(root)["params"]["memory_fill_episodes"])


def hyperparameters(root=ROOT) -> dict:
    a, p = agent_cfg(root), lock(root)["params"]
    return {"inherited_from": "calibration H1 lock", **{k: p[k] for k in (
        "batch_size", "actor_lr", "critic_lr", "alpha_lr", "tau", "gamma", "episodes",
        "episode_length", "memory_fill_episodes", "replay_capacity",
        "updates_per_transition", "automatic_entropy_tuning", "initial_alpha",
        "target_entropy", "weight_decay", "reward_scale", "log_std_min", "log_std_max")},
        "state_dim": STATE_DIM, "critic_input_dim": CRITIC_DIM,
        "state_encoding": CFG["env"]["state_encoding"], "domain": CFG["env"]["domain"],
        "batch_size_effective": a["train"]["batch_size"]}


def env_metadata():
    import torch
    return {"hostname": platform.node(), "os": platform.platform(),
            "python": sys.version.split()[0], "python_executable": sys.executable,
            "conda_prefix": os.environ.get("CONDA_PREFIX", sys.prefix),
            "pytorch": torch.__version__, "cuda_build": torch.version.cuda,
            "cuda_available": bool(torch.cuda.is_available()),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "numpy": np.__version__,
            "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
            "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
            "matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
            "float32_matmul_precision": str(torch.get_float32_matmul_precision())}


def atomic_write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=_default), encoding="utf-8")
    os.replace(tmp, path)


def _default(o):
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.bool_):
        return bool(o)
    raise TypeError(str(type(o)))


def trapz(y, n=None):
    y = np.asarray(y, dtype=np.float64)
    n = n or len(y)
    f = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return float(f(y, dx=1.0) / n)


def run_metrics(rh) -> dict:
    rh = np.asarray(rh, dtype=np.float64)
    return {"final100": float(rh[-100:].mean()),
            "AUC1500": trapz(rh, len(rh)),
            "early500_AUC": trapz(rh[:500], 500),
            "best_observed": float(rh.max()),
            "best_observed_episode": int(rh.argmax() + 1)}


def cell_tag(c1, c2) -> str:
    return f"C1_{float(c1):g}__C2_{float(c2):g}"


def run_dir(root, stage, task, c1, c2, seed) -> Path:
    return Path(root) / "generated" / "runs" / stage / task / cell_tag(c1, c2) / f"seed_{seed}"


def weights_dir(root, stage, task, c1, c2, seed) -> Path:
    return Path(root) / "generated" / "weights" / stage / task / cell_tag(c1, c2) / f"seed_{seed}"


def run_key(stage, task, c1, c2, seed) -> str:
    return f"{stage}__{task}__{cell_tag(c1, c2)}__seed{seed}"


def det_schedule(n_episodes) -> list:
    """Episode 20 plus every 25th episode; identical for every run."""
    s = set(int(e) for e in DET_EXTRA)
    s.update(range(DET_EVERY, n_episodes + 1, DET_EVERY))
    return sorted(e for e in s if 1 <= e <= n_episodes)


def gonogo_decision():
    """Read the go/no-go verdict (may be absent if that project has not finished)."""
    g = CFG["gonogo"]
    p = Path(g["project"]) / g["decision_file"]
    if not p.exists():
        return {"available": False, "classification": None, "path": str(p)}
    d = json.loads(p.read_text(encoding="utf-8"))
    return {"available": True, "classification": d.get("classification"),
            "reason": d.get("reason"), "selected_trl_arm": d.get("selected_trl_arm"),
            "path": str(p), "sha256": sha256(p),
            "allowed": d.get("classification") in g["allowed_classifications"]}
