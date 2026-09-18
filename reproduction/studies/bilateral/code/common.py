"""Shared immutable configuration, paths, hashing, and environment builders."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

from bilateral import REWARDS
from env60 import Environment, configure_determinism

CODE = Path(__file__).resolve().parent
ROOT = CODE.parent
OUTPUT = Path(os.environ.get("FGL_BILATERAL_OUTPUT", ROOT / "generated"))
LOCK_PATH = ROOT / "configs" / "BILATERAL_EXPERIMENT_PROTOCOL_LOCK.yaml"
CFG = yaml.safe_load(LOCK_PATH.read_text(encoding="utf-8"))
TASKS = list(CFG["scope"]["tasks"])
SEEDS = [int(x) for x in CFG["scope"]["seeds"]]
EPISODES = int(CFG["training"]["episodes"])
DET_EVERY = 25

MACHINE_SEEDS = {
    name: [int(x) for x in row["seeds"]]
    for name, row in CFG["machine_allocation"].items()
}

CODE_BUNDLE = [
    "code/common.py", "code/bilateral.py", "code/original_reward.py",
    "code/env60.py", "code/sac_agent.py", "code/ddpg_agent.py",
    "code/run_sac.py", "code/run_ddpg.py", "code/run_bo.py",
    "configs/BILATERAL_EXPERIMENT_PROTOCOL_LOCK.yaml",
]
ASSET_PATHS = {
    key: row["path"] for key, row in CFG["assets"].items()
}


def sha256(path) -> str | None:
    p = Path(path)
    if not p.is_file():
        return None
    h = hashlib.sha256()
    with p.open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def tensor_sha256(tensor: torch.Tensor) -> str:
    a = tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(a.tobytes()).hexdigest()


def atomic_write_json(path, value) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(p) + ".tmp")

    def default(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, Path):
            return str(obj)
        raise TypeError(type(obj).__name__)

    tmp.write_text(json.dumps(value, indent=2, default=default,
                              ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, p)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    configure_determinism()


def det_schedule(episodes: int = EPISODES) -> list[int]:
    return [0] + list(range(DET_EVERY, episodes + 1, DET_EVERY))


def machine_seeds(machine: str) -> list[int]:
    return list(MACHINE_SEEDS[machine])


def neural_run_dir(domain: str, task: str, method: str, seed: int,
                   root: Path = OUTPUT) -> Path:
    return Path(root) / "runs" / domain / task / method / f"seed_{seed}"


def checkpoint_dir(domain: str, task: str, method: str, seed: int,
                   root: Path = OUTPUT) -> Path:
    return Path(root) / "checkpoints" / domain / task / method / f"seed_{seed}"


def bo_run_dir(task: str, seed: int, root: Path = OUTPUT) -> Path:
    return Path(root) / "bo_runs" / "upper" / task / "BO" / f"seed_{seed}"


def scratch_path(algorithm: str, seed: int, role: str = "target",
                 root: Path = OUTPUT) -> Path:
    if algorithm == "SAC":
        return Path(root) / "scratch_bank" / "sac" / f"seed_{seed}" / "scratch_init.pt"
    return (Path(root) / "scratch_bank" / "ddpg" / role
            / f"seed_{seed}" / "ddpg_init.pt")


def source_checkpoint(algorithm: str, task: str, seed: int,
                      root: Path = OUTPUT) -> Path:
    return (Path(root) / "source_bank" / algorithm.lower() / task
            / f"seed_{seed}.pt")


def domain_assets(domain: str, root: Path = ROOT) -> tuple[Path, Path]:
    prefix = "source" if domain == "source" else "upper"
    return (Path(root) / CFG["assets"][f"{prefix}_surrogate"]["path"],
            Path(root) / CFG["assets"][f"{prefix}_scaler"]["path"])


def make_env(domain: str, task: str, device, root: Path = ROOT) -> Environment:
    surrogate, scaler = domain_assets(domain, root)
    return Environment(
        case="zigzag", model_row=10, model_col=3,
        surrogate_path=str(surrogate), y_scaler_path=str(scaler),
        reward_function=REWARDS[task], device=device,
        leakyrelu_para=0.005, domain=domain)


def sac_agent_cfg() -> dict:
    c = CFG["sac"]
    return {
        "train": {
            "gamma": float(c["gamma"]), "tau": float(c["tau"]),
            "memory_capacity": int(c["replay_capacity"]),
        },
        "sac": {
            "reward_scale": 1.0, "autotune_alpha": True,
            "log_std_min": -20.0, "log_std_max": 2.0,
            "weight_decay": float(c["weight_decay"]),
            "actor_lr": float(c["actor_lr"]),
            "critic_lr": float(c["critic_lr"]),
            "alpha_lr": float(c["alpha_lr"]),
            "target_entropy": float(c["target_entropy"]),
            "init_alpha": float(c["initial_alpha"]),
        },
    }


def verify_assets(root: Path = ROOT) -> dict:
    result = {}
    bad = []
    for key, row in CFG["assets"].items():
        actual = sha256(Path(root) / row["path"])
        ok = actual == row["sha256"]
        result[key] = {
            "path": str(Path(root) / row["path"]),
            "required_sha256": row["sha256"],
            "actual_sha256": actual, "ok": ok,
        }
        if not ok:
            bad.append(key)
    if bad:
        raise SystemExit(f"asset hash mismatch: {bad}")
    return result


def code_hashes(root: Path = ROOT) -> dict:
    return {rel: sha256(Path(root) / rel) for rel in CODE_BUNDLE}


def protocol_sha256(root: Path = ROOT) -> str:
    return sha256(Path(root) / "configs" / "BILATERAL_EXPERIMENT_PROTOCOL_LOCK.yaml")


def state_map_sha256() -> str:
    env = {
        "rows": 10, "cols": 3, "case": "zigzag",
        "state": "30x[action,normalized_location]_interleaved_row_major",
        "location": "sqrt(((row+1)*5)^2+((col+1)*5)^2)/sqrt(50^2+15^2)",
    }
    return sha256_text(json.dumps(env, sort_keys=True))


def run_metrics(reward_history, deterministic_records) -> dict:
    r = np.asarray(reward_history, dtype=np.float64)
    det = sorted(deterministic_records, key=lambda x: int(x["episode"]))
    early = [x["bilateral_reward"] for x in det if int(x["episode"]) <= 300]
    all_det = [x["bilateral_reward"] for x in det]
    first200 = list(r[:180])
    return {
        "final100": float(np.mean(r[-100:])),
        "AUC1500": float(np.mean(r)),
        "early300_AUC": float(np.mean(r[:300])),
        "zero_shot_deterministic": float(det[0]["bilateral_reward"]),
        "early300_deterministic_AUC": float(np.mean(early)),
        "best_deterministic": float(max(all_det)),
        "final_deterministic": float(det[-1]["bilateral_reward"]),
        "first200_best_observed": float(max(first200)) if first200 else None,
    }


def machine_metadata(machine: str) -> dict:
    gpu = None
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
    versions = {}
    for module in ("numpy", "sklearn", "skopt"):
        try:
            m = __import__(module)
            versions[module] = getattr(m, "__version__", "unknown")
        except Exception as exc:
            versions[module] = f"unavailable:{type(exc).__name__}"
    return {
        "machine": machine, "hostname": platform.node(),
        "platform": platform.platform(), "python": sys.version,
        "executable": sys.executable, "torch": torch.__version__,
        "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
        "gpu": gpu, **versions,
    }


def command_output(command: list[str]) -> str:
    return subprocess.check_output(command, text=True).strip()
