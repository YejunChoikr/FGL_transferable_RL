"""Run one locked 200-call bilateral Upper Bayesian-optimization experiment."""
from __future__ import annotations

import argparse
import csv
import json
import time

import numpy as np
import sklearn
import skopt
import torch
import yaml
from skopt import gp_minimize
from skopt.space import Real

from bilateral import metrics
from common import (
    CFG, atomic_write_json, bo_run_dir, code_hashes, domain_assets,
    machine_metadata, machine_seeds, make_env, protocol_sha256, set_seed,
    sha256, state_map_sha256, verify_assets,
)


def run(task: str, seed: int, machine: str) -> dict:
    wdir = bo_run_dir(task, seed)
    completion_path = wdir / "completion.json"
    if completion_path.is_file():
        old = json.loads(completion_path.read_text(encoding="utf-8"))
        hist = wdir / "objective_history.csv"
        if old.get("complete") and hist.is_file():
            with hist.open(encoding="utf-8") as stream:
                if sum(1 for _ in stream) == 201:
                    print(f"[upper/{task}/BO/seed{seed}] RESUME-SKIP complete")
                    return old
    wdir.mkdir(parents=True, exist_ok=True)
    verify_assets()
    set_seed(seed)
    env = make_env("upper", task, torch.device("cpu"))
    records = []

    def objective(action):
        reward, _ = env.evaluate_action_vector(np.asarray(action, dtype=np.float64))
        profile = env.surrogate_profile()
        row = metrics(profile, action, task)
        row["evaluation"] = len(records) + 1
        row["actions"] = [float(x) for x in action]
        row["objective_minimize"] = -float(reward)
        records.append(row)
        return -float(reward)

    started = time.strftime("%Y-%m-%dT%H:%M:%S")
    t0 = time.perf_counter()
    dimensions = [Real(-1.0, 1.0, name=f"a{i + 1}") for i in range(30)]
    result = gp_minimize(
        objective, dimensions, n_calls=200, n_initial_points=20,
        initial_point_generator="random", acq_func="gp_hedge",
        random_state=seed, verbose=False)
    wall = time.perf_counter() - t0
    if len(records) != 200 or not np.isfinite([r["bilateral_reward"]
                                               for r in records]).all():
        raise SystemExit(f"BO history invalid: n={len(records)}")

    best_values = []
    running = -np.inf
    for row in records:
        running = max(running, row["bilateral_reward"])
        best_values.append(running)
    best_index = int(np.argmax([r["bilateral_reward"] for r in records]))
    best = dict(records[best_index])
    best["best_so_far"] = best_values[best_index]
    atomic_write_json(wdir / "best_design.json", best)
    atomic_write_json(wdir / "final_design.json", records[-1])
    atomic_write_json(wdir / "objective_history.json", {
        "n_calls": 200, "records": records, "best_so_far": best_values})

    fixed = [
        "evaluation", "objective_minimize", "bilateral_reward",
        "bilateral_margin", "left_margin", "right_margin", "original_reward",
        "kappa", "rho_A", "density_code", "local_peak", "global_peak",
        "global_peak_index", "global_margin", "left_right_asymmetry",
        "action_sha256",
    ]
    with (wdir / "objective_history.csv").open(
            "w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(fixed + [f"u{i}" for i in range(1, 10)]
                        + [f"a{i}" for i in range(1, 31)])
        for row in records:
            writer.writerow([row.get(k) for k in fixed]
                            + row["u"] + row["actions"])
    with (wdir / "best_so_far.csv").open(
            "w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["evaluation", "best_bilateral_reward"])
        writer.writerows(enumerate(best_values, start=1))

    surrogate, scaler = domain_assets("upper")
    kernel_repr = repr(result.models[-1].kernel_) if result.models else None
    base_estimator_repr = repr(result.models[-1]) if result.models else None
    protocol = {
        "implementation": "skopt.gp_minimize",
        "python_package_versions": {
            "scikit_optimize": skopt.__version__,
            "scikit_learn": sklearn.__version__,
            "numpy": np.__version__,
        },
        "dimensions": 30, "bounds": [-1.0, 1.0],
        "acq_func": "gp_hedge", "n_initial_points": 20,
        "n_calls": 200, "initial_point_generator": "random",
        "random_state": seed, "base_estimator": base_estimator_repr,
        "kernel": kernel_repr,
        "kernel_hyperparameter_bounds": [
            str(x.bounds) for x in result.models[-1].kernel_.hyperparameters
        ] if result.models else [],
        "optimizer_settings": {
            "acq_optimizer": "auto", "noise": "gaussian",
            "model_queue_size": None,
        },
        "termination_reason": "completed_prespecified_200_calls",
    }
    (wdir / "config_resolved.yaml").write_text(
        yaml.safe_dump({"task": task, "seed": seed, "bo": protocol,
                        "reward": CFG["reward"]}, sort_keys=False),
        encoding="utf-8")
    provenance = {
        "protocol_sha256": protocol_sha256(), "code_sha256": code_hashes(),
        "surrogate_sha256": sha256(surrogate),
        "scaler_sha256": sha256(scaler),
        "state_map_sha256": state_map_sha256(),
        "machine": machine_metadata(machine),
        "reuse": False,
        "reuse_gate": "not accepted: historical run lacks a complete exact "
        "protocol/hash record under the successor native lock; BO rerun",
    }
    atomic_write_json(wdir / "provenance.json", provenance)
    manifest = {
        "run_key": f"upper__{task}__BO__seed{seed}",
        "domain": "upper", "task": task, "algorithm": "BO",
        "method": "BO", "seed": seed, "machine": machine,
        "state_dim": 60, "action_dim": 30,
        "selected_evaluation": best_index + 1,
        "selected_action_sha256": best["action_sha256"],
        "bo_protocol": protocol, **provenance,
    }
    atomic_write_json(wdir / "run_manifest.json", manifest)
    completion = {
        "run_key": manifest["run_key"], "domain": "upper", "task": task,
        "algorithm": "BO", "method": "BO", "seed": seed,
        "machine": machine, "started": started,
        "finished": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "wall_seconds": wall, "n_objective_calls": len(records),
        "history_finite": bool(np.isfinite(
            [r["bilateral_reward"] for r in records]).all()),
        "selected_evaluation": best_index + 1,
        "selected_bilateral_reward": best["bilateral_reward"],
        "selected_action_sha256": best["action_sha256"],
        "termination_reason": "completed_prespecified_200_calls",
        "complete": len(records) == 200,
    }
    atomic_write_json(completion_path, completion)
    print(f"[BO/{task}/s{seed}] DONE wall={wall:.1f}s "
          f"best={best['bilateral_reward']:.6f}@{best_index + 1}")
    return completion


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--machine", required=True,
                        choices=["local", "server", "workstation"])
    parser.add_argument("--task", required=True,
                        choices=["FGL5", "FGL6", "FGL7"])
    parser.add_argument("--seed", required=True, type=int)
    args = parser.parse_args()
    if args.seed not in machine_seeds(args.machine):
        raise SystemExit(f"seed {args.seed} not allocated to {args.machine}")
    run(args.task, args.seed, args.machine)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
