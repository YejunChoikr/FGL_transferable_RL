"""Run the 105 registered S10 experiments with their original algorithms."""
from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "code"))
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def cases():
    rows = []
    for g in [5, 6, 7]:
        for seed in range(5):
            for domain, methods in [("source", ["SAC-SOURCE", "DDPG-SOURCE"]),
                ("upper", ["SAC-RL", "SAC-TRL", "DDPG-RL", "DDPG-TRL", "BO"])]:
                for method in methods:
                    rows.append({"case_id": f"{domain}/FGL{g}/{method}/s{seed}",
                                 "domain": domain, "task": f"FGL{g}",
                                 "method": method, "seed": seed})
    return rows


def prepare_scratch(algo, seed):
    """Reconstruct the original bank and verify every tensor against run audits."""
    import torch
    from common import set_seed, scratch_path, sac_agent_cfg, tensor_sha256
    from env60 import SurrogateCNN
    expected = json.loads((ROOT/"configs/initial_tensor_hashes.json").read_text(encoding="utf-8"))
    set_seed(seed)
    if algo == "SAC":
        from sac_agent import SACAgent
        # The original bank constructed the CNN before constructing the agent.
        # Preserve that CPU RNG consumption exactly; hashes verify the result.
        SurrogateCNN(9, .005)
        agent = SACAgent(60, torch.device("cpu"), sac_agent_cfg())
        blob = {n: getattr(agent, n).state_dict() for n in
                ["actor", "q1", "q2", "q1_target", "q2_target"]}
        blob["log_alpha"] = agent.log_alpha.detach().cpu().clone()
    else:
        from run_ddpg import new_agent
        agent = new_agent(seed, torch.device("cpu"))
        blob = agent.state_dicts()
    for role in ["source", "target"]:
        hashes = expected[f"{algo}/{role}/s{seed}"]
        for network, tensors in hashes.items():
            for key, expected_hash in tensors.items():
                if tensor_sha256(blob[network][key]) != expected_hash:
                    raise RuntimeError(f"initial tensor mismatch: {algo}/{role}/{seed}/{network}/{key}")
        path = scratch_path(algo, seed, role)
        if path.exists():
            old = torch.load(path, map_location="cpu", weights_only=True)
            for network, tensors in hashes.items():
                for key, expected_hash in tensors.items():
                    if tensor_sha256(old[network][key]) != expected_hash:
                        raise RuntimeError(f"modified initial tensor bank: {path}")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(blob, path)


def execute(case_id, dependencies=False, smoke=False):
    from common import neural_run_dir, source_checkpoint, bo_run_dir
    table = {c["case_id"]: c for c in cases()}
    case = table[case_id]
    task, method, seed = case["task"], case["method"], case["seed"]
    algo = method.split("-")[0]
    if method.endswith("-TRL"):
        src = f"source/{task}/{algo}-SOURCE/s{seed}"
        if dependencies:
            execute(src, True, smoke)
        if not source_checkpoint(algo, task, seed).exists():
            raise RuntimeError(f"run {src} first, or use --with-dependencies")
    out = (bo_run_dir(task, seed) if method == "BO" else
           neural_run_dir(case["domain"], task, method, seed))
    if out.exists() and any(out.iterdir()) and not (out/"completion.json").exists():
        raise RuntimeError(f"incomplete output exists: {out}; preserve it before retrying")
    if method == "DDPG-TRL":
        # The manuscript uses the later actor-fc1..4 / critic-fc1..4 experiment.
        # Run it in a separate process to preserve its module names and RNG order.
        if (out/"completion.json").exists():
            print(f"Already complete: {case_id}")
            return
        cmd = [sys.executable, str(ROOT/"ddpg_c4/run.py"), "--task", task,
               "--seed", str(seed), "--output", str(out)]
        if smoke:
            cmd.append("--smoke")
        subprocess.run(cmd, check=True)
        print(f"Completed {case_id}")
        return
    if method == "BO":
        if smoke:
            raise ValueError("BO uses its complete 200-evaluation protocol; use the BO integration test for a short test")
        result = importlib.import_module("run_bo").run(task, seed, "local")
    else:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("the preserved SAC/DDPG training runners require CUDA")
        prepare_scratch(algo, seed)
        module = importlib.import_module("run_" + algo.lower())
        if smoke:
            module.EPISODES = 25
            module.det_schedule = lambda: [0, 25]
        result = module.train(task, method, seed, "local")
    if not result["complete"]:
        raise RuntimeError(f"incomplete run: {case_id}")
    mode = {"smoke": smoke, "case_id": case_id}
    if method == "BO":
        mode["objective_evaluations"] = 200
    else:
        mode.update(training_episodes=25 if smoke else 1500, prefill_episodes=20)
    (out/"execution_mode.json").write_text(json.dumps(mode, indent=2), encoding="utf-8")
    if case["domain"] == "source":
        dest = source_checkpoint(algo, task, seed)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(out/"compact_best_policy.pt", dest)
    print(f"Completed {case_id}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["list", "run", "smoke", "verify-init"])
    p.add_argument("--case")
    p.add_argument("--with-dependencies", action="store_true")
    args = p.parse_args()
    if args.command == "list":
        print("\n".join(c["case_id"] for c in cases()))
        return
    if args.command == "smoke":
        os.environ["FGL_BILATERAL_OUTPUT"] = str(ROOT/"pilot")
    if args.command == "verify-init":
        for algo in ["SAC", "DDPG"]:
            for seed in range(5):
                prepare_scratch(algo, seed)
        print("All original source/target initial tensors match the archived run audits.")
        return
    if args.case is None:
        p.error("--case is required")
    execute(args.case, args.with_dependencies or args.command == "smoke", args.command == "smoke")


if __name__ == "__main__":
    main()
