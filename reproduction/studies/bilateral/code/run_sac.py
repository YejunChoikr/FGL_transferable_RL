"""Run one locked bilateral SAC source, RL, or TRL experiment."""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from bilateral import metrics
from common import (
    CFG, EPISODES, ROOT, atomic_write_json, checkpoint_dir, code_hashes,
    det_schedule, domain_assets, machine_seeds, make_env, machine_metadata,
    neural_run_dir, protocol_sha256, run_metrics, sac_agent_cfg, scratch_path,
    set_seed, sha256, sha256_text, source_checkpoint, state_map_sha256, tensor_sha256,
    verify_assets,
)
from sac_agent import SACAgent

VALID_METHODS = ("SAC-SOURCE", "SAC-RL", "SAC-TRL")
ACTOR_LOAD = ("fc1", "fc2", "fc3", "fc4")
CRITIC_LOAD = ("fc1", "fc2", "fc3", "fc4")


class Tee:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.file = path.open("a", encoding="utf-8")
        self.stdout = sys.stdout

    def write(self, value):
        self.stdout.write(value)
        self.file.write(value)
        self.file.flush()

    def flush(self):
        self.stdout.flush()
        self.file.flush()

    def close(self):
        self.file.close()


def cpu_state(module):
    return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}


def state_hashes(module):
    return {k: tensor_sha256(v) for k, v in module.state_dict().items()}


def load_scratch(agent: SACAgent, path: Path) -> dict:
    blob = torch.load(path, map_location=agent.device, weights_only=False)
    for name in ("actor", "q1", "q2", "q1_target", "q2_target"):
        getattr(agent, name).load_state_dict(blob[name])
    with torch.no_grad():
        agent.log_alpha.copy_(blob["log_alpha"].to(agent.device))
    return blob


def copy_layers(module, source_state, prefixes):
    current = module.state_dict()
    loaded = []
    for key, value in source_state.items():
        if key.split(".")[0] in prefixes:
            if key not in current or current[key].shape != value.shape:
                raise SystemExit(f"transfer shape/key mismatch: {key}")
            current[key] = value.clone()
            loaded.append(key)
    module.load_state_dict(current)
    return sorted(loaded)


def initialize(task: str, method: str, seed: int, device):
    set_seed(seed)
    agent = SACAgent(60, device, sac_agent_cfg())
    scratch = scratch_path("SAC", seed)
    scratch_blob = load_scratch(agent, scratch)
    before = {
        name: {k: tensor_sha256(v) for k, v in scratch_blob[name].items()}
        for name in ("actor", "q1", "q2")
    }
    loaded = {"actor": [], "q1": [], "q2": []}
    src_path = None
    src_blob = None
    if method == "SAC-TRL":
        src_path = source_checkpoint("SAC", task, seed)
        if not src_path.is_file():
            raise SystemExit(f"missing task/seed source checkpoint: {src_path}")
        src_blob = torch.load(src_path, map_location=device, weights_only=False)
        if src_blob.get("task") != task or int(src_blob.get("seed", -1)) != seed:
            raise SystemExit("source task/seed metadata mismatch")
        loaded["actor"] = copy_layers(agent.actor, src_blob["actor"], ACTOR_LOAD)
        loaded["q1"] = copy_layers(agent.q1, src_blob["q1"], CRITIC_LOAD)
        loaded["q2"] = copy_layers(agent.q2, src_blob["q2"], CRITIC_LOAD)
        agent.q1_target.load_state_dict(agent.q1.state_dict())
        agent.q2_target.load_state_dict(agent.q2.state_dict())

    expected = {"actor": 8 if method == "SAC-TRL" else 0,
                "q1": 8 if method == "SAC-TRL" else 0,
                "q2": 8 if method == "SAC-TRL" else 0}
    problems = []
    now = {n: cpu_state(getattr(agent, n)) for n in ("actor", "q1", "q2")}
    for net in now:
        if len(loaded[net]) != expected[net]:
            problems.append(f"{net}:loaded={len(loaded[net])}:expected={expected[net]}")
        for key, value in now[net].items():
            actual = tensor_sha256(value)
            if key in loaded[net]:
                required = tensor_sha256(src_blob[net][key])
                origin = "bilateral_source"
            else:
                required = before[net][key]
                origin = "paired_scratch"
            if actual != required:
                problems.append(f"{net}.{key}:{origin}")
    for online, target in (("q1", "q1_target"), ("q2", "q2_target")):
        a, b = state_hashes(getattr(agent, online)), state_hashes(getattr(agent, target))
        if a != b:
            problems.append(f"{target}_not_hard_copy")
    optimizer_fresh = all(len(x.state) == 0 for x in
                          (agent.actor_opt, agent.q1_opt, agent.q2_opt,
                           agent.alpha_opt))
    if not optimizer_fresh or len(agent.replay) != 0:
        problems.append("optimizer_or_replay_not_fresh")
    audit = {
        "algorithm": "SAC", "method": method, "task": task, "seed": seed,
        "scratch_path": str(scratch), "scratch_sha256": sha256(scratch),
        "source_path": str(src_path) if src_path else None,
        "source_sha256": sha256(src_path) if src_path else None,
        "loaded_tensors": loaded, "expected_counts": expected,
        "post_load_tensor_sha256": {n: state_hashes(getattr(agent, n))
                                    for n in ("actor", "q1", "q2")},
        "target_critics": "hard_copy_after_assembly",
        "optimizers_fresh": optimizer_fresh,
        "replay_empty": len(agent.replay) == 0,
        "alpha_initial": float(agent.alpha.item()),
        "no_freeze": all(p.requires_grad for n in ("actor", "q1", "q2")
                         for p in getattr(agent, n).parameters()),
        "problems": problems, "pass": not problems,
    }
    if problems:
        raise SystemExit(f"SAC initialization audit failed: {problems[:5]}")
    return agent, audit


def rollout(env, agent: SACAgent, task: str, episode: int) -> dict:
    state = env.reset()
    actions = []
    for _ in range(30):
        action = agent.deterministic_action(state)
        actions.append(float(action[0]))
        state, _, _ = env.replay_step(np.asarray(action, dtype=np.float64))
    out = metrics(env.surrogate_profile(), actions, task)
    out.update({"episode": int(episode), "actions": actions})
    out["actor_checkpoint_identity"] = sha256_text(json.dumps(
        state_hashes(agent.actor), sort_keys=True))
    return out


def q_disagreement(agent: SACAgent, batch_size: int) -> float:
    if len(agent.replay) < batch_size:
        return float("nan")
    state, action, _, _, _ = agent.replay.sample(batch_size, agent.device)
    with torch.no_grad():
        return float((agent.q1(state, action) -
                      agent.q2(state, action)).abs().mean().item())


def design_from_terminal(env, actions, task):
    return metrics(env.surrogate_profile(), actions, task)


def save_compact(agent, path: Path, task: str, method: str, seed: int,
                 episode: int, selected: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "actor": cpu_state(agent.actor), "q1": cpu_state(agent.q1),
        "q2": cpu_state(agent.q2), "task": task, "method": method,
        "seed": seed, "episode": int(episode),
        "bilateral_reward": float(selected["bilateral_reward"]),
        "action_sha256": selected["action_sha256"],
        "protocol_sha256": protocol_sha256(),
    }, path)


def write_det_csv(path: Path, records: list[dict]):
    fixed = [
        "episode", "bilateral_reward", "bilateral_margin", "left_margin",
        "right_margin", "original_reward", "kappa", "rho_A", "density_code",
        "local_peak", "global_peak", "global_peak_index", "global_margin",
        "left_right_asymmetry", "action_sha256", "actor_checkpoint_identity",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(fixed + [f"u{i}" for i in range(1, 10)]
                        + [f"a{i}" for i in range(1, 31)])
        for row in records:
            writer.writerow([row.get(k) for k in fixed] + row["u"] + row["actions"])


def train(task: str, method: str, seed: int, machine: str) -> dict:
    if method not in VALID_METHODS:
        raise SystemExit(f"unknown SAC method {method}")
    domain = "source" if method == "SAC-SOURCE" else "upper"
    wdir = neural_run_dir(domain, task, method, seed)
    cdir = checkpoint_dir(domain, task, method, seed)
    completion = wdir / "completion.json"
    if completion.is_file():
        old = json.loads(completion.read_text(encoding="utf-8"))
        reward_path = wdir / "reward_history.npy"
        if old.get("complete") and reward_path.is_file() and len(np.load(reward_path)) == EPISODES:
            print(f"[{domain}/{task}/{method}/seed{seed}] RESUME-SKIP complete")
            return old
    verify_assets()
    wdir.mkdir(parents=True, exist_ok=True)
    cdir.mkdir(parents=True, exist_ok=True)
    tee = Tee(wdir / "training.log")
    previous_stdout = sys.stdout
    sys.stdout = tee
    try:
        started = time.strftime("%Y-%m-%dT%H:%M:%S")
        t0 = time.perf_counter()
        device = torch.device("cuda")
        torch.cuda.reset_peak_memory_stats()
        env = make_env(domain, task, device)
        agent, init_audit = initialize(task, method, seed, device)
        atomic_write_json(wdir / "initialization_audit.json", init_audit)
        atomic_write_json(wdir / "transfer_tensor_audit.json", init_audit)
        print(f"[{domain}/{task}/{method}/seed{seed}] start machine={machine}")

        deterministic = [rollout(env, agent, task, 0)]
        atomic_write_json(wdir / "zero_shot_design.json", deterministic[0])
        best = dict(deterministic[0])
        save_compact(agent, wdir / "compact_best_policy.pt", task, method,
                     seed, 0, best)

        fill_metrics = []
        for _ in range(int(CFG["sac"]["memory_fill_designs"])):
            state = env.reset()
            actions = []
            done = False
            while not done:
                action = agent.select_action(state)
                actions.append(float(action[0]))
                next_state, reward, done = env.replay_step(
                    np.asarray(action, dtype=np.float64))
                agent.replay.store(state, action, next_state, reward, done)
                state = next_state
            fill_metrics.append(design_from_terminal(env, actions, task))

        histories = {
            "reward": [], "left_margin": [], "right_margin": [],
            "bilateral_margin": [], "original_reward": [], "kappa": [],
            "rho_A": [], "local_peak": [], "global_peak": [],
            "alpha": [], "entropy": [], "action_std": [],
            "q_disagreement": [], "q1_loss": [], "q2_loss": [],
            "actor_loss": [], "alpha_loss": [],
        }
        batch = int(CFG["sac"]["batch_size"])
        updates = 0
        for episode in range(1, EPISODES + 1):
            state = env.reset()
            actions = []
            step_diag = {k: [] for k in
                         ("q1_loss", "q2_loss", "actor_loss",
                          "alpha_loss", "entropy", "alpha")}
            done = False
            terminal_reward = 0.0
            while not done:
                action = agent.select_action(state)
                actions.append(float(action[0]))
                next_state, reward, done = env.replay_step(
                    np.asarray(action, dtype=np.float64))
                agent.replay.store(state, action, next_state, reward, done)
                info = agent.update(batch)
                if info:
                    updates += 1
                    for key in step_diag:
                        step_diag[key].append(info[key])
                terminal_reward = float(reward) if done else terminal_reward
                state = next_state
            physical = design_from_terminal(env, actions, task)
            histories["reward"].append(terminal_reward)
            for key in ("left_margin", "right_margin", "bilateral_margin",
                        "original_reward", "kappa", "rho_A",
                        "local_peak", "global_peak"):
                histories[key].append(physical[key])
            histories["action_std"].append(float(np.std(actions)))
            histories["q_disagreement"].append(q_disagreement(agent, batch))
            for key in ("q1_loss", "q2_loss", "actor_loss", "alpha_loss",
                        "entropy", "alpha"):
                histories[key].append(
                    float(np.mean(step_diag[key])) if step_diag[key] else float("nan"))

            if episode % 25 == 0:
                record = rollout(env, agent, task, episode)
                deterministic.append(record)
                if record["bilateral_reward"] > best["bilateral_reward"]:
                    best = dict(record)
                    save_compact(agent, wdir / "compact_best_policy.pt", task,
                                 method, seed, episode, best)
            if episode % 250 == 0:
                print(f"[{method}/{task}/s{seed}] ep={episode} "
                      f"r={terminal_reward:.6f} "
                      f"best={best['bilateral_reward']:.6f}@{best['episode']}")

        reward_history = np.asarray(histories["reward"], dtype=np.float64)
        np.save(wdir / "reward_history.npy", reward_history)
        np.savetxt(wdir / "reward_history.txt", reward_history, fmt="%.12g")
        np.savez(wdir / "physical_metric_history.npz", **{
            k: np.asarray(histories[k]) for k in
            ("left_margin", "right_margin", "bilateral_margin",
             "original_reward", "kappa", "rho_A", "local_peak", "global_peak")
        })
        np.savez(wdir / "loss_history.npz", **{
            k: np.asarray(histories[k], dtype=np.float64) for k in
            ("q1_loss", "q2_loss", "actor_loss", "alpha_loss")
        })
        for key in ("alpha", "entropy", "action_std", "q_disagreement"):
            np.save(wdir / f"{key}_history.npy",
                    np.asarray(histories[key], dtype=np.float64))
        write_det_csv(wdir / "deterministic_history.csv", deterministic)
        atomic_write_json(wdir / "deterministic_history.json", {
            "schedule": det_schedule(), "selection_rule":
            "highest deterministic bilateral surrogate reward; tie earlier episode",
            "records": deterministic,
        })
        atomic_write_json(wdir / "best_design.json", best)
        atomic_write_json(wdir / "final_design.json", deterministic[-1])

        first200 = fill_metrics + [
            {"bilateral_reward": float(x)} for x in reward_history[:180]]
        first200_best = max(first200, key=lambda x: x["bilateral_reward"])
        run_met = run_metrics(reward_history, deterministic)
        run_met["first200_best_observed"] = float(
            first200_best["bilateral_reward"])
        run_met["first200_n"] = len(first200)
        surrogate, scaler = domain_assets(domain)
        resolved = {
            "domain": domain, "task": task, "algorithm": "SAC",
            "method": method, "seed": seed, "sac": CFG["sac"],
            "training": CFG["training"], "reward": CFG["reward"],
        }
        (wdir / "config_resolved.yaml").write_text(
            yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8")
        provenance = {
            "protocol_sha256": protocol_sha256(),
            "code_sha256": code_hashes(),
            "surrogate": str(surrogate), "surrogate_sha256": sha256(surrogate),
            "scaler": str(scaler), "scaler_sha256": sha256(scaler),
            "scratch_sha256": init_audit["scratch_sha256"],
            "source_sha256": init_audit["source_sha256"],
            "state_map_sha256": state_map_sha256(),
            "machine": machine_metadata(machine),
        }
        atomic_write_json(wdir / "provenance.json", provenance)
        compact_hash = sha256(wdir / "compact_best_policy.pt")
        manifest = {
            "run_key": f"{domain}__{task}__{method}__seed{seed}",
            "domain": domain, "task": task, "algorithm": "SAC",
            "method": method, "seed": seed, "machine": machine,
            "source_task": task if method == "SAC-TRL" else None,
            "source_seed": seed if method == "SAC-TRL" else None,
            "state_dim": 60, "critic_input_dim": 61,
            "source_target_pairing": "identity" if method == "SAC-TRL" else None,
            "selection_episode": int(best["episode"]),
            "selected_action_sha256": best["action_sha256"],
            "compact_best_policy_sha256": compact_hash,
            "hyperparameters": CFG["sac"], **provenance,
        }
        atomic_write_json(wdir / "run_manifest.json", manifest)
        atomic_write_json(wdir / "source_target_pairing.json", {
            "task": task, "seed": seed,
            "source_task": task if method == "SAC-TRL" else None,
            "source_seed": seed if method == "SAC-TRL" else None,
            "pairing": "identity" if method == "SAC-TRL" else "not_applicable",
        })
        wall = time.perf_counter() - t0
        result = {
            "run_key": manifest["run_key"], "domain": domain, "task": task,
            "algorithm": "SAC", "method": method, "seed": seed,
            "machine": machine, "episodes": EPISODES,
            "started": started, "finished": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "wall_seconds": wall, "gpu_peak_gb":
            torch.cuda.max_memory_allocated() / 1e9,
            "n_deterministic": len(deterministic), "updates": updates,
            "metrics": run_met, "selected_episode": int(best["episode"]),
            "selected_bilateral_reward": float(best["bilateral_reward"]),
            "selected_action_sha256": best["action_sha256"],
            "reward_history_len": len(reward_history),
            "reward_history_finite": bool(np.isfinite(reward_history).all()),
            "complete": bool(
                len(reward_history) == EPISODES
                and np.isfinite(reward_history).all()
                and [x["episode"] for x in deterministic] == det_schedule()),
        }
        atomic_write_json(completion, result)
        print(f"[{method}/{task}/s{seed}] DONE wall={wall:.0f}s "
              f"best={best['bilateral_reward']:.6f}@{best['episode']}")
        del agent, env
        torch.cuda.empty_cache()
        return result
    finally:
        sys.stdout = previous_stdout
        tee.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--machine", required=True,
                        choices=["local", "server", "workstation"])
    parser.add_argument("--task", required=True, choices=["FGL5", "FGL6", "FGL7"])
    parser.add_argument("--method", required=True, choices=VALID_METHODS)
    parser.add_argument("--seed", required=True, type=int)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required; CPU fallback is forbidden")
    if args.seed not in machine_seeds(args.machine):
        raise SystemExit(f"seed {args.seed} not allocated to {args.machine}")
    train(args.task, args.method, args.seed, args.machine)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
