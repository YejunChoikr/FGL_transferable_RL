"""Run one locked bilateral DDPG source, RL, or native-recipe TRL experiment."""
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
    CFG, EPISODES, atomic_write_json, checkpoint_dir, code_hashes,
    det_schedule, domain_assets, machine_metadata, machine_seeds, make_env,
    neural_run_dir, protocol_sha256, run_metrics, scratch_path, set_seed,
    sha256, sha256_text, source_checkpoint, state_map_sha256, tensor_sha256,
    verify_assets,
)
from ddpg_agent import DDPGAgent

VALID_METHODS = ("DDPG-SOURCE", "DDPG-RL", "DDPG-TRL")
ACTOR_LOAD = ("fc1", "fc2", "fc3", "fc4")
CRITIC_LOAD = ("fc1", "fc2", "fc3")


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


def new_agent(seed: int, device):
    c = CFG["ddpg"]
    return DDPGAgent(
        60, device, actor_lr=float(c["actor_lr"]),
        critic_lr=float(c["critic_lr"]), weight_decay=float(c["weight_decay"]),
        gamma=float(c["gamma"]), tau=float(c["tau"]),
        batch_size=int(c["batch_size"]),
        memory_capacity=int(c["replay_capacity"]),
        ou=(float(c["ou"]["mu"]), float(c["ou"]["theta"]),
            float(c["ou"]["sigma"])),
        noise_rng=np.random.default_rng(seed))


def initialize(task: str, method: str, seed: int, device):
    set_seed(seed)
    agent = new_agent(seed, device)
    role = "source" if method == "DDPG-SOURCE" else "target"
    scratch = scratch_path("DDPG", seed, role=role)
    scratch_blob = torch.load(scratch, map_location=device, weights_only=False)
    agent.load_state_dicts(scratch_blob)
    before = {n: state_hashes(getattr(agent, n)) for n in ("actor", "critic")}
    source_path = None
    source_blob = None
    loaded = {"actor": [], "critic": []}
    if method == "DDPG-TRL":
        source_path = source_checkpoint("DDPG", task, seed)
        if not source_path.is_file():
            raise SystemExit(f"missing task/seed source checkpoint: {source_path}")
        source_blob = torch.load(source_path, map_location=device, weights_only=False)
        if source_blob.get("task") != task or int(source_blob.get("seed", -1)) != seed:
            raise SystemExit("source task/seed metadata mismatch")
        loaded["actor"] = copy_layers(agent.actor, source_blob["actor"], ACTOR_LOAD)
        loaded["critic"] = copy_layers(agent.critic, source_blob["critic"], CRITIC_LOAD)
        agent.hard_update()

    expected = {"actor": 8 if method == "DDPG-TRL" else 0,
                "critic": 6 if method == "DDPG-TRL" else 0}
    problems = []
    for net in ("actor", "critic"):
        if len(loaded[net]) != expected[net]:
            problems.append(f"{net}:loaded={len(loaded[net])}:expected={expected[net]}")
        current = getattr(agent, net).state_dict()
        for key, value in current.items():
            actual = tensor_sha256(value)
            required = (tensor_sha256(source_blob[net][key])
                        if key in loaded[net] else before[net][key])
            if actual != required:
                problems.append(f"{net}.{key}:origin_mismatch")
    for online, target in (("actor", "actor_target"),
                           ("critic", "critic_target")):
        if state_hashes(getattr(agent, online)) != state_hashes(getattr(agent, target)):
            problems.append(f"{target}_not_hard_copy")
    optimizer_fresh = len(agent.actor_opt.state) == 0 and len(agent.critic_opt.state) == 0
    if not optimizer_fresh or len(agent.memory) != 0:
        problems.append("optimizer_or_replay_not_fresh")
    audit = {
        "algorithm": "DDPG", "method": method, "task": task, "seed": seed,
        "scratch_role": role, "scratch_path": str(scratch),
        "scratch_sha256": sha256(scratch),
        "source_path": str(source_path) if source_path else None,
        "source_sha256": sha256(source_path) if source_path else None,
        "loaded_tensors": loaded, "expected_counts": expected,
        "post_load_tensor_sha256": {
            n: state_hashes(getattr(agent, n))
            for n in ("actor", "critic", "actor_target", "critic_target")},
        "targets": "hard_copy_after_assembly",
        "optimizers_fresh": optimizer_fresh,
        "replay_empty": len(agent.memory) == 0,
        "ou_state_fresh": bool(np.allclose(agent.noise.state, 0.0)),
        "no_freeze": all(p.requires_grad for n in ("actor", "critic")
                         for p in getattr(agent, n).parameters()),
        "actor_lr": [float(g["lr"]) for g in agent.actor_opt.param_groups],
        "critic_lr": [float(g["lr"]) for g in agent.critic_opt.param_groups],
        "weight_decay": [float(g["weight_decay"]) for opt in
                         (agent.actor_opt, agent.critic_opt)
                         for g in opt.param_groups],
        "problems": problems, "pass": not problems,
    }
    if problems:
        raise SystemExit(f"DDPG initialization audit failed: {problems[:5]}")
    return agent, audit


def rollout(env, agent: DDPGAgent, task: str, episode: int) -> dict:
    state = env.reset()
    actions = []
    for _ in range(30):
        action = agent.act(state, noise=False)
        actions.append(float(action[0]))
        state, _, _ = env.replay_step(np.asarray(action, dtype=np.float64))
    out = metrics(env.surrogate_profile(), actions, task)
    out.update({"episode": int(episode), "actions": actions})
    out["actor_checkpoint_identity"] = sha256_text(json.dumps(
        state_hashes(agent.actor), sort_keys=True))
    return out


def save_compact(agent, path: Path, task: str, method: str, seed: int,
                 episode: int, selected: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "actor": cpu_state(agent.actor), "critic": cpu_state(agent.critic),
        "task": task, "method": method, "seed": seed,
        "episode": int(episode),
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
        raise SystemExit(f"unknown DDPG method {method}")
    domain = "source" if method == "DDPG-SOURCE" else "upper"
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
        for _ in range(int(CFG["ddpg"]["memory_fill_designs"])):
            state = env.reset()
            agent.noise.reset()
            actions = []
            done = False
            while not done:
                action = agent.act(state, noise=True)
                actions.append(float(action[0]))
                next_state, reward, done = env.replay_step(
                    np.asarray(action, dtype=np.float64))
                agent.memory.push(state, action, reward, next_state, float(done))
                state = next_state
            fill_metrics.append(metrics(env.surrogate_profile(), actions, task))

        histories = {
            "reward": [], "left_margin": [], "right_margin": [],
            "bilateral_margin": [], "original_reward": [], "kappa": [],
            "rho_A": [], "local_peak": [], "global_peak": [],
            "critic_loss": [], "actor_loss": [], "q_mean": [],
            "q_target_mean": [], "ou_mean": [], "ou_std": [],
        }
        for episode in range(1, EPISODES + 1):
            state = env.reset()
            agent.noise.reset()
            actions, ou_values = [], []
            step_diag = {k: [] for k in
                         ("critic_loss", "actor_loss", "q_mean", "q_target_mean")}
            done = False
            terminal_reward = 0.0
            while not done:
                action = agent.act(state, noise=True)
                actions.append(float(action[0]))
                ou_values.append(float(agent.noise.state[0]))
                next_state, reward, done = env.replay_step(
                    np.asarray(action, dtype=np.float64))
                agent.memory.push(state, action, reward, next_state, float(done))
                info = agent.update()
                if info:
                    for key in step_diag:
                        step_diag[key].append(info[key])
                terminal_reward = float(reward) if done else terminal_reward
                state = next_state
            physical = metrics(env.surrogate_profile(), actions, task)
            histories["reward"].append(terminal_reward)
            for key in ("left_margin", "right_margin", "bilateral_margin",
                        "original_reward", "kappa", "rho_A",
                        "local_peak", "global_peak"):
                histories[key].append(physical[key])
            for key in step_diag:
                histories[key].append(float(np.mean(step_diag[key])))
            histories["ou_mean"].append(float(np.mean(ou_values)))
            histories["ou_std"].append(float(np.std(ou_values)))

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
            k: np.asarray(histories[k], dtype=np.float64)
            for k in ("critic_loss", "actor_loss", "q_mean", "q_target_mean")
        })
        np.savez(wdir / "ou_history.npz",
                 mean=np.asarray(histories["ou_mean"]),
                 std=np.asarray(histories["ou_std"]))
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
            "domain": domain, "task": task, "algorithm": "DDPG",
            "method": method, "seed": seed, "ddpg": CFG["ddpg"],
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
            "domain": domain, "task": task, "algorithm": "DDPG",
            "method": method, "seed": seed, "machine": machine,
            "source_task": task if method == "DDPG-TRL" else None,
            "source_seed": seed if method == "DDPG-TRL" else None,
            "scratch_role": init_audit["scratch_role"],
            "state_dim": 60, "critic_input_dim": 61,
            "source_target_pairing": "identity" if method == "DDPG-TRL" else None,
            "selection_episode": int(best["episode"]),
            "selected_action_sha256": best["action_sha256"],
            "compact_best_policy_sha256": compact_hash,
            "hyperparameters": CFG["ddpg"], **provenance,
        }
        atomic_write_json(wdir / "run_manifest.json", manifest)
        atomic_write_json(wdir / "source_target_pairing.json", {
            "task": task, "seed": seed,
            "source_task": task if method == "DDPG-TRL" else None,
            "source_seed": seed if method == "DDPG-TRL" else None,
            "pairing": "identity" if method == "DDPG-TRL" else "not_applicable",
        })
        wall = time.perf_counter() - t0
        result = {
            "run_key": manifest["run_key"], "domain": domain, "task": task,
            "algorithm": "DDPG", "method": method, "seed": seed,
            "machine": machine, "episodes": EPISODES,
            "started": started, "finished": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "wall_seconds": wall, "gpu_peak_gb":
            torch.cuda.max_memory_allocated() / 1e9,
            "n_deterministic": len(deterministic),
            "updates": int(agent.n_updates), "metrics": run_met,
            "selected_episode": int(best["episode"]),
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
