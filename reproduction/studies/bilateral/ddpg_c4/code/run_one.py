"""Execute ONE A4_C4_NATIVE_OPT TRL run, crash-safe and resumable.

Resume checkpoints exist only to survive an interruption and are DELETED by the
worker once the run completes and validates, so no completed run keeps a
replay-buffer-sized file. Only the compact best/final policies are retained.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import random
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))

from common_c4 import (CFG, D, ROOT, action_to_thickness,  # noqa: E402
                       atomic_write_json, atomic_write_text, code_hashes,
                       det_schedule, make_numerator, physical_metrics, run_dir,
                       sha256, source_path, verify_assets)
from ddpg_agent import DDPGAgent  # noqa: E402  (byte-identical to the authority)
from env60 import Environment  # noqa: E402
from recipe_c4 import apply_a4c4_native, optimizer_audit  # noqa: E402


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        torch.set_float32_matmul_precision("highest")
    except Exception:
        pass


def capture_rng(rng):
    st = {"python": random.getstate(), "numpy": np.random.get_state(),
          "torch_cpu": torch.get_rng_state(), "ou_rng": rng.bit_generator.state}
    if torch.cuda.is_available():
        st["torch_cuda"] = torch.cuda.get_rng_state_all()
    return st


def restore_rng(st, rng):
    random.setstate(st["python"])
    np.random.set_state(st["numpy"])
    torch.set_rng_state(st["torch_cpu"].cpu() if torch.is_tensor(st["torch_cpu"])
                        else st["torch_cpu"])
    rng.bit_generator.state = st["ou_rng"]
    if torch.cuda.is_available() and st.get("torch_cuda") is not None:
        torch.cuda.set_rng_state_all([s.cpu() if torch.is_tensor(s) else s
                                      for s in st["torch_cuda"]])


def dump_replay(mem):
    if len(mem.buf) == 0:
        return {"n": 0}
    s, a, r, s2, d = zip(*mem.buf)
    return {"n": len(mem.buf), "s": np.stack(s), "a": np.stack(a),
            "r": np.asarray(r, np.float32), "s2": np.stack(s2),
            "d": np.asarray(d, np.float32)}


def load_replay(mem, blob):
    mem.buf.clear()
    if not blob or blob.get("n", 0) == 0:
        return
    for i in range(int(blob["n"])):
        mem.buf.append((blob["s"][i], blob["a"][i], float(blob["r"][i]),
                        blob["s2"][i], float(blob["d"][i])))


def build_env(spec, device):
    dom = CFG["domains"][spec["domain"]]
    return Environment(
        case=CFG["env"]["case"], model_row=CFG["env"]["model_row"],
        model_col=CFG["env"]["model_col"],
        surrogate_path=str(ROOT / dom["surrogate"]),
        y_scaler_path=str(ROOT / dom["scaler"]),
        reward_function=make_numerator(spec["objective"], spec["task"]),
        device=device, leakyrelu_para=CFG["env"]["leakyrelu_para"],
        domain=spec["domain"], row_factor=float(dom["row_factor"]),
        col_factor=float(dom["col_factor"]))


def deterministic_eval(env, agent, spec):
    s = env.reset()
    for _ in range(env.hw):
        s, _, _ = env.replay_step(agent.act(s, noise=False))
    acts = env.actions_row_major()
    _, prof = env.eval_surr()
    prof = np.asarray(prof, dtype=np.float64).reshape(-1)
    m = physical_metrics(prof, acts, spec["task"])
    m["actions"] = acts.tolist()
    m["reward_objective"] = (m["reward_bilateral"] if spec["objective"] == "bilateral"
                             else m["reward_original"])
    return m


def execute(spec, out, device, checkpoint_every=25, max_episodes=None, crash_at=None):
    episodes = int(max_episodes or D["episodes"])
    out.mkdir(parents=True, exist_ok=True)
    ck_path = out / "latest_resume.pt"

    set_seed(int(spec["seed"]))
    ou_rng = np.random.default_rng(int(spec["seed"]) + 10_000)
    agent = DDPGAgent(
        state_dim=int(CFG["env"]["state_dim"]), device=device,
        actor_lr=float(D["actor_lr"]), critic_lr=float(D["critic_lr"]),
        weight_decay=float(D["actor_weight_decay"]), gamma=float(D["gamma"]),
        tau=float(D["tau"]), batch_size=int(D["batch_size"]),
        memory_capacity=int(D["replay_capacity"]),
        ou=(float(D["ou"]["mu"]), float(D["ou"]["theta"]), float(D["ou"]["sigma"])),
        noise_rng=ou_rng)
    env = build_env(spec, device)

    # ---- transfer (always TRL in this experiment) --------------------------
    src = source_path(spec["source_bank"], spec["task"], spec["seed"])
    if not src.is_file():
        raise FileNotFoundError(f"source checkpoint missing: {src}")
    blob = torch.load(src, map_location=device, weights_only=False)
    sd = blob.get("state_dicts", blob)
    if "actor" not in sd or "critic" not in sd:
        raise ValueError(f"source {src} has no actor/critic state dicts")
    audit = apply_a4c4_native(agent, sd)
    audit.update({"source_checkpoint": str(src), "source_sha256": sha256(src),
                  "source_bank": spec["source_bank"],
                  "pairing": "identity: target seed s <- same-task source seed s",
                  "source_task": spec["task"], "source_seed": int(spec["seed"])})

    # Evaluated BEFORE any training: this is the transferred policy's zero-shot
    # performance on the target domain, which is one of the reported metrics.
    zero_shot = deterministic_eval(env, agent, spec)
    zero_shot["episode"] = 0

    start_ep, rewards, det_rows, best = 0, [], [], None
    losses = {"critic": [], "actor": [], "q": []}
    ou_trace = []

    if ck_path.is_file():
        try:
            ck = torch.load(ck_path, map_location=device, weights_only=False)
            agent.load_state_dicts(ck["nets"])
            agent.actor_opt.load_state_dict(ck["actor_opt"])
            agent.critic_opt.load_state_dict(ck["critic_opt"])
            load_replay(agent.memory, ck["replay"])
            agent.noise.state = np.asarray(ck["ou_state"], dtype=np.float64)
            agent.n_updates = int(ck["n_updates"])
            restore_rng(ck["rng"], ou_rng)
            start_ep = int(ck["episode"])
            rewards, det_rows, best = list(ck["rewards"]), list(ck["det_rows"]), ck["best"]
            losses = {k: list(v) for k, v in ck["losses"].items()}
            ou_trace = list(ck["ou_trace"])
            print(f"[resume] episode {start_ep}/{episodes}", flush=True)
        except Exception as e:                                    # noqa: BLE001
            raise RuntimeError(f"latest_resume.pt unreadable ({e}); delete the run "
                               "directory to restart it cleanly") from e

    sched = set(det_schedule())
    fill = int(D["memory_fill_designs"])
    t0 = time.time()

    def save_ckpt(ep):
        blob = {"episode": ep, "nets": agent.state_dicts(),
                "actor_opt": agent.actor_opt.state_dict(),
                "critic_opt": agent.critic_opt.state_dict(),
                "replay": dump_replay(agent.memory),
                "ou_state": agent.noise.state.copy(),
                "n_updates": agent.n_updates, "rng": capture_rng(ou_rng),
                "rewards": rewards, "det_rows": det_rows, "best": best,
                "losses": losses, "ou_trace": ou_trace, "spec": spec}
        tmp = out / f".resume.{os.getpid()}.tmp"
        with open(tmp, "wb") as f:
            torch.save(blob, f)
            f.flush()
            os.fsync(f.fileno())
        for attempt in range(20):
            try:
                os.replace(tmp, ck_path)
                break
            except PermissionError:
                if attempt == 19:
                    raise
                time.sleep(0.25)

    if start_ep == 0 and 0 in sched:
        m = dict(zero_shot)
        m["episode"] = 0
        det_rows.append(m)
        best = dict(m)

    for ep in range(start_ep, episodes):
        if crash_at is not None and ep == int(crash_at):
            save_ckpt(ep)
            os._exit(9)
        s = env.reset()
        agent.noise.reset()
        ep_r = 0.0
        for _ in range(env.hw):
            a = agent.act(s, noise=True)
            s2, r, done = env.replay_step(a)
            agent.memory.push(s, a, r, s2, float(done))
            s = s2
            ep_r = r if done else ep_r
            if ep >= fill:
                for _ in range(int(D["updates_per_transition"])):
                    info = agent.update()
                    if info and len(losses["critic"]) < 200000:
                        losses["critic"].append(info["critic_loss"])
                        losses["actor"].append(info["actor_loss"])
                        losses["q"].append(info["q_mean"])
        rewards.append(float(ep_r))
        ou_trace.append(float(agent.noise.state[0]))

        if (ep + 1) in sched:
            m = deterministic_eval(env, agent, spec)
            m["episode"] = ep + 1
            det_rows.append(m)
            if best is None or m["reward_objective"] > best["reward_objective"]:
                best = dict(m)
        if (ep + 1) % checkpoint_every == 0 or (ep + 1) == episodes:
            save_ckpt(ep + 1)

    # ------------------------------------------------------------- artifacts
    rh = np.asarray(rewards, dtype=np.float64)
    det_ep = np.asarray([d["episode"] for d in det_rows], dtype=np.int64)
    det_r = np.asarray([d["reward_objective"] for d in det_rows], dtype=np.float64)
    dom = CFG["domains"][spec["domain"]]
    final = det_rows[-1]

    np.save(out / "reward_history.npy", rh)
    atomic_write_text(out / "reward_history.txt",
                      "\n".join(f"{v:.12g}" for v in rh) + "\n")
    np.savez_compressed(out / "loss_history.npz",
                        critic=np.asarray(losses["critic"], np.float32),
                        actor=np.asarray(losses["actor"], np.float32),
                        q_mean=np.asarray(losses["q"], np.float32))
    np.savez_compressed(out / "ou_history.npz", ou=np.asarray(ou_trace, np.float64))

    cols = ["episode", "reward_objective", "reward_original", "reward_bilateral",
            "kappa", "rho_A", "left_margin", "right_margin", "bilateral_margin",
            "bilateral_margin_normalized", "global_margin", "target_is_global_max"]
    with (out / "deterministic_history.csv").open("w", encoding="utf-8", newline="") as f:
        f.write(",".join(cols) + "\n")
        for d in det_rows:
            f.write(",".join(str(d[c]) for c in cols) + "\n")
    atomic_write_json(out / "deterministic_history.json", det_rows)

    np.savez_compressed(
        out / "physical_metric_history.npz", episode=det_ep,
        u=np.asarray([d["u"] for d in det_rows], np.float64),
        actions=np.asarray([d["actions"] for d in det_rows], np.float64),
        kappa=np.asarray([d["kappa"] for d in det_rows], np.float64),
        rho_A=np.asarray([d["rho_A"] for d in det_rows], np.float64),
        bilateral_margin=np.asarray([d["bilateral_margin"] for d in det_rows], np.float64),
        reward_objective=det_r)

    def design(m, name):
        a = np.asarray(m["actions"], dtype=np.float64)
        return {"name": name, "episode": int(m["episode"]),
                "actions": a.tolist(),
                "actions_copy_ready": " ".join(f"{v:.12g}" for v in a),
                "thickness_mm": action_to_thickness(a, dom["t_min"], dom["t_max"]).tolist(),
                **{k: v for k, v in m.items() if k != "actions"}}

    atomic_write_json(out / "zero_shot_design.json", design(zero_shot, "zero_shot"))
    atomic_write_json(out / "best_design.json", design(best, "best_deterministic"))
    atomic_write_json(out / "final_design.json", design(final, "final_deterministic"))
    np.save(out / "best_action.npy", np.asarray(best["actions"], dtype=np.float64))
    atomic_write_text(out / "best_action.txt",
                      " ".join(f"{v:.12g}" for v in best["actions"]) + "\n")

    meta = {"task": spec["task"], "seed": int(spec["seed"]),
            "domain": spec["domain"], "objective": spec["objective"],
            "recipe_id": audit["recipe_id"], "selected_episode": int(best["episode"]),
            "state_dim": int(CFG["env"]["state_dim"]),
            "critic_input_dim": int(CFG["env"]["critic_input_dim"])}
    torch.save({**meta, "state_dicts": agent.state_dicts()},
               out / "compact_final_policy.pt")
    torch.save({**meta, "which": "best_deterministic",
                "state_dicts": agent.state_dicts()}, out / "compact_best_policy.pt")

    early = det_r[det_ep <= 300]
    atomic_write_json(out / "transfer_tensor_audit.json", audit)
    atomic_write_json(out / "optimizer_audit.json", optimizer_audit(agent))
    atomic_write_json(out / "source_target_pairing.json", {
        "target_task": spec["task"], "target_seed": int(spec["seed"]),
        "source_task": spec["task"], "source_seed": int(spec["seed"]),
        "rule": "identity pairing, fixed before launch",
        "source_bank": spec["source_bank"],
        "source_path": str(src), "source_sha256": sha256(src),
        "selected_by_target_performance": False})
    atomic_write_json(out / "provenance.json", {
        "hostname": platform.node(), "os": platform.platform(),
        "python": sys.version.split()[0], "python_executable": sys.executable,
        "torch": torch.__version__, "cuda_build": torch.version.cuda,
        "gpu": (torch.cuda.get_device_name(0) if torch.cuda.is_available() else None),
        "cuda_used": bool(torch.cuda.is_available()) and str(device).startswith("cuda"),
        "numpy": np.__version__, "code_sha256": code_hashes(),
        "asset_sha256": verify_assets()})
    atomic_write_json(out / "config_resolved.yaml", {
        "spec": spec, "ddpg": D, "transfer": CFG["transfer"],
        "env": CFG["env"], "domain": dom, "selection": CFG["selection"]})
    atomic_write_json(out / "run_manifest.json", {
        "spec": spec, "episodes": episodes, "n_rewards": int(rh.size),
        "n_deterministic_evals": len(det_rows), "n_updates": agent.n_updates,
        "selected_episode": int(best["episode"]),
        "metrics": {
            "zero_shot_deterministic_reward": float(zero_shot["reward_objective"]),
            "early300_deterministic_auc": float(early.mean()) if early.size else None,
            "full_deterministic_auc1500": float(det_r.mean()),
            "final100_training_reward": float(rh[-100:].mean()),
            "best_deterministic_surrogate_reward": float(det_r.max()),
            "final_deterministic_reward": float(det_r[-1]),
            "best_kappa": float(best["kappa"]),
            "best_rho_A": float(best["rho_A"]),
            "best_bilateral_margin": float(best["bilateral_margin"]),
        },
        "wall_seconds": time.time() - t0})
    atomic_write_json(out / "completion.json", {
        "complete": True, "run_key": spec["run_key"],
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "n_rewards": int(rh.size), "n_deterministic_evals": len(det_rows),
        "n_transferred_tensors": audit["n_transferred_tensors"],
        "reward_history_sha256": sha256(out / "reward_history.npy"),
        "best_action_sha256": sha256(out / "best_action.npy"),
        "hostname": platform.node()})
    return {"run_key": spec["run_key"],
            "best_det": float(det_r.max()), "final100": float(rh[-100:].mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--episodes", type=int, default=None)
    ap.add_argument("--checkpoint-every", type=int, default=25)
    ap.add_argument("--crash-at", type=int, default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    spec = json.loads(Path(a.spec).read_text(encoding="utf-8"))
    out = Path(a.out) if a.out else run_dir(spec)
    print(json.dumps(execute(spec, out, a.device, a.checkpoint_every,
                             a.episodes, a.crash_at)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
