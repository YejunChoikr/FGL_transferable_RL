"""Reward-coefficient runner: one SAC run per (stage, task, C1, C2, seed).

SAC protocol is inherited from the calibration H1 lock and never re-searched.
Source-domain scratch SAC only (no transfer / warm start), so the coefficient
effect is isolated from policy transfer.

Every episode's terminal design is written to physical_metric_history.npz with no
episode skipped and no smoothing. Deterministic rollouts happen at episode 20 and
every 25th episode; their actions never enter the replay buffer and never produce
a gradient update.

Usage:
  python code/run_reward.py --machine local --stage stage1_C1
  python code/run_reward.py --machine local --cells "0.3:0.0,0.3:0.25" --stage stage2_C2
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch

warnings.filterwarnings("ignore")

CODE = Path(__file__).resolve().parent
ROOT = CODE.parent
sys.path.insert(0, str(CODE))

from common import (  # noqa: E402
    C1_GRID, CFG, CRITIC_DIM, SEEDS, STATE_DIM, agent_cfg, asset_hashes,
    atomic_write_json, bundle_hash, cell_tag, code_hashes, det_schedule, env_metadata,
    episodes as EPISODES_FN, fill_episodes, hyperparameters, lock, run_dir, run_key,
    run_metrics, sha256, weights_dir,
)
from env_cc import STATE_ENCODING, make_env, set_seed  # noqa: E402
from reward_cc import HISTORY_SCALARS, TASK_INDICES, physical_metrics  # noqa: E402
from sac_agent import SACAgent  # noqa: E402

EPISODES = EPISODES_FN()
FILL = fill_episodes()
DET_EPS = det_schedule(EPISODES)
LATEST_EVERY = int(CFG["train"]["latest_checkpoint_every"])


def assert_contract(env, agent):
    assert tuple(env.observation_space.shape) == (STATE_DIM,)
    s0 = env.reset()
    assert s0.shape == (STATE_DIM,) and s0.dtype == np.float32
    assert np.all(np.isfinite(s0)) and np.all(s0 == 0.0)
    assert agent.actor.fc1.in_features == STATE_DIM
    assert agent.q1.fc1.in_features == CRITIC_DIM and agent.q2.fc1.in_features == CRITIC_DIM
    done, n = False, 0
    while not done:
        s, r, done = env.replay_step(np.array([0.1], dtype=np.float64))
        n += 1
        assert s.shape == (STATE_DIM,) and s.dtype == np.float32 and np.all(np.isfinite(s))
    assert n == 30
    assert np.array_equal(s, env.last_surrogate_input)
    env.reset()
    return {"episode_length": n, "terminal_obs_equals_surrogate_input": True}


def det_rollout(env, agent, task, c1, c2):
    s = env.reset()
    done = False
    r_last = float("nan")
    while not done:
        a = agent.deterministic_action(s)
        s, r, done = env.replay_step(np.array([a[0]], dtype=np.float64))
        if done:
            r_last = float(r)
    u = env.surrogate_profile()
    m = physical_metrics(u, env.actions_row_major(), task, c1, c2)
    m["reward_env"] = r_last
    return m


def q_disagreement(agent, bs):
    if len(agent.replay) < bs:
        return float("nan")
    s, a, ns, r, d = agent.replay.sample(bs, agent.device)
    with torch.no_grad():
        return float((agent.q1(s, a) - agent.q2(s, a)).abs().mean().item())


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state().tolist(),
            "cuda": torch.cuda.get_rng_state().tolist() if torch.cuda.is_available() else None}


def _tup(x):
    return tuple(_tup(v) for v in x) if isinstance(x, list) else x


def save_latest(path, agent, ep, hist, rng):
    mem = list(agent.replay.memory)
    torch.save({"episode": ep,
                "actor": agent.actor.state_dict(), "q1": agent.q1.state_dict(),
                "q2": agent.q2.state_dict(), "q1_target": agent.q1_target.state_dict(),
                "q2_target": agent.q2_target.state_dict(),
                "log_alpha": agent.log_alpha.detach().cpu(),
                "actor_opt": agent.actor_opt.state_dict(),
                "q1_opt": agent.q1_opt.state_dict(), "q2_opt": agent.q2_opt.state_dict(),
                "alpha_opt": agent.alpha_opt.state_dict(),
                "replay_s": np.array([m[0] for m in mem], dtype=np.float32),
                "replay_a": np.array([m[1] for m in mem], dtype=np.float32),
                "replay_ns": np.array([m[2] for m in mem], dtype=np.float32),
                "replay_r": np.array([m[3] for m in mem], dtype=np.float64),
                "replay_d": np.array([m[4] for m in mem], dtype=np.float32),
                "hist": hist, "rng": rng}, Path(str(path) + ".tmp"))
    Path(str(path) + ".tmp").replace(path)


def load_latest(path, agent):
    b = torch.load(path, map_location=agent.device, weights_only=False)
    agent.actor.load_state_dict(b["actor"])
    agent.q1.load_state_dict(b["q1"]); agent.q2.load_state_dict(b["q2"])
    agent.q1_target.load_state_dict(b["q1_target"])
    agent.q2_target.load_state_dict(b["q2_target"])
    with torch.no_grad():
        agent.log_alpha.copy_(b["log_alpha"].to(agent.device))
    agent.actor_opt.load_state_dict(b["actor_opt"])
    agent.q1_opt.load_state_dict(b["q1_opt"]); agent.q2_opt.load_state_dict(b["q2_opt"])
    agent.alpha_opt.load_state_dict(b["alpha_opt"])
    agent.replay.memory.clear()
    for i in range(len(b["replay_r"])):
        agent.replay.store(b["replay_s"][i], b["replay_a"][i], b["replay_ns"][i],
                           float(b["replay_r"][i]), float(b["replay_d"][i]))
    rng = b["rng"]
    random.setstate(_tup(rng["python"]))
    np.random.set_state(_tup(rng["numpy"]))
    torch.set_rng_state(torch.ByteTensor(rng["torch"]))
    if torch.cuda.is_available() and rng.get("cuda") is not None:
        torch.cuda.set_rng_state(torch.ByteTensor(rng["cuda"]))
    return int(b["episode"]), b["hist"]


def save_full_checkpoint(agent, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"actor": agent.actor.state_dict(), "q1": agent.q1.state_dict(),
                "q2": agent.q2.state_dict(), "q1_target": agent.q1_target.state_dict(),
                "q2_target": agent.q2_target.state_dict(),
                "log_alpha": agent.log_alpha.detach().cpu(),
                "actor_opt": agent.actor_opt.state_dict(),
                "q1_opt": agent.q1_opt.state_dict(), "q2_opt": agent.q2_opt.state_dict(),
                "alpha_opt": agent.alpha_opt.state_dict()}, Path(str(path) + ".tmp"))
    Path(str(path) + ".tmp").replace(path)


# ----------------------------------------------------------------------
def train_one(stage, task, c1, c2, seed, machine_id):
    key = run_key(stage, task, c1, c2, seed)
    wdir = run_dir(ROOT, stage, task, c1, c2, seed)
    ckdir = weights_dir(ROOT, stage, task, c1, c2, seed)
    wdir.mkdir(parents=True, exist_ok=True)
    ckdir.mkdir(parents=True, exist_ok=True)
    rj, latest = wdir / "run.json", ckdir / "latest.pt"

    if rj.exists():
        try:
            prev = json.loads(rj.read_text(encoding="utf-8"))
            rh = wdir / "reward_history.npy"
            if prev.get("complete") and rh.exists() and len(np.load(rh)) == EPISODES:
                print(f"[{key}] RESUME-SKIP", flush=True)
                return prev
        except Exception:
            pass
    if not torch.cuda.is_available():
        raise SystemExit(f"[{key}] CUDA required (no CPU fallback)")

    acfg = agent_cfg()
    bs = acfg["train"]["batch_size"]
    set_seed(seed)
    torch.cuda.reset_peak_memory_stats()
    t0, started = time.perf_counter(), time.strftime("%Y-%m-%dT%H:%M:%S")

    env = make_env(task, c1, c2, CFG, ROOT, "cuda")
    agent = SACAgent(env.observation_space.shape[0], "cuda", acfg)
    contract = assert_contract(env, agent)

    hist = {"reward": [], "alpha": [], "entropy": [], "qdis": [],
            "q1_loss": [], "q2_loss": [], "actor_loss": [], "alpha_loss": []}
    phys = {k: [] for k in HISTORY_SCALARS}
    phys_u, phys_act = [], []
    det_records = []
    best_det = {"reward_env": -float("inf")}
    start_ep, env_calls, grad_updates = 0, 0, 0

    if latest.exists():
        try:
            start_ep, blob = load_latest(latest, agent)
            hist = blob["hist"]
            pp = wdir / "phys_partial.npz"
            if pp.exists():
                z = np.load(pp, allow_pickle=True)
                phys = {k: list(z[k]) for k in HISTORY_SCALARS}
                phys_u, phys_act = list(z["u"]), list(z["action"])
            dr = wdir / "det_partial.json"
            det_records = json.loads(dr.read_text(encoding="utf-8")) if dr.exists() else []
            if det_records:
                best_det = det_records[int(np.argmax([d["reward_env"] for d in det_records]))]
            print(f"[{key}] RESUMED at ep {start_ep}", flush=True)
        except Exception as e:
            print(f"[{key}] latest unusable ({e}); restart", flush=True)
            start_ep = 0

    def record_terminal():
        m = physical_metrics(env.surrogate_profile(), env.actions_row_major(), task, c1, c2)
        for k in HISTORY_SCALARS:
            phys[k].append(float(m[k]))
        phys_u.append(m["u"])
        phys_act.append(m["action"])

    if start_ep == 0:
        for _ in range(FILL):
            s = env.reset(); done = False
            while not done:
                a = agent.select_action(s)
                ns, r, done = env.replay_step(np.array([a[0]], dtype=np.float64))
                agent.replay.store(s, a, ns, r, done)
                env_calls += 1
                s = ns

    for ep in range(start_ep, EPISODES):
        s = env.reset(); done = False
        eal, een, eq1, eq2, eac, ealo = [], [], [], [], [], []
        ep_reward = 0.0
        while not done:
            a = agent.select_action(s)
            ns, r, done = env.replay_step(np.array([a[0]], dtype=np.float64))
            env_calls += 1
            agent.replay.store(s, a, ns, r, done)
            m = agent.update(bs)
            if m is not None:
                grad_updates += 1
                eal.append(m["alpha"]); een.append(m["entropy"])
                eq1.append(m["q1_loss"]); eq2.append(m["q2_loss"])
                eac.append(m["actor_loss"]); ealo.append(m["alpha_loss"])
            if done:
                ep_reward = float(r)
                record_terminal()
            s = ns
        hist["reward"].append(ep_reward)
        hist["alpha"].append(float(np.mean(eal)) if eal else float(agent.alpha.item()))
        hist["entropy"].append(float(np.mean(een)) if een else float("nan"))
        hist["q1_loss"].append(float(np.mean(eq1)) if eq1 else float("nan"))
        hist["q2_loss"].append(float(np.mean(eq2)) if eq2 else float("nan"))
        hist["actor_loss"].append(float(np.mean(eac)) if eac else float("nan"))
        hist["alpha_loss"].append(float(np.mean(ealo)) if ealo else float("nan"))
        hist["qdis"].append(q_disagreement(agent, bs))

        if (ep + 1) in DET_EPS:
            d = det_rollout(env, agent, task, c1, c2)
            d["episode"] = ep + 1
            det_records.append(d)
            if d["reward_env"] > best_det["reward_env"]:
                best_det = dict(d)
                torch.save(agent.actor.state_dict(), ckdir / "best_det_actor.pth")
                save_full_checkpoint(agent, ckdir / "best_det_full.pt")
            (wdir / "det_partial.json").write_text(json.dumps(det_records), encoding="utf-8")

        if (ep + 1) % LATEST_EVERY == 0 and ep + 1 < EPISODES:
            save_latest(latest, agent, ep + 1, hist, rng_state())
            np.savez_compressed(wdir / "phys_partial.npz",
                                **{k: np.array(v, dtype=np.float32) for k, v in phys.items()},
                                u=np.array(phys_u, dtype=np.float32),
                                action=np.array(phys_act, dtype=np.float32))
        if (ep + 1) % 500 == 0:
            print(f"[{key}] ep {ep+1}/{EPISODES} r={ep_reward:.5f} "
                  f"kappa={phys['kappa'][-1]:.4f} rho={phys['rho_A'][-1]:.4f}", flush=True)

    wall = time.perf_counter() - t0
    peak = torch.cuda.max_memory_allocated() / 1e9
    final_det = det_rollout(env, agent, task, c1, c2)
    final_det["episode"] = EPISODES
    torch.save(agent.actor.state_dict(), ckdir / "final_actor.pth")
    save_full_checkpoint(agent, ckdir / "final_full.pt")

    rh = np.array(hist["reward"], dtype=np.float64)
    np.save(wdir / "reward_history.npy", rh)
    np.savetxt(wdir / "reward_history.txt", rh, fmt="%.10f")
    for n, k in (("alpha_history", "alpha"), ("entropy_history", "entropy"),
                 ("q_disagreement", "qdis")):
        np.save(wdir / f"{n}.npy", np.array(hist[k], dtype=np.float64))
    np.savez(wdir / "losses.npz", **{k: np.array(hist[k], dtype=np.float64)
                                     for k in ("q1_loss", "q2_loss", "actor_loss", "alpha_loss")})
    # per-episode terminal-design physics: every episode, no smoothing
    np.savez_compressed(wdir / "physical_metric_history.npz",
                        **{k: np.array(v, dtype=np.float32) for k, v in phys.items()},
                        u=np.array(phys_u, dtype=np.float32),
                        action=np.array(phys_act, dtype=np.float32),
                        episode=np.arange(1, len(phys["kappa"]) + 1, dtype=np.int32))
    np.savez(wdir / "deterministic_eval.npz",
             episodes=np.array([d["episode"] for d in det_records], dtype=np.int64),
             reward=np.array([d["reward_env"] for d in det_records], dtype=np.float64),
             kappa=np.array([d["kappa"] for d in det_records], dtype=np.float64),
             rho_A=np.array([d["rho_A"] for d in det_records], dtype=np.float64),
             bilateral_raw=np.array([d["bilateral_raw"] for d in det_records], dtype=np.float64),
             u=np.array([d["u"] for d in det_records], dtype=np.float64),
             action=np.array([d["action"] for d in det_records], dtype=np.float64))
    atomic_write_json(wdir / "best_det_design.json", best_det)
    atomic_write_json(wdir / "final_det_design.json", final_det)

    met = run_metrics(rh)
    finite = bool(np.all(np.isfinite(rh))) and len(rh) == EPISODES
    loss_finite = all(bool(np.all(np.isfinite(np.array(hist[k], dtype=np.float64))))
                      for k in ("q1_loss", "q2_loss", "actor_loss", "alpha_loss"))
    phys_finite = all(np.all(np.isfinite(np.array(v, dtype=np.float64))) for v in phys.values())

    res = {"run_key": key, "stage": stage, "task": task, "C1": float(c1), "C2": float(c2),
           "cell": cell_tag(c1, c2), "seed": seed, "machine_id": machine_id,
           "state_encoding": STATE_ENCODING, "state_dim": STATE_DIM,
           "critic_input_dim": CRITIC_DIM,
           "runtime_contract": contract, "hyperparameters": hyperparameters(),
           "calibration_lock": {"selected_config": lock()["selected_config"],
                                "lock_sha256": lock()["lock_sha256"]},
           "reward_form": {"numerator": "u_i - C1*(u_(i-1)+u_(i+1))",
                           "W": "1 if C2==0 else 0.5 + C2*rho_A",
                           "rho_A": "mean((action+1)/2)"},
           "reward_indices_0based": TASK_INDICES[task],
           "asset_sha256": asset_hashes(ROOT), "code_sha256": code_hashes(ROOT),
           "bundle_sha256": bundle_hash(ROOT), "env": env_metadata(),
           "started": started, "finished": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "wall_s": wall, "gpu_peak_gb": peak,
           "environment_calls": env_calls, "gradient_updates": grad_updates,
           "n_deterministic_evals": len(det_records),
           "deterministic_eval_episodes": DET_EPS[:3] + ["..."] + DET_EPS[-1:],
           "metrics": met,
           "best_det": {k: best_det.get(k) for k in
                        ("episode", "reward_env", "kappa", "rho_A", "bilateral_raw",
                         "bilateral_normalized", "global_margin", "target_is_global_peak",
                         "own_objective_reward",
                         "original_locked_reward_at_C1_0.5_C2_1.0", "W")},
           "final_det": {k: final_det.get(k) for k in
                         ("episode", "reward_env", "kappa", "rho_A", "bilateral_raw",
                          "bilateral_normalized", "global_margin", "target_is_global_peak",
                          "own_objective_reward",
                          "original_locked_reward_at_C1_0.5_C2_1.0", "W")},
           "reward_history_finite": finite, "losses_finite": loss_finite,
           "physical_history_finite": bool(phys_finite),
           "complete": bool(finite and loss_finite and phys_finite)}
    atomic_write_json(rj, res)
    for f in (latest, wdir / "phys_partial.npz", wdir / "det_partial.json"):
        if f.exists():
            f.unlink()
    print(f"[{key}] DONE wall={wall:.0f}s final100={met['final100']:.5f} "
          f"best_det_kappa={best_det['kappa']:.5f} rho={best_det['rho_A']:.5f} "
          f"bilat={best_det['bilateral_raw']:.5f}", flush=True)
    del agent, env
    torch.cuda.empty_cache()
    return res


def parse_cells(text):
    out = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        a, b = part.split(":")
        out.append((float(a), float(b)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--machine", required=True, choices=["local", "server", "workstation"])
    ap.add_argument("--stage", required=True,
                    choices=["stage1_C1", "stage2_C2", "joint_audit"])
    ap.add_argument("--cells", default=None, help="C1:C2 pairs, comma separated")
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    args = ap.parse_args()

    mid = args.machine
    task = CFG["machines"][mid]["task"]
    seeds = args.seeds if args.seeds else SEEDS
    if args.cells:
        cells = parse_cells(args.cells)
    elif args.stage == "stage1_C1":
        cells = [(c1, 0.0) for c1 in C1_GRID]
    else:
        raise SystemExit("--cells is required for stage2_C2 / joint_audit")

    print(f"[runner] machine={mid} task={task} stage={args.stage} "
          f"cells={cells} seeds={seeds} lock={lock()['selected_config']} "
          f"gpu={torch.cuda.get_device_name(0)}", flush=True)
    out = []
    for (c1, c2) in cells:
        for sd in seeds:
            out.append(train_one(args.stage, task, c1, c2, sd, mid))
            atomic_write_json(ROOT / "results" / f"shard_{mid}_{args.stage}_partial.json", out)
    atomic_write_json(ROOT / "results" / f"shard_{mid}_{args.stage}.json", out)
    print(f"[runner] STAGE DONE {args.stage} {len(out)} runs", flush=True)


if __name__ == "__main__":
    main()
