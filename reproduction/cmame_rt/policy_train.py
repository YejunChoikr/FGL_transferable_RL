"""Policy training runner: 1,500 episodes, 45,000 transitions, 44,400 updates.

Experimental settings are defined in spec/protocol.json.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import socket
import time
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch

from . import ddpg as DDPG
from . import evaluation as EV
from . import policy_init as PI
from . import sac as SAC
from .determinism import apply_runtime_flags, runtime_state
from .env import SurrogateEnv, physical_thickness_from_actions, relabel_terminal_rewards
from .paths import INIT_BANK, PILOT, ROOT
from .protocol import PROTOCOL_HASH, load_protocol
from .replay import GPUReplay
from .rng import subseed, torch_gen

_PROTO = load_protocol()
_POLICY = _PROTO["policy"]

EPISODES_TOTAL: int = int(_POLICY["episodes_total"])
STEPS_PER_EPISODE: int = int(_POLICY["steps_per_episode"])
PREFILL_EPISODES: int = int(_POLICY["prefill_episodes_in_total"])
TRANSITIONS_EXPECTED: int = int(_POLICY["training_transition_count"])
UPDATES_EXPECTED: int = int(_POLICY["gradient_update_count"])
SHARED_GOALS: tuple = tuple(int(g) for g in _POLICY["shared"]["goals"])
GOALS_PER_BLOCK: int = len(SHARED_GOALS)
RESUME_INTERVAL_S: float = 600.0

#: Arms that copy hidden layers from a source policy checkpoint.
TRANSFER_ARMS: dict = {
    "ST_PA": {"copy_actor": True, "copy_critics": False},
    "ST_PAC": {"copy_actor": True, "copy_critics": True},
    "TRL": {"copy_actor": True, "copy_critics": True},
}


# ------------------------------------------------------------------- helpers
def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _save_atomic(obj, path: Path, saver=torch.save) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    saver(obj, tmp)
    tmp.replace(path)


def _write_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=_json_default)
        fh.write("\n")
    tmp.replace(path)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError("not JSON serialisable: %r" % type(o))


def goal_schedule(seed: int, n_episodes: int, goals: Sequence[int],
                  device="cpu") -> list:
    """Five-episode block shuffle: every goal appears once per block.

    Stream ``goal_schedule`` (1300). The first 20 episodes are four whole
    blocks, which is exactly the four prefill episodes per goal required by
    protocol.policy.shared.prefill_episodes_per_goal.
    """
    g = list(int(x) for x in goals)
    k = len(g)
    if n_episodes % k != 0:
        raise ValueError("episode count must be a multiple of the goal count")
    gen = torch_gen(subseed(int(seed), "goal_schedule"), "cpu")
    out = []
    for _ in range(n_episodes // k):
        perm = torch.randperm(k, generator=gen, device="cpu").tolist()
        out.extend(g[i] for i in perm)
    return out


def resolve_arm(case: dict) -> dict:
    """Split the case arm into surrogate and policy transfer decisions."""
    arm = str(case.get("arm", "P0"))
    shared = bool(case.get("shared", False))
    spec = TRANSFER_ARMS.get(arm)
    if spec is None and shared and arm in ("ST_PAC",):
        spec = TRANSFER_ARMS["ST_PAC"]
    return {
        "arm": arm,
        "shared": shared,
        "policy_transfer": spec is not None,
        "copy_actor": bool(spec["copy_actor"]) if spec else False,
        "copy_critics": bool(spec["copy_critics"]) if spec else False,
    }


def _load_source_blob(source_paths: dict, algo: str) -> dict:
    """Load ``best_actor.pt`` and ``best_critics.pt`` of the source case."""
    actor = torch.load(source_paths["actor"], map_location="cpu",
                       weights_only=True)
    critics = torch.load(source_paths["critics"], map_location="cpu",
                         weights_only=True)
    if str(algo).upper() == "SAC":
        return {"actor": actor, "q1": critics["q1"], "q2": critics["q2"]}
    return {"actor": actor, "critic": critics["critic"]}


def _load_surrogate(surrogate_path, scaler_path, device):
    """Load the frozen surrogate and its y scaler for the environment."""
    try:  # SURR provides the canonical loader once surrogate_train.py lands.
        from .surrogate_train import load_surrogate_for_env  # type: ignore
    except Exception:
        load_surrogate_for_env = None
    if load_surrogate_for_env is not None:
        return load_surrogate_for_env(surrogate_path, device)
    from .models_surrogate import CNN  # type: ignore

    model = CNN().to(device)
    sd = torch.load(surrogate_path, map_location=device, weights_only=True)
    model.load_state_dict(sd)
    with open(scaler_path, "r", encoding="utf-8") as fh:
        scaler = json.load(fh)
    return model, scaler


# ---------------------------------------------------------------- the runner
def run_policy(case: dict, attempt_dir, device, adam_backend: str = "reference",
               surrogate_path=None, scaler_path=None,
               source_paths: Optional[dict] = None, exclusive: bool = False,
               episodes: Optional[int] = None,
               surrogate=None, scaler=None,
               resume_enabled: bool = True,
               eval_episodes: Optional[Sequence[int]] = None,
               resume_interval_s: float = RESUME_INTERVAL_S) -> dict:
    """Train one policy case and write every artifact of INTERFACE section 4.

    Parameters
    ----------
    case : dict
        One row of ``spec/cases.json`` with ``kind == 'policy'``.
    attempt_dir : path
        ``runs/<case path>/attempt_<k>``; every output lands here.
    device : str or torch.device
    adam_backend : {'reference', 'foreach', 'fused'}
        Frozen in ``locks/RUNTIME_LOCK.json`` before production.
    surrogate_path, scaler_path : path
        Accepted surrogate checkpoint and its y scaler; ignored when
        ``surrogate``/``scaler`` objects are supplied directly (pilot use).
    source_paths : dict
        ``{'actor': ..., 'critics': ...}`` of the source case, for transfer arms.
    exclusive : bool
        True when the worker guarantees the host is otherwise idle; recorded in
        ``timing.json`` so reference timings stay identifiable.
    episodes : int
        Short-run override. Permitted only under ``pilot/``.
    eval_episodes : sequence of int
        Evaluation schedule override, used by the gate that compares a run with
        scheduled evaluations against one without. Permitted only under
        ``pilot/``; production always uses protocol.evaluation.episodes.
    """
    t_wall_start = time.time()
    attempt_dir = Path(attempt_dir)
    attempt_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(device)
    flags = apply_runtime_flags()

    n_episodes = EPISODES_TOTAL if episodes is None else int(episodes)
    if n_episodes != EPISODES_TOTAL or eval_episodes is not None:
        try:
            attempt_dir.resolve().relative_to(PILOT.resolve())
        except ValueError:
            raise RuntimeError(
                "a shortened episode count or a modified evaluation schedule "
                "is allowed only under pilot/; production runs use %d episodes "
                "and protocol.evaluation.episodes" % EPISODES_TOTAL)

    # ---- case ---------------------------------------------------------------
    case_id = str(case["case_id"])
    seed = int(case["seed"])
    algo = str(case["algorithm"]).upper()
    domain = str(case["domain"])
    goals = [int(g) for g in case["goals"]]
    C1, C2 = float(case["C1"]), float(case["C2"])
    objective = str(case.get("objective", "arithmetic"))
    if objective not in ("arithmetic", "bilateral"):
        raise ValueError("objective must be 'arithmetic' or 'bilateral'")
    traversal = str(case["traversal"])
    arm = resolve_arm(case)
    shared = arm["shared"]
    if shared and tuple(goals) != SHARED_GOALS:
        raise ValueError("shared case %s has goals %s" % (case_id, goals))
    if case.get("protocol_hash") and case["protocol_hash"] != PROTOCOL_HASH:
        raise RuntimeError("case protocol hash does not match spec/protocol.json")

    # ---- surrogate ----------------------------------------------------------
    if surrogate is None:
        surrogate, scaler = _load_surrogate(surrogate_path, scaler_path, device)

    # ---- environments -------------------------------------------------------
    env = SurrogateEnv(surrogate, scaler, traversal, C1, C2, goals, device,
                       batch=1, objective=objective)
    env_eval = SurrogateEnv(surrogate, scaler, traversal, C1, C2, goals, device,
                            batch=len(goals), objective=objective)

    # ---- agent --------------------------------------------------------------
    init_blob = {}
    for net in (("actor", "q1", "q2") if algo == "SAC"
                else ("actor", "critic")):
        init_blob[net] = PI.load_policy_init(algo, net, seed)
    if algo == "SAC":
        agent = SAC.SACAgent(device, shared=shared, init_blob=init_blob,
                             adam_backend=adam_backend)
        batch_size = SAC.BATCH_SIZE
    else:
        agent = DDPG.DDPGAgent(device, shared=shared, init_blob=init_blob,
                               adam_backend=adam_backend)
        batch_size = DDPG.BATCH_SIZE

    transfer_manifest = None
    if arm["policy_transfer"]:
        if not source_paths:
            raise ValueError("arm %s needs source_paths" % arm["arm"])
        src = _load_source_blob(source_paths, algo)
        transfer_manifest = agent.apply_transfer(
            src, copy_actor=arm["copy_actor"],
            copy_critics=arm["copy_critics"])
        transfer_manifest["source_paths"] = {
            k: str(v) for k, v in source_paths.items()}
        transfer_manifest["source_sha256"] = {
            k: _sha256_file(Path(v)) for k, v in source_paths.items()}
        _write_json(transfer_manifest, attempt_dir / "transfer_manifest.json")

    # ---- rng streams --------------------------------------------------------
    dev_str = str(device)
    gens = {
        "policy_rollout": torch_gen(subseed(seed, "policy_rollout"), dev_str),
        "replay_sampling": torch_gen(subseed(seed, "replay_sampling"), dev_str),
        "goal_relabel": torch_gen(subseed(seed, "goal_relabel"), dev_str),
        "policy_update_noise": torch_gen(
            subseed(seed, "policy_update_noise"), dev_str),
    }
    if algo == "DDPG":
        gens["ddpg_ou"] = torch_gen(subseed(seed, "ddpg_ou"), dev_str)

    replay = GPUReplay(device=device)
    prefill = torch.as_tensor(PI.load_prefill(seed), dtype=torch.float32,
                              device=device)
    schedule = (goal_schedule(seed, n_episodes, SHARED_GOALS) if shared
                else [goals[0]] * n_episodes)

    goal_tensor_cache = {g: torch.full((1, 1), float(g), dtype=torch.float32,
                                       device=device) for g in set(schedule)}
    goals_t = torch.as_tensor(goals, dtype=torch.long, device=device)
    n_goals = len(goals)

    # ---- resolved config ----------------------------------------------------
    resolved = {
        "case": case,
        "case_id": case_id,
        "protocol_hash": PROTOCOL_HASH,
        "arm": arm,
        "algorithm": algo,
        "domain": domain,
        "goals": goals,
        "traversal": traversal,
        "C1": C1, "C2": C2,
        "objective": objective,
        "episodes": n_episodes,
        "steps_per_episode": STEPS_PER_EPISODE,
        "prefill_episodes": PREFILL_EPISODES,
        "batch_size": batch_size,
        "adam_backend": adam_backend,
        "device": dev_str,
        "gpu": (torch.cuda.get_device_name(device)
                if device.type == "cuda" else None),
        "host": socket.gethostname(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "runtime_flags": flags,
        "env": env.config(),
        "optimizer": agent.optimizer_report(),
        "lr": agent.lr_report(),
        "rng_subseeds": {k: int(subseed(seed, k)) for k in
                         list(gens) + ["goal_schedule", "prefill"]},
        "init_bank": {
            "dir": str(INIT_BANK),
            "files": {net: str(PI.init_path(algo, net, seed))
                      for net in init_blob},
            "sha256": {net: PI.file_sha256(PI.init_path(algo, net, seed))
                       for net in init_blob},
            "prefill": str(PI.prefill_path(seed)),
            "prefill_sha256": PI.file_sha256(PI.prefill_path(seed)),
        },
        "surrogate_path": str(surrogate_path) if surrogate_path else None,
        "scaler_path": str(scaler_path) if scaler_path else None,
        "surrogate_sha256": (_sha256_file(Path(surrogate_path))
                             if surrogate_path else None),
        "source_paths": ({k: str(v) for k, v in source_paths.items()}
                         if source_paths else None),
        "shared": shared,
        "relabel": bool(shared),
        "exclusive": bool(exclusive),
        "resume_interval_s": float(resume_interval_s),
        "eval_episodes": (sorted(int(e) for e in eval_episodes)
                          if eval_episodes is not None
                          else "protocol.evaluation.episodes"),
    }
    _write_json(resolved, attempt_dir / "resolved_config.json")

    # ---- lr assertion before the first update -------------------------------
    lr = agent.lr_report()
    if algo == "SAC":
        assert lr["actor"] == [SAC.ACTOR_LR], lr
        assert lr["q1"] == [SAC.CRITIC_LR] and lr["q2"] == [SAC.CRITIC_LR], lr
        assert lr["alpha"] == [SAC.ALPHA_LR], lr
    else:
        assert lr["actor"] == [DDPG.ACTOR_LR], lr
        assert lr["critic"] == [DDPG.CRITIC_LR], lr

    stat_names = SAC.STAT_NAMES if algo == "SAC" else DDPG.STAT_NAMES
    n_stats = len(stat_names)

    eval_set = (set(EV.EVAL_EPISODES) if eval_episodes is None
                else {int(e) for e in eval_episodes})
    eval_records: list = []
    episode_rows: list = []
    best = {"score": -float("inf"), "episode": None, "actor": None,
            "critics": None}
    nonfinite = False
    t_eval_total = 0.0
    updates_done = 0
    transitions = 0
    last_resume = time.time()

    def _sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def run_evaluation(ep: int) -> float:
        nonlocal t_eval_total, nonfinite
        before_rng = EV.rng_snapshot(gens)
        before_fp = EV.training_fingerprint(agent, replay)
        _sync()
        t0 = time.time()
        recs = EV.evaluate_checkpoint(agent, env_eval, goals, ep, C1, C2,
                                      objective=objective)
        _sync()
        t_eval_total += time.time() - t0
        after_rng = EV.rng_snapshot(gens)
        cmp = EV.rng_equal(before_rng, after_rng)
        if not cmp["all_equal"]:
            raise AssertionError("evaluation at episode %d moved the RNG: %s"
                                 % (ep, cmp))
        EV.assert_unchanged(before_fp,
                            EV.training_fingerprint(agent, replay),
                            "evaluation at episode %d" % ep)
        eval_records.extend(recs)
        scores = [r["selection_score"] for r in recs]
        if not all(np.isfinite(scores)):
            nonfinite = True
        mean_score = float(np.mean(scores))
        if mean_score > best["score"]:
            best["score"] = mean_score
            best["episode"] = int(ep)
            best["actor"] = agent.actor_state()
            best["critics"] = agent.critic_states()
        return mean_score

    # ---- learning phase -----------------------------------------------------
    _sync()
    t_init = time.time() - t_wall_start
    t_learn_start = time.time()
    t_fill_end = None

    for ep in range(n_episodes):
        if ep == PREFILL_EPISODES:
            _sync()
            t_fill_end = time.time()
        if ep in eval_set:
            run_evaluation(ep)
        g = schedule[ep]
        state = env.reset([g])
        goal_col = goal_tensor_cache[g]
        w = env.omega() if shared else None
        if algo == "DDPG" and DDPG.OU_RESET_EVERY_EPISODE:
            agent.noise.reset()
        prefill_phase = ep < PREFILL_EPISODES
        stat_sum = torch.zeros(n_stats, dtype=torch.float32, device=device)
        n_upd_ep = 0
        ep_reward = torch.zeros((), dtype=torch.float32, device=device)

        for t in range(STEPS_PER_EPISODE):
            s = state.clone()
            if prefill_phase:
                action = prefill[ep, t].reshape(1, 1)
            elif algo == "SAC":
                noise = torch.randn((1, 1), generator=gens["policy_rollout"],
                                    device=device)
                action = agent.act(state, noise, w)
            else:
                eps = torch.randn((1,), generator=gens["ddpg_ou"],
                                  device=device)
                action = agent.act(state, eps, w)
            state, reward, done, info = env.step(action)
            replay.add(s, action, reward, state.clone(),
                       torch.full((1, 1), float(done), device=device),
                       u9=info["u9"], actions30=info["actions30"],
                       goal=goal_col)
            transitions += 1
            if done:
                ep_reward = reward.reshape(())
            if not prefill_phase:
                stat_sum += _one_update(agent, algo, replay, batch_size, gens,
                                        shared, C1, C2, objective, device)
                updates_done += 1
                n_upd_ep += 1

        stats = (stat_sum / max(n_upd_ep, 1)).detach().cpu().numpy()
        r_ep = float(ep_reward.detach().cpu())
        if not np.isfinite(stats).all() or not np.isfinite(r_ep):
            nonfinite = True
        row = {"episode": ep, "goal": int(g), "training_reward": r_ep,
               "prefill": int(prefill_phase), "n_updates": n_upd_ep}
        row.update({name: float(stats[i]) for i, name in enumerate(stat_names)})
        episode_rows.append(row)

        if (resume_enabled and not prefill_phase
                and time.time() - last_resume >= float(resume_interval_s)):
            _save_resume(attempt_dir, agent, replay, gens, ep, updates_done,
                         transitions, best, episode_rows, eval_records,
                         resolved)
            last_resume = time.time()

    if n_episodes in eval_set:
        run_evaluation(n_episodes)

    # ---- save the selected checkpoint --------------------------------------
    if best["actor"] is None:  # only reachable with an empty gate schedule
        best["actor"] = agent.actor_state()
        best["critics"] = agent.critic_states()
    _save_atomic(best["actor"], attempt_dir / "best_actor.pt")
    _save_atomic(best["critics"], attempt_dir / "best_critics.pt")
    _sync()
    t_learning = time.time() - t_learn_start

    _save_atomic(agent.actor_state(), attempt_dir / "final_actor.pt")
    _save_atomic(agent.critic_states(), attempt_dir / "final_critics.pt")

    # ---- logs ---------------------------------------------------------------
    _write_episode_log(attempt_dir / "episode_log.csv", episode_rows)
    _write_eval_jsonl(attempt_dir / "eval.jsonl", eval_records)
    curve = EV.curve_from_records(eval_records, "canonical_reward")
    _write_curve(attempt_dir / "curve_canonical.csv", eval_records)

    if eval_records:
        selection = EV.selection_report(
            eval_records,
            strict=(n_episodes == EPISODES_TOTAL and eval_episodes is None))
        if selection["selected_episode"] != best["episode"]:
            raise AssertionError(
                "running best episode %s disagrees with the reference "
                "selection %s" % (best["episode"],
                                  selection["selected_episode"]))
    else:
        selection = {"selected_episode": None, "rule": "no evaluation ran",
                     "goals": goals, "episodes": [], "scores": []}
    _write_json(selection, attempt_dir / "selection.json")

    auc = (EV.early_auc_report(eval_records)
           if (n_episodes >= 300 and eval_episodes is None)
           else {"skipped": "pilot or gate run without the full 0-300 grid"})
    _write_json(auc, attempt_dir / "deterministic_auc.json")

    chosen = [r for r in eval_records
              if r["episode"] == selection.get("selected_episode")]
    designs = []
    for r in chosen:
        t30 = physical_thickness_from_actions(
            np.asarray(r["actions30"], dtype=np.float64), domain)
        designs.append({
            "case_id": case_id, "seed": seed, "domain": domain,
            "goal": r["goal"], "selected_episode": r["episode"],
            "actions30": r["actions30"],
            "physical_thickness30": [float(x) for x in t30],
            "thickness_bounds_mm": _PROTO["domains"][domain][
                "thickness_bounds_mm"],
            "u9_pred": r["u9_pred"],
            "training_objective": r["training_objective"],
            "canonical_reward": r["canonical_reward"],
            "peak_success": r["peak_success"],
            "bilateral_margin": r["bilateral_margin"],
            "kappa": r["kappa"],
            "normalized_thickness": r["normalized_thickness"],
            "traversal": traversal,
        })
    _write_json({"case_id": case_id, "selected_episode":
                 selection["selected_episode"], "designs": designs},
                attempt_dir / "selected_designs.json")

    t_fill = (float(t_fill_end - t_learn_start) if t_fill_end is not None
              else float(t_learning))
    post_fill_episodes = max(n_episodes - PREFILL_EPISODES, 0)
    t_post_fill = float(t_learning - t_fill)
    timing = {
        "t_init_s": float(t_init),
        "t_learning_s": float(t_learning),
        "t_fill_s": t_fill,
        "t_post_fill_s": t_post_fill,
        "post_fill_episodes": int(post_fill_episodes),
        "t_eval_total_s": float(t_eval_total),
        "n_evaluations": int(len({r["episode"] for r in eval_records})),
        "t_learn_start_epoch": float(t_learn_start),
        "t_fill_end_epoch": (float(t_fill_end) if t_fill_end is not None
                             else None),
        "t_learn_end_epoch": float(t_learn_start + t_learning),
        "updates_done": int(updates_done),
        "transitions": int(transitions),
        "episodes": int(n_episodes),
        "exclusive": bool(exclusive),
        "device": dev_str,
        "gpu": resolved["gpu"],
        "updates_per_s": (float(updates_done) / t_learning
                          if t_learning > 0 else None),
        "t_learning_scope": ("from just before the episode-0 evaluation to the "
                             "completed save of the selected checkpoint; "
                             "includes fill, training and scheduled "
                             "evaluations, excludes FEA"),
    }
    _write_json(timing, attempt_dir / "timing.json")

    expected_transitions = n_episodes * STEPS_PER_EPISODE
    expected_updates = ((n_episodes - PREFILL_EPISODES) * STEPS_PER_EPISODE
                        if n_episodes > PREFILL_EPISODES else 0)
    artifacts = ["best_actor.pt", "best_critics.pt", "final_actor.pt",
                 "final_critics.pt", "episode_log.csv", "eval.jsonl",
                 "curve_canonical.csv", "selection.json", "deterministic_auc.json",
                 "selected_designs.json", "timing.json",
                 "resolved_config.json"]
    if transfer_manifest is not None:
        artifacts.append("transfer_manifest.json")
    result = {
        "complete": True,
        "case_id": case_id,
        "attempt": int(case.get("attempt", 1)),
        "kind": "policy",
        "episodes_done": int(n_episodes),
        "updates_done": int(updates_done),
        "transitions": int(transitions),
        "eval_count": len({r["episode"] for r in eval_records}),
        "counts_match_protocol": bool(
            transitions == expected_transitions
            and updates_done == expected_updates),
        "expected_transitions": int(expected_transitions),
        "expected_updates": int(expected_updates),
        "selected_episode": selection["selected_episode"],
        "best_mean_selection_score": float(best["score"]),
        "nonfinite": bool(nonfinite),
        "adam_backend": adam_backend,
        "protocol_hash": PROTOCOL_HASH,
        "host": socket.gethostname(),
        "device": dev_str,
        "gpu": resolved["gpu"],
        "vram_peak_bytes": (int(torch.cuda.max_memory_allocated(device))
                            if device.type == "cuda" else None),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S",
                                    time.localtime(t_wall_start)),
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "t_train_s": float(time.time() - t_wall_start),
        "artifact_sha256": {a: _sha256_file(attempt_dir / a)
                            for a in artifacts
                            if (attempt_dir / a).exists()},
        "generator_states_sha256": {
            k: hashlib.sha256(g.get_state().cpu().numpy().tobytes()).hexdigest()
            for k, g in gens.items()},
    }
    if n_episodes == EPISODES_TOTAL and not result["counts_match_protocol"]:
        result["complete"] = False
    _write_json(result, attempt_dir / "result.json")
    return result


def _one_update(agent, algo: str, replay: GPUReplay, batch_size: int,
                gens: dict, shared: bool, C1: float, C2: float,
                objective: str, device) -> torch.Tensor:
    """Draw one minibatch, relabel it when goal-conditioned, and update."""
    b = replay.sample(batch_size, gens["replay_sampling"])
    batch = {"s": b["s"], "a": b["a"], "r": b["r"], "s2": b["s2"],
             "done": b["done"]}
    if shared:
        k = b["s"].shape[0]
        pick = torch.randint(0, GOALS_PER_BLOCK, (k,),
                             generator=gens["goal_relabel"], device=device)
        new_goal = torch.as_tensor(SHARED_GOALS, dtype=torch.long,
                                   device=device)[pick]
        batch["r"] = relabel_terminal_rewards(b["u9"], b["actions30"],
                                              b["done"], new_goal, C1, C2,
                                              objective=objective)
        batch["w"] = ((new_goal.to(torch.float32) - 5.0) / 2.0).unsqueeze(1)
    if algo == "SAC":
        nt = torch.randn((batch_size, 1),
                         generator=gens["policy_update_noise"], device=device)
        na = torch.randn((batch_size, 1),
                         generator=gens["policy_update_noise"], device=device)
        return agent.update(batch, nt, na)
    return agent.update(batch)


# ------------------------------------------------------------------ log files
def _write_episode_log(path: Path, rows: list) -> None:
    if not rows:
        return
    cols = list(rows[0].keys())
    tmp = path.with_suffix(".csv.tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=cols)
        wr.writeheader()
        wr.writerows(rows)
    tmp.replace(path)


def _write_eval_jsonl(path: Path, records: list) -> None:
    tmp = path.with_suffix(".jsonl.tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, default=_json_default) + "\n")
    tmp.replace(path)


def _write_curve(path: Path, records: list) -> None:
    canon = EV.curve_from_records(records, "canonical_reward")
    train = EV.curve_from_records(records, "training_objective")
    cols = (["episode", "canonical_mean", "training_objective_mean"]
            + ["canonical_g%d" % g for g in canon["goals"]]
            + ["training_objective_g%d" % g for g in train["goals"]])
    tmp = path.with_suffix(".csv.tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        wr.writerow(cols)
        for i, ep in enumerate(canon["episodes"]):
            row = [ep, canon["mean"][i], train["mean"][i]]
            row += [canon["per_goal"][g][i] for g in canon["goals"]]
            row += [train["per_goal"][g][i] for g in train["goals"]]
            wr.writerow(row)
    tmp.replace(path)


# --------------------------------------------------------------------- resume
def _save_resume(attempt_dir: Path, agent, replay, gens: dict, episode: int,
                 updates_done: int, transitions: int, best: dict,
                 episode_rows: list, eval_records: list,
                 resolved: dict) -> Path:
    """Snapshot the full restartable state at a completed update boundary."""
    d = attempt_dir / "resume"
    d.mkdir(parents=True, exist_ok=True)
    blob = {
        "episode_next": int(episode) + 1,
        "updates_done": int(updates_done),
        "transitions": int(transitions),
        "networks": agent.cpu_state(),
        "optimizer": agent.optimizer_state(),
        "replay": replay.state_dict(),
        "generators": {k: g.get_state().clone() for k, g in gens.items()},
        "best": {"score": best["score"], "episode": best["episode"],
                 "actor": best["actor"], "critics": best["critics"]},
        "episode_rows": episode_rows,
        "eval_records": eval_records,
        "case_id": resolved["case_id"],
        "host": resolved["host"],
        "device": resolved["device"],
        "adam_backend": resolved["adam_backend"],
        "protocol_hash": PROTOCOL_HASH,
    }
    path = d / "latest.pt"
    _save_atomic(blob, path)
    return path


def load_resume(attempt_dir, agent, replay, gens: dict) -> dict:
    """Restore a snapshot written by :func:`_save_resume`.

    Resume is valid only for the same case, config and host; anything else must
    start a new attempt (protocol.orchestration.resume).
    """
    path = Path(attempt_dir) / "resume" / "latest.pt"
    blob = torch.load(path, map_location="cpu", weights_only=False)
    agent.load_cpu_state(blob["networks"])
    agent.load_optimizer_state(blob["optimizer"])
    replay.load_state_dict(blob["replay"])
    for k, g in gens.items():
        if k in blob["generators"]:
            g.set_state(blob["generators"][k])
    return blob


__all__ = ["run_policy", "goal_schedule", "resolve_arm", "load_resume",
           "EPISODES_TOTAL", "STEPS_PER_EPISODE", "PREFILL_EPISODES",
           "TRANSITIONS_EXPECTED", "UPDATES_EXPECTED", "SHARED_GOALS",
           "TRANSFER_ARMS"]
