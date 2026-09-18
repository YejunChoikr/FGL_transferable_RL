"""Pre-flight tests -> results/unit_tests_<machine>.json. Non-zero exit blocks training."""
from __future__ import annotations

import json
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
    C1_GRID, C2_GRID, CFG, CRITIC_DIM, SEEDS, STATE_DIM, agent_cfg, atomic_write_json,
    det_schedule, episodes, lock,
)
from env_cc import make_env, set_seed  # noqa: E402
from reward_cc import (  # noqa: E402
    ORIGINAL_C1, ORIGINAL_C2, TASK_INDICES, kappa, numerator, physical_metrics,
    reward, rho_a, weight_W,
)
from sac_agent import SACAgent  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
TASKS = list(CFG["tasks"].keys())
R = {}


def check(name):
    def deco(fn):
        def w(*a, **k):
            t0 = time.perf_counter()
            try:
                R[name] = {"pass": True, "detail": fn(*a, **k),
                           "s": round(time.perf_counter() - t0, 3)}
            except Exception as e:
                R[name] = {"pass": False, "error": f"{type(e).__name__}: {e}",
                           "s": round(time.perf_counter() - t0, 3)}
            return R[name]
        return w
    return deco


@check("t01_C2_zero_is_W_equals_one")
def t01():
    """THE critical branch: C2==0 -> W=1, never 0.5 + 0*rho_A."""
    rng = np.random.default_rng(0)
    a = rng.uniform(-1, 1, 30)
    rho = rho_a(a)
    assert weight_W(0.0, rho) == 1.0, weight_W(0.0, rho)
    assert weight_W(0, rho) == 1.0
    naive = 0.5 + 0.0 * rho
    assert naive == 0.5 and weight_W(0.0, rho) != naive, "C2=0 must NOT collapse to 0.5"
    for c2 in (0.25, 0.5, 0.75, 1.0):
        assert abs(weight_W(c2, rho) - (0.5 + c2 * rho)) < 1e-15
    u = np.array([0.7, 1.7, 3.0, 4.0, 4.4, 4.9, 5.0, 3.4, 1.8])
    r0 = reward(u, a, "FGL6", 0.5, 0.0)
    assert abs(r0 - numerator(u, "FGL6", 0.5)) < 1e-15, "C2=0 reward must equal the numerator"
    r_naive = numerator(u, "FGL6", 0.5) / 0.5
    assert abs(r0 - r_naive) > 1e-6, "the wrong branch would double the reward"
    return {"W_at_C2_0": weight_W(0.0, rho), "W_at_C2_1": weight_W(1.0, rho),
            "reward_C2_0": r0, "wrong_branch_would_give": r_naive}


@check("t02_original_pair_reproduces_locked_reward")
def t02():
    """(C1,C2)=(0.5,1.0) must reproduce the original density-normalised reward."""
    rng = np.random.default_rng(1)
    worst = 0.0
    for _ in range(200):
        a = rng.uniform(-1, 1, 30)
        u = rng.uniform(0.2, 6.0, 9)
        for t in TASKS:
            ix = TASK_INDICES[t]
            dens = float(np.sum((a + 2) / 2) / 30)          # original density code
            orig = (u[ix["target"]] - 0.5 * (u[ix["lower"]] + u[ix["upper"]])) / dens
            worst = max(worst, abs(reward(u, a, t, ORIGINAL_C1, ORIGINAL_C2) - orig))
    assert worst < 1e-12, worst
    a = rng.uniform(-1, 1, 30)
    assert abs((0.5 + rho_a(a)) - float(np.sum((a + 2) / 2) / 30)) < 1e-15
    return {"max_abs_diff_vs_original": worst,
            "identity": "0.5 + rho_A == mean((action+2)/2)"}


@check("t03_kappa_is_coefficient_free")
def t03():
    """kappa must not depend on C1 or C2 -> valid across coefficient cells."""
    rng = np.random.default_rng(2)
    u = rng.uniform(0.5, 6.0, 9)
    vals = set()
    for t in TASKS:
        k = [kappa(u, t)]
        for c1 in C1_GRID:
            for c2 in C2_GRID:
                _ = reward(u, rng.uniform(-1, 1, 30), t, c1, c2)
            vals.add(round(kappa(u, t), 15))
        ix = TASK_INDICES[t]
        manual = u[ix["target"]] / (0.5 * (u[ix["lower"]] + u[ix["upper"]]))
        assert abs(k[0] - manual) < 1e-15
    assert len(vals) == len(TASKS)
    return {"kappa_per_task": {t: kappa(u, t) for t in TASKS}}


@check("t04_numerator_scales_with_C1")
def t04():
    u = np.array([1.0, 2.0, 3.0, 4.0, 9.0, 5.0, 4.0, 3.0, 2.0])
    got = {c: numerator(u, "FGL5", c) for c in C1_GRID}
    for c in C1_GRID:
        assert abs(got[c] - (9.0 - c * (4.0 + 5.0))) < 1e-15
    assert got[0.1] > got[0.5], "larger C1 must suppress more"
    return {"numerators": got}


@check("t05_env_reward_matches_reward_module")
def t05():
    """Environment's terminal reward == reward_cc.reward for the same design."""
    rng = np.random.default_rng(3)
    worst = 0.0
    for (c1, c2) in ((0.5, 0.0), (0.3, 1.0), (0.1, 0.25), (0.5, 1.0)):
        e = make_env("FGL6", c1, c2, CFG, ROOT, DEV)
        for _ in range(4):
            a = rng.uniform(-1, 1, 30)
            e.reset()
            r_last = None
            for x in a:
                _, r, done = e.replay_step(np.array([x], dtype=np.float64))
                r_last = r
            u = e.surrogate_profile()
            worst = max(worst, abs(float(r_last) - reward(u, a, "FGL6", c1, c2)))
            assert abs(e.rho_A() - rho_a(a)) < 1e-12
            assert abs(e.W() - weight_W(c2, rho_a(a))) < 1e-12
        del e
    assert worst < 1e-9, worst
    return {"max_abs_env_vs_module": worst}


@check("t06_state_60_actor60_critic61")
def t06():
    e = make_env("FGL6", 0.5, 1.0, CFG, ROOT, DEV)
    assert tuple(e.observation_space.shape) == (STATE_DIM,)
    s = e.reset()
    assert s.shape == (STATE_DIM,) and s.dtype == np.float32 and np.all(s == 0.0)
    done, n = False, 0
    while not done:
        s, r, done = e.replay_step(np.array([0.2], dtype=np.float64))
        n += 1
        assert s.shape == (STATE_DIM,) and s.dtype == np.float32 and np.all(np.isfinite(s))
    assert n == 30 and np.array_equal(s, e.last_surrogate_input)
    ag = SACAgent(e.observation_space.shape[0], DEV, agent_cfg())
    assert ag.actor.fc1.in_features == STATE_DIM
    assert ag.q1.fc1.in_features == CRITIC_DIM and ag.q2.fc1.in_features == CRITIC_DIM
    o = {"steps": n, "actor_in": STATE_DIM, "critic_in": CRITIC_DIM}
    del ag, e
    torch.cuda.empty_cache()
    return o


@check("t07_inherits_calibration_H1_unchanged")
def t07():
    lk, a = lock(), agent_cfg()
    p = lk["params"]
    assert lk["selected_config"] == "H1"
    expect = {"batch_size": 256, "actor_lr": 3e-4, "critic_lr": 3e-4, "alpha_lr": 3e-4,
              "tau": 0.005, "gamma": 0.99, "episodes": 1500, "episode_length": 30,
              "memory_fill_episodes": 20, "replay_capacity": 100000,
              "updates_per_transition": 1, "initial_alpha": 0.2, "target_entropy": -1,
              "weight_decay": 0, "reward_scale": 1}
    for k, v in expect.items():
        assert abs(float(p[k]) - float(v)) < 1e-12, f"{k}: {p[k]} != {v}"
    assert a["train"]["batch_size"] == 256 and a["sac"]["actor_lr"] == 3e-4
    assert bool(p["automatic_entropy_tuning"]) is True
    return {"selected_config": lk["selected_config"], "lock_sha256": lk["lock_sha256"],
            **{k: p[k] for k in ("batch_size", "actor_lr", "tau", "episodes")}}


@check("t08_det_schedule_is_ep20_plus_every25")
def t08():
    s = det_schedule(episodes())
    assert 20 in s, "episode 20 must be evaluated"
    assert 25 in s and 50 in s and 1500 in s
    assert s[0] == 20 and s[1] == 25
    every25 = [e for e in s if e % 25 == 0]
    assert len(every25) == 60, len(every25)
    assert len(s) == 61, len(s)
    return {"n_evals": len(s), "first": s[:4], "last": s[-2:]}


@check("t09_physical_metrics_complete")
def t09():
    rng = np.random.default_rng(5)
    a = rng.uniform(-1, 1, 30)
    u = np.array([0.7, 1.7, 3.0, 4.0, 4.4, 4.9, 5.0, 3.4, 1.8])
    m = physical_metrics(u, a, "FGL7", 0.3, 0.5)
    need = ["u", "rho_A", "kappa", "m_lower", "m_upper", "bilateral_raw",
            "bilateral_normalized", "global_margin", "target_is_global_peak",
            "own_objective_reward", "original_locked_reward_at_C1_0.5_C2_1.0",
            "action", "W", "numerator"]
    for k in need:
        assert k in m, k
    assert len(m["u"]) == 9 and len(m["action"]) == 30
    assert abs(m["bilateral_raw"] - min(m["m_lower"], m["m_upper"])) < 1e-15
    assert abs(m["bilateral_normalized"] - m["bilateral_raw"] / (0.5 + m["rho_A"])) < 1e-15
    # FGL7 target u7=5.0 is the global peak here
    assert m["target_is_global_peak"] is True
    assert abs(m["own_objective_reward"] - reward(u, a, "FGL7", 0.3, 0.5)) < 1e-15
    assert abs(m["original_locked_reward_at_C1_0.5_C2_1.0"]
               - reward(u, a, "FGL7", 0.5, 1.0)) < 1e-15
    return {k: m[k] for k in ("rho_A", "kappa", "bilateral_raw", "bilateral_normalized",
                              "global_margin", "W", "own_objective_reward")}


@check("t10_seeded_determinism_and_no_nan")
def t10():
    outs = []
    for _ in range(2):
        set_seed(11)
        e = make_env("FGL5", 0.4, 0.75, CFG, ROOT, DEV)
        ag = SACAgent(e.observation_space.shape[0], DEV, agent_cfg())
        s = e.reset(); done = False; acts = []
        while not done:
            a = ag.deterministic_action(s)
            acts.append(float(a[0]))
            s, r, done = e.replay_step(np.array([a[0]], dtype=np.float64))
        u = e.surrogate_profile()
        assert np.all(np.isfinite(u)) and np.isfinite(r)
        outs.append((acts, float(r), u.tolist()))
        del ag, e
        torch.cuda.empty_cache()
    assert outs[0][0] == outs[1][0], "deterministic rollout not reproducible"
    assert outs[0][1] == outs[1][1]
    return {"reward": outs[0][1], "n_actions": len(outs[0][0]), "reproducible": True}


def main():
    machine = sys.argv[1] if len(sys.argv) > 1 else "local"
    set_seed(20260727)
    t01(); t02(); t03(); t04(); t05(); t06(); t07(); t08(); t09(); t10()
    n = sum(1 for v in R.values() if v["pass"])
    atomic_write_json(ROOT / "results" / f"unit_tests_{machine}.json",
                      {"machine_id": machine, "device": DEV, "n_pass": n, "n_total": len(R),
                       "all_pass": n == len(R),
                       "failures": [k for k, v in R.items() if not v["pass"]],
                       "checks": R, "time": time.strftime("%Y-%m-%dT%H:%M:%S")})
    for k, v in R.items():
        print(f"  {'PASS' if v['pass'] else 'FAIL'}  {k}"
              + ("" if v["pass"] else f"   {v.get('error')}"))
    print(f"[unit_tests] {n}/{len(R)} pass on {machine}", flush=True)
    return 0 if n == len(R) else 1


if __name__ == "__main__":
    raise SystemExit(main())
