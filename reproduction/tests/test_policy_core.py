"""Regression tests for policy networks, transfer, replay and evaluation."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from cmame_rt import ddpg as DDPG  # noqa: E402
from cmame_rt import evaluation as EV  # noqa: E402
from cmame_rt import models_policy as M  # noqa: E402
from cmame_rt import policy_init as PI  # noqa: E402
from cmame_rt import policy_train as PT  # noqa: E402
from cmame_rt import sac as SAC  # noqa: E402
from cmame_rt.env import SurrogateEnv, relabel_terminal_rewards  # noqa: E402
from cmame_rt.protocol import load_protocol, load_reference_math  # noqa: E402
from cmame_rt.replay import (GPUReplay, chi_square_uniform,  # noqa: E402
                             sampling_histogram, within_batch_unique)
from cmame_rt.reward import reward as reward_ref  # noqa: E402
from cmame_rt.rng import subseed, torch_gen  # noqa: E402

PROTO = load_protocol()
REF = load_reference_math()


# ----------------------------------------------------------------- networks
def test_input_widths_and_keys():
    assert M.actor_in_dim(False) == 60 and M.actor_in_dim(True) == 61
    assert M.critic_in_dim(False) == 61 and M.critic_in_dim(True) == 62
    a = M.make_actor("SAC", False)
    assert tuple(a.state_dict()) == M.expected_keys("actor", "SAC")
    d = M.make_actor("DDPG", False)
    assert tuple(d.state_dict()) == M.expected_keys("actor", "DDPG")
    c = M.make_critic(False)
    assert tuple(c.state_dict()) == M.expected_keys("critic")
    assert M.count_params(a) == 11264514
    assert M.count_params(c) == 11268097
    assert M.count_params(M.make_actor("SAC", True)) == 11268610
    assert M.count_params(M.make_critic(True)) == 11272193


def test_sac_logprob_matches_torch_distribution():
    torch.manual_seed(0)
    a = M.make_actor("SAC", False)
    s = torch.randn(16, 60)
    n = torch.randn(16, 1)
    act, logp = a.rsample(s, n)
    mu, log_std = a(s)
    std = log_std.exp()
    z = mu + std * n
    ref = (torch.distributions.Normal(mu, std).log_prob(z)
           - torch.log(1 - torch.tanh(z).pow(2) + M.TANH_JACOBIAN_EPS))
    assert torch.allclose(act, torch.tanh(z))
    assert torch.allclose(logp, ref.sum(-1, keepdim=True), atol=1e-6)


def test_deterministic_consumes_no_rng():
    a = M.make_actor("SAC", False)
    s = torch.randn(8, 60)
    before = torch.get_rng_state().clone()
    out1 = a.deterministic(s)
    after = torch.get_rng_state().clone()
    assert torch.equal(before, after)
    assert torch.equal(out1, a.deterministic(s))
    mu, _ = a(s)
    assert torch.equal(out1, torch.tanh(mu))


# ---------------------------------------------------------------------- init
def test_init_is_a_function_of_seed_and_stream():
    b1 = PI.build_policy_init(0, "SAC")
    b2 = PI.build_policy_init(0, "SAC")
    assert PI.state_dict_sha256(b1["actor"]) == PI.state_dict_sha256(b2["actor"])
    assert (PI.state_dict_sha256(b1["q1"])
            != PI.state_dict_sha256(b1["q2"]))
    assert (PI.state_dict_sha256(PI.build_policy_init(1, "SAC")["actor"])
            != PI.state_dict_sha256(b1["actor"]))


def test_init_does_not_disturb_global_rng():
    torch.manual_seed(1234)
    before = torch.get_rng_state().clone()
    PI.build_policy_init(2, "DDPG")
    assert torch.equal(before, torch.get_rng_state())


def test_widening_is_function_preserving():
    sd = PI.build_policy_init(0, "SAC")["actor"]
    spec = M.make_actor("SAC", False)
    spec.load_state_dict(sd)
    sh = M.make_actor("SAC", True)
    sh.load_state_dict(PI.widen_fc1_with_zero_goal_column(sd))
    s = torch.randn(8, 60)
    for omega in (-1.0, 0.0, 0.5, 1.0):
        w = torch.full((8, 1), omega)
        assert torch.equal(spec.deterministic(s), sh.deterministic(s, w))


def test_prefill_shape_and_range():
    p = PI.build_prefill(0)
    assert p.shape == (20, 30) and p.dtype == np.float32
    assert np.all(np.abs(p) <= 1.0)
    assert np.array_equal(p, PI.build_prefill(0))
    assert not np.array_equal(p, PI.build_prefill(1))


# ------------------------------------------------------------------ transfer
def _transfer_case(algo, agent_cls, nets, copy_critics):
    src = PI.build_policy_init(3, algo)
    tgt = PI.build_policy_init(0, algo)
    agent = agent_cls("cpu", shared=False, init_blob=tgt,
                      adam_backend="reference")
    heads_before = {n: PI.head_hashes(getattr(agent, n),
                                      "actor" if n == "actor" else "critic",
                                      algo) for n in nets}
    man = agent.apply_transfer(src, copy_actor=True, copy_critics=copy_critics)
    for n in nets:
        net = getattr(agent, n)
        want_src = n == "actor" or copy_critics
        ref = src[n] if want_src else tgt[n]
        assert PI.trunk_hashes(net) == {k: PI.tensor_sha256(ref[k])
                                        for k in PI.TRANSFER_KEYS}
        assert PI.head_hashes(net, "actor" if n == "actor" else "critic",
                              algo) == heads_before[n]
    assert man["heads_unchanged"]
    assert man["optimizer_state_transferred"] is False
    return agent


def test_sac_transfer_arms():
    a = _transfer_case("SAC", SAC.SACAgent, ("actor", "q1", "q2"), True)
    for on, tg in ((a.q1, a.q1_target), (a.q2, a.q2_target)):
        for k, v in on.state_dict().items():
            assert torch.equal(v, tg.state_dict()[k])
    assert all(len(o.state) == 0 for o in (a.actor_opt, a.q1_opt, a.q2_opt,
                                           a.alpha_opt))
    assert abs(float(a.alpha.detach()) - 0.2) < 1e-7
    _transfer_case("SAC", SAC.SACAgent, ("actor", "q1", "q2"), False)


def test_ddpg_transfer_includes_target_actor():
    a = _transfer_case("DDPG", DDPG.DDPGAgent, ("actor", "critic"), True)
    for k, v in a.actor.state_dict().items():
        assert torch.equal(v, a.actor_target.state_dict()[k])
    for k, v in a.critic.state_dict().items():
        assert torch.equal(v, a.critic_target.state_dict()[k])


def test_protocol_forbids_freeze_and_warmup():
    assert PROTO["policy"]["transfer"]["freeze_or_critic_warmup"] is False
    assert PROTO["policy"]["transfer"]["all_trainable"] is True


# ----------------------------------------------------------------- optimizer
def test_learning_rates_match_protocol():
    sac = SAC.SACAgent("cpu", False, PI.build_policy_init(0, "SAC"),
                       "reference")
    lr = sac.lr_report()
    assert lr["actor"] == [3e-4] and lr["q1"] == [3e-4] and lr["q2"] == [3e-4]
    assert lr["alpha"] == [3e-4]
    dd = DDPG.DDPGAgent("cpu", False, PI.build_policy_init(0, "DDPG"),
                        "reference")
    assert dd.lr_report()["actor"] == [1e-4]
    assert dd.lr_report()["critic"] == [1e-3]
    g = sac.actor_opt.param_groups[0]
    assert tuple(g["betas"]) == (0.9, 0.999)
    assert g["eps"] == 1e-8 and g["weight_decay"] == 0.0
    assert g["amsgrad"] is False


def test_actor_update_leaves_critic_grads_untouched_but_moves_actor():
    blob = PI.build_policy_init(0, "SAC")
    ag = SAC.SACAgent("cpu", False, blob, "reference")
    before = {k: v.clone() for k, v in ag.actor.state_dict().items()}
    b = 8
    batch = {"s": torch.randn(b, 60), "a": torch.randn(b, 1).clamp(-1, 1),
             "r": torch.randn(b, 1), "s2": torch.randn(b, 60),
             "done": torch.zeros(b, 1)}
    ag.update(batch, torch.randn(b, 1), torch.randn(b, 1))
    assert any(not torch.equal(before[k], v)
               for k, v in ag.actor.state_dict().items())
    assert all(p.requires_grad for p in ag.q1.parameters())
    assert all(p.requires_grad for p in ag.q2.parameters())
    assert ag.n_updates == 1


# -------------------------------------------------------------------- replay
def test_replay_is_uniform_without_replacement():
    rp = GPUReplay(capacity=100000, device="cpu")
    n = 2000
    g = np.random.default_rng(0)
    rp.add(torch.as_tensor(g.normal(size=(n, 60)).astype(np.float32)),
           torch.as_tensor(g.uniform(-1, 1, (n, 1)).astype(np.float32)),
           torch.as_tensor(g.normal(size=(n, 1)).astype(np.float32)),
           torch.as_tensor(g.normal(size=(n, 60)).astype(np.float32)),
           torch.zeros(n, 1), goal=torch.full((n, 1), 5.0))
    assert len(rp) == n and rp.capacity == PROTO["policy"]["replay_capacity"]
    gen = torch_gen(subseed(0, "replay_sampling"), "cpu")
    idx = rp.sample_indices(256, gen)
    assert within_batch_unique(idx)
    assert rp.compare_with_reference(idx)["identical"]
    counts = sampling_histogram(rp, 256, 300, gen)
    chi = chi_square_uniform(counts, batch_size=256)
    assert abs(chi["z_wilson_hilferty"]) < 4.0


def test_replay_resume_roundtrip():
    rp = GPUReplay(capacity=1000, device="cpu")
    n = 100
    rp.add(torch.randn(n, 60), torch.randn(n, 1), torch.randn(n, 1),
           torch.randn(n, 60), torch.zeros(n, 1), goal=torch.full((n, 1), 4.0))
    sd = rp.state_dict()
    rp2 = GPUReplay(capacity=1000, device="cpu")
    rp2.load_state_dict(sd)
    assert len(rp2) == len(rp) and rp2.position == rp.position
    idx = torch.arange(n)
    for k in ("s", "a", "r", "s2", "done", "goal", "omega"):
        assert torch.equal(rp.gather(idx)[k], rp2.gather(idx)[k])


# ----------------------------------------------------------------- environment
class _ToySurrogate(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(60, 9)

    def forward(self, x):
        return self.lin(x.reshape(x.shape[0], -1))


def _toy_env(goals, batch=1, C1=0.5, C2=1.0, traversal="rowwise_raster"):
    torch.manual_seed(0)
    return SurrogateEnv(_ToySurrogate(), {"mean": np.zeros(9),
                                          "scale": np.ones(9)},
                        traversal, C1, C2, goals, "cpu", batch=batch)


def test_terminal_only_at_step_30_and_reward_matches_reference():
    env = _toy_env([5])
    env.reset([5])
    acts = np.random.default_rng(1).uniform(-1, 1, 30).astype(np.float32)
    for t in range(30):
        _, r, done, info = env.step(torch.as_tensor(acts[t]).reshape(1, 1))
        assert done == (t == 29)
        if t < 29:
            assert float(r) == 0.0 and info["u9"] is None
    u = info["u9"].numpy().astype(np.float64)[0]
    a = info["actions30"].numpy().astype(np.float64)[0]
    want = REF.design_metrics(u, a, 5, 0.5, 1.0)["reward"]
    assert abs(float(r) - want) <= 1e-6 * max(abs(want), 1.0)
    assert abs(reward_ref(u, a, 5) - want) == 0.0


def test_every_traversal_visits_each_cell_once():
    acts = np.random.default_rng(2).uniform(-1, 1, 30).astype(np.float32)
    finals = []
    for name in PROTO["encoding"]["traversal_indices"]:
        env = _toy_env([5], traversal=name)
        env.reset([5])
        for t in range(30):
            k = env.order[t]
            env.step(torch.as_tensor(acts[k]).reshape(1, 1))
            assert int(torch.count_nonzero(env.state[:, 1::2])) == t + 1
        finals.append(env.state.clone())
    for f in finals[1:]:
        assert torch.equal(finals[0], f)


def test_assigned_zero_is_distinguishable_from_unassigned():
    env = _toy_env([5])
    env.reset([5])
    assert float(env.state[0, 0]) == 0.0 and float(env.state[0, 1]) == 0.0
    env.step(torch.zeros(1, 1))
    assert float(env.state[0, 0]) == 0.0 and float(env.state[0, 1]) > 0.0


def test_relabel_zeroes_non_terminal_and_matches_direct():
    g = np.random.default_rng(5)
    n = 64
    u9 = torch.as_tensor(g.normal(0, 2, (n, 9)).astype(np.float32))
    a30 = torch.as_tensor(g.uniform(-1, 1, (n, 30)).astype(np.float32))
    done = torch.as_tensor((g.random(n) > 0.5).astype(np.float32)).reshape(-1, 1)
    goal = torch.as_tensor(g.integers(3, 8, n))
    r = relabel_terminal_rewards(u9, a30, done, goal, 0.5, 1.0).reshape(-1)
    direct = SurrogateEnv.reward_from(u9, a30, goal, 0.5, 1.0)
    assert torch.equal(r, direct * done.reshape(-1))
    assert float(r[done.reshape(-1) == 0].abs().max() if
                 (done == 0).any() else 0.0) == 0.0


# ----------------------------------------------------------------- evaluation
def test_eval_schedule_and_reference_agreement():
    assert EV.EVAL_EPISODES == tuple(range(0, 1501, 25))
    assert EV.N_EVALS == 61
    e = np.arange(0, 1501, 25)
    assert EV.early_auc(e, np.ones(len(e))) == REF.early_auc(e, np.ones(len(e)))
    assert EV.early_auc(e, e / 300.0) == REF.early_auc(e, e / 300.0)
    scores = np.zeros((61, 5))
    scores[7, :] = 2
    scores[8, 0] = 3
    assert EV.select_checkpoint(e, scores) == REF.select_checkpoint(e, scores)
    assert EV.select_checkpoint(e, np.ones((61, 5))) == 0  # episode 0 can win


def test_rng_snapshot_detects_a_draw():
    gens = {"x": torch_gen(7, "cpu")}
    a = EV.rng_snapshot(gens)
    assert EV.rng_equal(a, EV.rng_snapshot(gens))["all_equal"]
    torch.randn(1, generator=gens["x"])
    assert not EV.rng_equal(a, EV.rng_snapshot(gens))["all_equal"]
    b = EV.rng_snapshot(gens)
    torch.randn(1)
    assert not EV.rng_equal(b, EV.rng_snapshot(gens))["all_equal"]


def test_evaluation_leaves_agent_untouched():
    env = _toy_env([3, 4, 5, 6, 7], batch=5)
    ag = SAC.SACAgent("cpu", True, PI.build_policy_init(0, "SAC"), "reference")
    fp = EV.training_fingerprint(ag)
    before = EV.rng_snapshot({})
    recs = EV.evaluate_checkpoint(ag, env, [3, 4, 5, 6, 7], 0, 0.5, 1.0)
    assert len(recs) == 5
    assert EV.rng_equal(before, EV.rng_snapshot({}))["all_equal"]
    assert EV.training_fingerprint(ag) == fp
    for r in recs:
        assert set(("episode", "goal", "actions30", "u9_pred",
                    "training_objective", "selection_score",
                    "canonical_reward", "peak_success", "bilateral_margin",
                    "kappa")) <= set(r)


def test_training_and_canonical_objectives_are_separate_fields():
    env = _toy_env([5], C1=0.4, C2=0.75)
    ag = SAC.SACAgent("cpu", False, PI.build_policy_init(0, "SAC"),
                      "reference")
    rec = EV.evaluate_checkpoint(ag, env, [5], 0, 0.4, 0.75)[0]
    u = np.asarray(rec["u9_pred"])
    a = np.asarray(rec["actions30"])
    assert abs(rec["training_objective"]
               - REF.design_metrics(u, a, 5, 0.4, 0.75)["reward"]) < 1e-9
    assert abs(rec["canonical_reward"]
               - REF.design_metrics(u, a, 5, 0.5, 1.0)["reward"]) < 1e-9
    assert rec["selection_score"] == rec["training_objective"]


# ------------------------------------------------------------------ schedules
def test_shared_goal_schedule_is_block_balanced():
    sched = PT.goal_schedule(0, 1500, PT.SHARED_GOALS)
    assert len(sched) == 1500
    for i in range(0, 1500, 5):
        assert sorted(sched[i:i + 5]) == sorted(PT.SHARED_GOALS)
    for g in PT.SHARED_GOALS:
        assert sched.count(g) == 300
        assert sched[:20].count(g) == 4          # prefill_episodes_per_goal
    assert sched == PT.goal_schedule(0, 1500, PT.SHARED_GOALS)
    assert sched != PT.goal_schedule(1, 1500, PT.SHARED_GOALS)


def test_counters_match_protocol():
    p = PROTO["policy"]
    assert p["episodes_total"] * p["steps_per_episode"] == 45000
    assert ((p["episodes_total"] - p["prefill_episodes_in_total"])
            * p["steps_per_episode"]) == 44400
    assert p["training_transition_count"] == 45000
    assert p["gradient_update_count"] == 44400


def test_arm_resolution():
    assert PT.resolve_arm({"arm": "ST_P0"})["policy_transfer"] is False
    assert PT.resolve_arm({"arm": "S0_P0"})["policy_transfer"] is False
    pa = PT.resolve_arm({"arm": "ST_PA"})
    assert pa["copy_actor"] and not pa["copy_critics"]
    pac = PT.resolve_arm({"arm": "ST_PAC"})
    assert pac["copy_actor"] and pac["copy_critics"]


def _main():
    fns = [(n, f) for n, f in sorted(globals().items())
           if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in fns:
        try:
            fn()
            print("PASS %s" % name)
        except Exception as exc:
            failed += 1
            print("FAIL %s: %s" % (name, exc))
    print("%d passed, %d failed" % (len(fns) - failed, failed))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_main())
