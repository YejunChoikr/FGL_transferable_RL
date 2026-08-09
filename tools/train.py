"""Train one policy on one task in one design domain.

A 20-episode replay fill, then 1500 episodes of 30 steps with one update per
transition. The deterministic policy is evaluated at episode 0 and every 25
episodes; the design retained is the evaluation with the highest reward.

    python tools/train.py --domain upper --task FGL7 --agent sac --seed 0
    python tools/train.py --domain upper --task FGL7 --agent sac --seed 0 \
        --init models/policy_source_FGL7
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from env import make_env, set_seed                      # noqa: E402
from reward import full_metrics                         # noqa: E402
import transfer as transfer_mod                         # noqa: E402


def load_configs(agent_name):
    cfg = yaml.safe_load((ROOT / "configs" / "env.yaml").read_text(encoding="utf-8"))
    cfg.update(yaml.safe_load(
        (ROOT / "configs" / ("%s.yaml" % agent_name)).read_text(encoding="utf-8")))
    return cfg


def build_agent(agent_name, cfg, device, seed):
    if agent_name == "sac":
        from sac import SACAgent
        return SACAgent(cfg["env"]["state_dim"], device, cfg)
    from ddpg import DDPGAgent
    d, t = cfg["ddpg"], cfg["train"]
    return DDPGAgent(cfg["env"]["state_dim"], device,
                     actor_lr=d["actor_lr"], critic_lr=d["critic_lr"],
                     weight_decay=d["weight_decay"], gamma=t["gamma"], tau=t["tau"],
                     batch_size=t["batch_size"], memory_capacity=t["memory_capacity"],
                     ou=(d["ou"]["mu"], d["ou"]["theta"], d["ou"]["sigma"]),
                     noise_rng=np.random.default_rng(seed))


class Driver:
    """One interface over the two agents, so the loop below reads the same."""

    def __init__(self, name, agent, batch_size):
        self.name, self.agent, self.bs = name, agent, batch_size

    def act(self, s, deterministic=False):
        if self.name == "sac":
            a = (self.agent.deterministic_action(s) if deterministic
                 else self.agent.select_action(s))
        else:
            a = self.agent.act(s, noise=not deterministic)
        return np.asarray(a, dtype=np.float64).reshape(-1)

    def store(self, s, a, ns, r, done):
        if self.name == "sac":
            self.agent.replay.store(s, a, ns, r, done)
        else:
            self.agent.memory.push(s, a, r, ns, done)

    def update(self):
        return self.agent.update(self.bs) if self.name == "sac" else self.agent.update()

    def start_episode(self):
        if self.name == "ddpg":
            self.agent.noise.reset()

    def actor_state(self):
        return {k: v.detach().cpu().clone() for k, v in self.agent.actor.state_dict().items()}

    def save(self, out: Path):
        out.mkdir(parents=True, exist_ok=True)
        torch.save(self.agent.actor.state_dict(), out / "actor.pth")
        if self.name == "sac":
            torch.save(self.agent.q1.state_dict(), out / "q1.pth")
            torch.save(self.agent.q2.state_dict(), out / "q2.pth")
        else:
            torch.save(self.agent.critic.state_dict(), out / "critic.pth")


def rollout(env, driver, task, deterministic=True):
    """One complete design under the current policy."""
    s = env.reset()
    done, terminal_reward = False, float("nan")
    while not done:
        a = driver.act(s, deterministic)
        s, r, done = env.replay_step(np.array([a[0]], dtype=np.float64))
        if done:
            terminal_reward = float(r)
    u, actions = env.surrogate_profile(), env.actions_row_major()
    m = full_metrics(u, actions, task)
    m.update({"reward_env": terminal_reward, "u": u.tolist(),
              "actions": actions.tolist()})
    env.reset()
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True)
    ap.add_argument("--task", required=True, choices=["FGL5", "FGL6", "FGL7"])
    ap.add_argument("--agent", default="sac", choices=["sac", "ddpg"])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--init", type=Path, default=None,
                    help="directory of a source agent to initialize from")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--episodes", type=int, default=None,
                    help="overrides the configured episode count")
    args = ap.parse_args()

    cfg = load_configs(args.agent)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    episodes = args.episodes or int(cfg["train"]["episodes"])
    fill = int(cfg["train"]["memory_fill_episodes"])
    every = int(cfg["train"]["deterministic_eval_interval"])
    out = args.out or (ROOT / "runs" /
                       ("%s_%s_%s_seed%d" % (args.domain, args.task, args.agent, args.seed)))

    set_seed(args.seed)
    env = make_env(args.task, args.domain, cfg, ROOT, device)
    agent = build_agent(args.agent, cfg, device, args.seed)
    driver = Driver(args.agent, agent, int(cfg["train"]["batch_size"]))

    if args.init is not None:
        state = {name: torch.load(args.init / ("%s.pth" % name), map_location=device, weights_only=True)
                 for name in (("actor", "q1", "q2") if args.agent == "sac"
                              else ("actor", "critic"))}
        fn = transfer_mod.transfer_sac if args.agent == "sac" else transfer_mod.transfer_ddpg
        moved = fn(agent, state)
        print("initialized %d tensors from %s" % (len(moved), args.init))

    records = [dict(rollout(env, driver, args.task, True), episode=0)]
    best = dict(records[0])
    driver.save(out / "best")

    for _ in range(fill):
        s = env.reset()
        driver.start_episode()
        done = False
        while not done:
            a = driver.act(s)
            ns, r, done = env.replay_step(np.array([a[0]], dtype=np.float64))
            driver.store(s, a, ns, r, done)
            s = ns

    history = []
    for ep in range(episodes):
        s = env.reset()
        driver.start_episode()
        done, terminal_reward = False, 0.0
        while not done:
            a = driver.act(s)
            ns, r, done = env.replay_step(np.array([a[0]], dtype=np.float64))
            driver.store(s, a, ns, r, done)
            driver.update()
            if done:
                terminal_reward = float(r)
            s = ns
        history.append(terminal_reward)

        if (ep + 1) % every == 0:
            d = dict(rollout(env, driver, args.task, True), episode=ep + 1)
            records.append(d)
            if d["reward_env"] > best["reward_env"]:
                best = dict(d)
                driver.save(out / "best")
        if (ep + 1) % 250 == 0:
            print("episode %d/%d  reward %.5f  best %.5f"
                  % (ep + 1, episodes, terminal_reward, best["reward_env"]), flush=True)

    out.mkdir(parents=True, exist_ok=True)
    driver.save(out / "final")
    np.save(out / "reward_history.npy", np.asarray(history, dtype=np.float64))
    (out / "deterministic_evaluations.json").write_text(
        json.dumps({"domain": args.domain, "task": args.task, "agent": args.agent,
                    "seed": args.seed, "episodes": episodes,
                    "evaluation_interval": every, "records": records,
                    "selected": best}, indent=2), encoding="utf-8")
    print("selected episode %d, surrogate reward %.5f, bilateral margin %.4f mm"
          % (best["episode"], best["reward_env"], best["m_bilateral"]))
    print("written to %s" % out)


if __name__ == "__main__":
    main()
