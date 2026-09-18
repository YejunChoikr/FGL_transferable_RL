"""Training-reward summaries used by the manuscript learning curves."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


def training_summary(rows, goals, early_episodes=300, window=50):
    """Compute unsmoothed early AUC and trailing means for each trained goal.

    The episode interval is shared by all goals. A goal-conditioned run has
    60 observations per goal in episodes 1--300; its trailing window contains
    50 observations of that goal. Incomplete early windows are not reported.
    """
    episodes = np.array([int(r["episode"]) for r in rows])
    goal_ids = np.array([int(r["goal"]) for r in rows])
    rewards = np.array([float(r["training_reward"]) for r in rows])
    if len(rows) == 0:
        raise ValueError("empty training log")
    # The training runtime logs zero-based episode indices. Report completed
    # episode counts (1..N), while also accepting already one-based exports.
    if episodes[0] == 0:
        episodes = episodes + 1
    if not np.array_equal(episodes, np.arange(1, len(rows) + 1)):
        raise ValueError("training rows must contain consecutive episodes starting at 0 or 1")
    if window <= 0 or early_episodes <= 0 or not goals or early_episodes % len(goals):
        raise ValueError("positive windows and an early interval divisible by the goal count are required")
    if not np.isfinite(rewards).all():
        raise ValueError("non-finite training reward")
    if not set(goal_ids).issubset(set(goals)):
        raise ValueError("training log contains an unregistered goal")
    result = {"source": "episode_log.csv:training_reward", "early_interval": [1, early_episodes],
              "smoothing_window": window, "smoothing_unit": "observations per goal",
              "early_auc_uses_unsmoothed_rewards": True, "goals": {}}
    for goal in goals:
        mask = goal_ids == goal
        values, e = rewards[mask], episodes[mask]
        early = values[e <= early_episodes]
        complete = episodes[-1] >= early_episodes
        expected = early_episodes // len(goals)
        if complete and len(early) != expected:
            raise ValueError(f"goal {goal}: expected {expected} early rewards, found {len(early)}")
        sums = np.r_[0., np.cumsum(values)]
        smooth = [(sums[i+1] - sums[max(0, i+1-window)]) / min(i+1, window)
                  for i in range(len(values))]
        result["goals"][str(goal)] = {
            "early_auc": float(early.mean()) if complete else None,
            "early_observation_count": len(early), "complete_early_window": bool(complete),
            "episodes": e.tolist(), "average_reward": smooth}
    return result


def summarize_run(directory, goals):
    directory = Path(directory)
    with (directory / "episode_log.csv").open(encoding="utf-8", newline="") as f:
        summary = training_summary(list(csv.DictReader(f)), goals)
    (directory / "training_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    return summary


def aggregate_curves(summaries, goal):
    """Mean and sample SD across runs, after smoothing each run separately."""
    if len(summaries) < 2:
        raise ValueError("sample standard deviation requires at least two runs")
    key = str(goal)
    series = [s["goals"][key] for s in summaries]
    # Goal-conditioned runs use different permutations within five-episode
    # blocks; align the kth observation of each goal, not its episode timestamp.
    lengths = {len(s["average_reward"]) for s in series}
    if len(lengths) != 1:
        raise ValueError("runs must have equal observation counts")
    y = np.array([s["average_reward"] for s in series])
    return {"observation": list(range(1, y.shape[1]+1)),
            "mean": y.mean(axis=0).tolist(), "sd": y.std(axis=0, ddof=1).tolist()}
