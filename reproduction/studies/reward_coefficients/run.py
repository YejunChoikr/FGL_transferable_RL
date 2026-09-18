"""Run one original Fig. 5 coefficient experiment, including the C2=0 baseline."""
import argparse
import os
from pathlib import Path
import sys

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
sys.path.insert(0, str(Path(__file__).resolve().parent/"code"))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", choices=["FGL5", "FGL6", "FGL7"], required=True)
    p.add_argument("--C1", type=float, default=.5)
    p.add_argument("--C2", type=float, default=0.)
    p.add_argument("--seed", type=int, choices=range(5), default=0)
    args = p.parse_args()
    if args.C2 < 0:
        p.error("C2 must be nonnegative")
    from run_reward import train_one
    stage = "stage1_C1" if args.C2 == 0 else "stage2_C2"
    train_one(stage, args.task, args.C1, args.C2, args.seed, "local")


if __name__ == "__main__":
    main()
