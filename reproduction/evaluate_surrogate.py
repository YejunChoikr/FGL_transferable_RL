"""Evaluate saved surrogate checkpoints without retraining."""
import argparse
import json
import os

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
from cmame_rt.checkpoint_evaluation import evaluate_checkpoint
from cmame_rt.determinism import apply_runtime_flags


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--checkpoint", choices=["best", "last", "both"], default="both")
    p.add_argument("--output-dir")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    apply_runtime_flags()
    for which in (["best", "last"] if args.checkpoint == "both" else [args.checkpoint]):
        print(json.dumps(evaluate_checkpoint(args.run_dir, which, args.output_dir, args.device), indent=2))


if __name__ == "__main__":
    main()
