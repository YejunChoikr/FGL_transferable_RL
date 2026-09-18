"""Execute the final S10 DDPG transfer implementation."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "code"))
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", choices=["FGL5", "FGL6", "FGL7"], required=True)
    p.add_argument("--seed", type=int, choices=range(5), required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    import torch
    torch.set_num_threads(1)
    from common_c4 import verify_assets
    from run_one import execute
    verify_assets()
    spec = dict(objective="bilateral", domain="upper", task=args.task,
                seed=args.seed, method="TRL_DDPG_C4", source_bank="bilateral",
                run_key=f"BILAT__upper__{args.task}__seed{args.seed}")
    execute(spec, args.output, "cuda", max_episodes=25 if args.smoke else None)
    (args.output / "execution_mode.json").write_text(json.dumps({
        "smoke": args.smoke, "episodes_including_prefill":25 if args.smoke else 1500,
        "prefill_episodes":20}), encoding="utf-8")


if __name__ == "__main__":
    main()
