"""Explicit best/last checkpoint evaluation on the frozen common test set."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch

from .data import budget_split, load_domain
from .protocol import file_sha256, surrogate_cfg


def evaluate_checkpoint(run_dir, checkpoint="last", output_dir=None, device="cpu"):
    from .surrogate_train import load_surrogate_for_env, _predict_mm, _regression_metrics
    if checkpoint not in ("best", "last"):
        raise ValueError("checkpoint must be best or last")
    run_dir = Path(run_dir)
    output_dir = Path(output_dir) if output_dir is not None else run_dir
    case = json.loads((run_dir / "resolved_config.json").read_text(encoding="utf-8"))["case"]
    log = list(csv.DictReader((run_dir / "train_log.csv").open(encoding="utf-8", newline="")))
    if not log:
        raise ValueError("empty training history")
    completed_epochs = int(log[-1]["epoch"])
    # The trainer logs completed epochs, starting at one.
    if [int(r["epoch"]) for r in log] != list(range(1, completed_epochs+1)):
        raise ValueError("incomplete or nonconsecutive epoch history")
    epoch = (int(json.loads((run_dir/"metrics.json").read_text(encoding="utf-8"))["best_epoch"])
             if checkpoint == "best" else completed_epochs)
    path = run_dir / f"{checkpoint}.pt"
    model, scaler = load_surrogate_for_env(path, device, arch=case["architecture"])
    data = load_domain(case["domain"])
    idx = budget_split(case["domain"], case["n_total"])["common_test_idx"]
    x = torch.as_tensor(data.X60[idx], dtype=torch.float32, device=device)
    pred = _predict_mm(model, x, scaler["mean"], scaler["scale"],
                       surrogate_cfg()["test_batch_size"]).cpu().numpy()
    # Match the training evaluator's physical-unit float32 ground truth.
    true = data.U9[idx].astype(np.float32)
    result = {"case_id": case["case_id"], "checkpoint": checkpoint,
              "checkpoint_sha256": file_sha256(path), "epoch": epoch,
              "completed_epochs": completed_epochs,
              "is_epoch_300": epoch == 300 and completed_epochs == 300,
              "scaler_sha256": file_sha256(run_dir/"scaler.json"),
              "common_test": _regression_metrics(pred, true)}
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_dir/f"predictions_common_test_{checkpoint}.npz",
                        pred_mm=pred.astype(np.float64), true_mm=true.astype(np.float64),
                        record_id=data.record_id[idx])
    (output_dir/f"metrics_{checkpoint}.json").write_text(
        json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    return result
