"""Train a surrogate for one design domain, or fine-tune one from another.

300 epochs, batch 64, mean squared error on standardized displacements, Adam at
5e-4 with weight decay 5e-4, and the learning rate halved after 10 epochs
without validation improvement. Fine-tuning uses 5e-5. The partitions are those
stored in the data file.

    python tools/train_surrogate.py --domain upper
    python tools/train_surrogate.py --domain upper --init models/surrogate_source.pth
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from env import SurrogateCNN, location_grid              # noqa: E402

EPOCHS = 300
BATCH = 64
LR = 5e-4
LR_FINETUNE = 5e-5
WEIGHT_DECAY = 5e-4
PLATEAU_FACTOR = 0.5
PLATEAU_PATIENCE = 10

COL_STEP = {"source": 2, "upper": 2, "lower": 2, "ar08": 1}


def encode(thickness_mm, t_min, t_max, rows, cols, col_step):
    """Interleave the normalized thickness of each cell with its location."""
    a = (2.0 * (np.asarray(thickness_mm) - t_min) / (t_max - t_min)) - 1.0
    loc = location_grid(rows, cols, col_step)
    a = a.reshape(-1, rows, cols)
    loc = np.broadcast_to(loc, a.shape)
    return np.stack((a, loc), axis=3).reshape(len(a), -1).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", required=True)
    ap.add_argument("--init", type=Path, default=None,
                    help="source surrogate to fine-tune from")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cfg = yaml.safe_load((ROOT / "configs" / "env.yaml").read_text(encoding="utf-8"))
    dom = cfg["domains"][args.domain]
    rows, cols = cfg["grid"]["rows"], cfg["grid"]["cols"]
    data = json.loads((ROOT / dom["data"]).read_text(encoding="utf-8"))

    x = encode(np.asarray(data["t"]), dom["t_min"], dom["t_max"], rows, cols,
               COL_STEP[args.domain])
    y = np.asarray(data["u"], dtype=np.float64)
    idx = data["split"]

    mean = y[idx["train"]].mean(axis=0)
    scale = y[idx["train"]].std(axis=0)
    ys = (y - mean) / scale

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tensors = {k: (torch.tensor(x[idx[k]], device=device),
                   torch.tensor(ys[idx[k]], dtype=torch.float32, device=device))
               for k in ("train", "val", "test")}

    model = SurrogateCNN(y.shape[1], cfg["env"]["leakyrelu_para"]).to(device)
    lr = LR
    if args.init is not None:
        model.load_state_dict(torch.load(args.init, map_location=device, weights_only=True))
        lr = LR_FINETUNE
        print("fine-tuning from %s at lr %g" % (args.init, lr))

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=PLATEAU_FACTOR, patience=PLATEAU_PATIENCE)
    loss_fn = nn.MSELoss()

    xt, yt = tensors["train"]
    xv, yv = tensors["val"]
    history = {"train": [], "val": []}

    for epoch in range(args.epochs):
        model.train()
        order = torch.randperm(len(xt), device=device)
        total = 0.0
        for i in range(0, len(order), BATCH):
            b = order[i:i + BATCH]
            opt.zero_grad()
            loss = loss_fn(model(xt[b]), yt[b])
            loss.backward()
            opt.step()
            total += float(loss.item()) * len(b)
        model.eval()
        with torch.no_grad():
            pv = model(xv)
            vl = float(loss_fn(pv, yv).item())
        sched.step(vl)
        history["train"].append(total / len(order))
        history["val"].append(vl)
        if (epoch + 1) % 50 == 0:
            print("epoch %d/%d  train %.6f  val %.6f"
                  % (epoch + 1, args.epochs, history["train"][-1], vl), flush=True)

    out = args.out or (ROOT / "runs" / ("surrogate_%s" % args.domain))
    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / "surrogate.pth")
    (out / "scaler.json").write_text(json.dumps(
        {"domain": args.domain, "type": "StandardScaler",
         "mean": mean.tolist(), "scale": scale.tolist(),
         "note": "u_mm = prediction * scale + mean, per probe u1..u9"},
        indent=2), encoding="utf-8")

    model.eval()
    xs, ys_t = tensors["test"]
    with torch.no_grad():
        pred = model(xs).cpu().numpy()
    mae = np.abs(pred * scale + mean - y[idx["test"]]).mean()
    (out / "history.json").write_text(json.dumps(
        {"epochs": args.epochs, "test_mae_mm": float(mae), **history}, indent=2),
        encoding="utf-8")
    print("test mean absolute error %.4e mm" % mae)
    print("written to %s" % out)


if __name__ == "__main__":
    main()
