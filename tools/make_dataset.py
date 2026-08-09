"""Convert a raw finite element sample set into the JSON format used here.

Two raw layouts are accepted. A directory of ``epi*.txt`` files holds one
sample per file: the nine probe displacements as ``xr1``--``xr9`` in
millimetres, followed by the 30 cell actions on [-1, 1]. An ``.npz`` archive
holds the same information in the arrays ``actions`` and ``u``.

Actions are mapped to thicknesses with the centre and half-range of the domain,
and the train/validation/test indices used in the paper are regenerated from
the frozen split.

    python tools/make_dataset.py --raw <dir-or-npz> --domain source \
        --out source_1.2_1.8.json
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import yaml
from sklearn.model_selection import train_test_split

SPLIT_RANDOM_STATE = 4
XR = re.compile(r"xr(\d+)\s*=\s*(-?[\d.]+(?:[eE][-+]?\d+)?)")


def read_txt_sample(path: Path):
    """Return the nine displacements and the 30 actions of one raw file."""
    text = path.read_text(encoding="utf-8", errors="replace")
    found = {int(i): float(v) for i, v in XR.findall(text)}
    missing = [i for i in range(1, 10) if i not in found]
    if missing:
        raise ValueError("%s is missing xr%s" % (path.name, missing))
    u = [found[i] for i in range(1, 10)]
    action = None
    for line in text.splitlines():
        parts = line.split()
        if len(parts) == 30:
            action = [float(p) for p in parts]
    if action is None:
        raise ValueError("no 30-value action line in %s" % path.name)
    if min(action) < -1.0 or max(action) > 1.0:
        raise ValueError("%s has actions outside [-1, 1]" % path.name)
    return u, action


def read_raw(raw: Path):
    """Return actions (n, 30) and displacements (n, 9) from either layout."""
    if raw.suffix == ".npz":
        d = np.load(raw)
        return np.asarray(d["actions"], dtype=float), np.asarray(d["u"], dtype=float)
    files = sorted(raw.glob("epi*.txt"),
                   key=lambda p: int(re.search(r"\d+", p.stem).group()))
    if not files:
        raise SystemExit("no epi*.txt under %s" % raw)
    u, a = [], []
    for f in files:
        ui, ai = read_txt_sample(f)
        u.append(ui)
        a.append(ai)
    return np.asarray(a), np.asarray(u)


def frozen_split(n):
    """The split used in the paper: 8:1:1 by two successive sklearn calls."""
    idx = np.arange(n)
    train_part, test = train_test_split(
        idx, test_size=0.1, random_state=SPLIT_RANDOM_STATE, shuffle=True)
    train, val = train_test_split(
        train_part, test_size=0.11111, random_state=SPLIT_RANDOM_STATE, shuffle=True)
    return train, val, test


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, type=Path)
    ap.add_argument("--domain", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--configs", default=Path(__file__).resolve().parents[1] / "configs",
                    type=Path)
    args = ap.parse_args()

    cfg = yaml.safe_load((args.configs / "env.yaml").read_text(encoding="utf-8"))
    dom = cfg["domains"][args.domain]
    tmin, tmax = dom["t_min"], dom["t_max"]
    centre, half = (tmin + tmax) / 2, (tmax - tmin) / 2

    actions, u = read_raw(args.raw)
    if actions.shape[1] != cfg["grid"]["n_independent"] or u.shape[1] != 9:
        raise SystemExit("unexpected shapes %s and %s" % (actions.shape, u.shape))

    t = np.round(centre + half * actions, 6)
    u = np.round(u, 6)
    train, val, test = frozen_split(len(t))

    payload = {
        "domain": args.domain,
        "thickness_range_mm": [tmin, tmax],
        "grid": {"rows": cfg["grid"]["rows"], "cols": cfg["grid"]["cols"],
                 "cell_mm": dom["cell_mm"]},
        "cell_order": "row-wise raster, left to right within each row",
        "units": {"thickness": "mm", "displacement": "mm"},
        "compression_ratio": cfg["compression_ratio"],
        "n": len(t),
        "split": {"ratio": [8, 1, 1], "random_state": SPLIT_RANDOM_STATE,
                  "train": train.tolist(), "val": val.tolist(), "test": test.tolist()},
        "t": t.tolist(),
        "u": u.tolist(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload), encoding="utf-8")
    print("%s  n=%d  train/val/test %d/%d/%d  %.1f MB"
          % (args.out, len(t), len(train), len(val), len(test),
             args.out.stat().st_size / 1e6))


if __name__ == "__main__":
    main()
