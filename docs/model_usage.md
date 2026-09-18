# Models and data

This guide describes the supplied datasets, models and selected designs. Paths
are relative to the repository root. Use [the reproduction guide](../reproduction/README.md)
for installation and the registered training and optimization experiments.

## Contents

`data/` — finite element datasets for the four design domains. Each file gives
the cell thicknesses in millimetres, the nine probe displacements, and the
train, validation and test indices.

| file | domain | thickness range | unit cell | samples |
| --- | --- | --- | --- | --- |
| `source_1.2_1.8.json` | source | 1.2–1.8 mm | 10 × 10 mm | 30,000 |
| `upper_1.2_2.4.json` | upper-bound expanded domain | 1.2–2.4 mm | 10 × 10 mm | 6,000 |
| `lower_0.9_1.8.json` | lower-bound expanded domain | 0.9–1.8 mm | 10 × 10 mm | 6,000 |
| `ar08_1.2_1.8.json` | aspect ratio 0.8 | 1.2–1.8 mm | 8 × 10 mm | 6,000 |

`models/` — the convolutional surrogate for each domain with its output scaler,
and the source-domain actor and critics for the three target locations.

`src/` — the environment and the surrogate network (`env.py`), the reward and
the margin quantities (`reward.py`), the SAC and DDPG agents (`sac.py`,
`ddpg.py`), and the initialization of a target-domain agent from a source-domain
agent (`transfer.py`).

`configs/` — environment and domain definitions, and the settings of every
optimizer.

`designs/designs.json` — thickness field and finite element displacements of the
fourteen optimized designs shown in the paper.

`tools/` — `train.py`, `train_surrogate.py`, and `make_dataset.py`, which builds
a data file from a raw finite element sample set.

## Model loading

Use each checkpoint with its associated architecture, input encoding and output
scaler. The Table S9 verification loads `models/surrogate_upper.pth` with
`src.env.SurrogateCNN`, `src.env.location_grid` and `models/scaler_upper.json`.
Its exact design inputs and model hashes are recorded in `evidence/table_s9.json`.

The checkpoints in `evidence/surrogates.zip` use `reproduction/cmame_rt/` and the
scaler saved alongside each checkpoint. After extracting the archive, evaluate
a case using:

```sh
python reproduction/evaluate_surrogate.py --run-dir PATH_TO_EXTRACTED_CASE --checkpoint both
```

The [evidence guide](../evidence/README.md) maps the models and saved results to
the manuscript's tables and figures.

## Conventions

Only the half domain is parameterized: 30 cells in 10 rows of 3, visited in a
row-wise raster from the lower left. Actions lie on [-1, 1] and map linearly onto
the admissible thickness range. Displacements are the horizontal displacements at
the nine probe points under 15 % compression, in millimetres.

The surrogate takes a single-channel 10 × 6 tensor in which each cell contributes
its normalized thickness followed by a radial location, and returns standardized
displacements. To recover millimetres, multiply by `scale` and add `mean` from
the matching `scaler_*.json`, probe by probe.

Transfer copies the four hidden layers of the actor and of each critic, 24
tensors for SAC and 16 for DDPG.
