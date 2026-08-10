# Transferable reinforcement learning for targeted deformation shaping in functionally graded lattice structures

Yejun Choi, Yeoneung Kim, Keun Park
Seoul National University of Science and Technology

Data, trained models and training code for the above paper.

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

## Installation

    pip install -r requirements.txt

## Training a surrogate

    python tools/train_surrogate.py --domain source
    python tools/train_surrogate.py --domain upper --init models/surrogate_source.pth

300 epochs, batch 64, Adam at 5e-4 with weight decay 5e-4, and the learning rate
halved after 10 epochs without validation improvement. Fine-tuning uses 5e-5.

## Training a policy

    python tools/train.py --domain source --task FGL7 --agent sac --seed 0
    python tools/train.py --domain upper --task FGL7 --agent sac --seed 0 \
        --init models/policy_source_FGL7

1500 episodes of 30 steps, preceded by a 20-episode replay fill, with the
deterministic policy evaluated every 25 episodes. Each run writes its reward
history, the evaluation record and the selected design to
`runs/<domain>_<task>_<agent>_seed<n>/`.

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

## Citation

    @article{choi_fgl_trl,
      title  = {Transferable reinforcement learning for targeted deformation
                shaping in functionally graded lattice structures},
      author = {Choi, Yejun and Kim, Yeoneung and Park, Keun},
      year   = {2026}
    }

## License

Code MIT. Data and models CC-BY-4.0.
