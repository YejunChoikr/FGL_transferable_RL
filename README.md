# Transferable reinforcement learning for targeted deformation shaping in functionally graded lattice structures

Yejun Choi, Yeoneung Kim, Keun Park
Seoul National University of Science and Technology

Code, finite element datasets, and models for learning continuous lattice
thickness assignments and transferring them across related design domains.

## Experiment reproduction

Start with **[the reproduction guide](reproduction/README.md)** for installation
and the training commands used for the paper's experiments.

```sh
python -m pip install -r reproduction/requirements.txt
python reproduction/run.py verify
python reproduction/run.py list --kind policy --domain upper --seed 0
python reproduction/run.py smoke --case policy/SAC/upper/shared/ST_PAC/s0
```

The reproduction package includes CNN/MLP surrogate training, nested data-budget
experiments, SAC and DDPG, actor/critic transfer, goal-conditioned policies,
coefficient sensitivity, and BO/L-BFGS-B/GA/DE. A common configuration controls
learning rates, location encoding, splits, and checkpoint selection. The frozen
case registry identifies all 590 experiments; the smoke command only tests the
installation with short runs.

Saved experiment logs, surrogate checkpoints, selected designs, and FEA
responses are provided in **[the evidence guide](evidence/README.md)**. Recompute
the manuscript statistics without retraining:

```sh
python reproduction/verify_paper.py
```

The [bilateral-objective study](reproduction/studies/bilateral/README.md)
contains the SAC/DDPG training and 200-evaluation BO paths used for S10,
including the four-layer DDPG critic transfer. The
[coefficient study](reproduction/studies/reward_coefficients/README.md)
includes the separate `C2=0`, `W=1` baseline used for Fig. 5.

| Directory | Contents |
| --- | --- |
| `reproduction/cmame_rt/` | Training, evaluation, and optimization code |
| `reproduction/spec/` | Numerical protocol, case registry, and reference math |
| `reproduction/data/` | Exact encoded arrays, fixed splits, and output scalers |
| `reproduction/tests/` | Regression and integration tests |
| `reproduction/studies/` | Coefficient and bilateral-objective experiments |
| `evidence/` | Saved run artifacts and their mapping to manuscript results |
| `src/`, `tools/`, `configs/` | Model utilities, training helpers, and configuration |
| `data/`, `models/`, `designs/` | Domain datasets, supplied checkpoints, and selected designs |

Saved surrogate checkpoints for the registered experiments are in
`evidence/surrogates.zip`. The [model and data guide](docs/model_usage.md)
describes the supplied files and their associated loading utilities.

## Citation

```bibtex
@article{choi_fgl_trl,
  title  = {Transferable reinforcement learning for targeted deformation
            shaping in functionally graded lattice structures},
  author = {Choi, Yejun and Kim, Yeoneung and Park, Keun},
  year   = {2026}
}
```

## License

Code MIT. Data and models CC-BY-4.0.
