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
case registry includes the manuscript experiments and coefficient sweeps. The
smoke command tests the installation with short runs.

Saved experiment logs, surrogate checkpoints, selected designs, and FEA
responses are provided in **[the evidence guide](evidence/README.md)**. Recompute
the manuscript statistics without retraining:

```sh
python reproduction/verify_paper.py
```

The [bilateral-objective study](reproduction/studies/bilateral/README.md)
contains the SAC/DDPG training and 200-evaluation BO paths used for S10,
including four-layer actor and critic transfer.
[Coefficient sweeps](reproduction/README.md#reward-coefficient-sweeps) use the
same SAC, source surrogate, encoding and training settings as the main runner.

| Directory | Contents |
| --- | --- |
| `reproduction/cmame_rt/` | Training, evaluation, and optimization code |
| `reproduction/spec/` | Numerical protocol, case registry, and reference math |
| `reproduction/data/` | Exact encoded arrays, fixed splits, and output scalers |
| `reproduction/tests/` | Regression and integration tests |
| `reproduction/studies/` | Bilateral-objective experiments |
| `evidence/` | Saved run artifacts and their mapping to manuscript results |

Saved surrogate checkpoints for the registered experiments are in
`evidence/surrogates.zip`. Models required for Table S9 are in
`evidence/validation_models/`; displayed designs are recorded in
`evidence/representative_designs.json`.

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
