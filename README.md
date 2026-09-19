# Surrogate and policy transfer for targeted deformation shaping of graded lattices

Yejun Choi, Yeoneung Kim, Keun Park
Seoul National University of Science and Technology

Code and finite element datasets for learning continuous lattice thickness
assignments and transferring them across related design domains.

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
coefficient sensitivity, and BO/L-BFGS-B/GA/DE. The main runner uses
`reproduction/spec/protocol.json` for learning rates, location encoding, splits,
and checkpoint selection. The case registry specifies the experiments and
coefficient sweeps. The smoke command tests the installation with short runs.

The [bilateral-objective study](reproduction/studies/bilateral/README.md)
contains the SAC/DDPG training and 200-evaluation BO paths used for S10,
including four-layer actor and critic transfer.
[Coefficient sweeps](reproduction/README.md#reward-coefficient-sweeps) use the
same SAC, source surrogate, encoding and training settings as the main runner.

| Directory | Contents |
| --- | --- |
| `reproduction/cmame_rt/` | Training, evaluation, and optimization code |
| `reproduction/spec/` | Numerical protocol, case registry, and reference math |
| `reproduction/data/` | Design inputs, FEA displacements, fixed splits, and output scalers |
| `reproduction/tests/` | Regression and integration tests |
| `reproduction/studies/` | Bilateral-objective code and its required surrogate/scaler pairs |

The data comprise 30,000 source-domain records and 6,000 records for each of
the upper-bound, lower-bound, and aspect-ratio domains. Each record contains
design inputs and nine FEA displacement components. The reproduction guide
defines the data-budget subsets, output normalization, and evaluation metrics.
Training commands generate their own logs, checkpoints, and selected designs.

## Citation

```bibtex
@article{choi_fgl_trl,
  title  = {Surrogate and policy transfer for targeted deformation shaping
            of graded lattices},
  author = {Choi, Yejun and Kim, Yeoneung and Park, Keun},
  year   = {2026}
}
```

## License

Code MIT. Data and models CC-BY-4.0.
