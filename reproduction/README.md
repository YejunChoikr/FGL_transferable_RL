# CMAME experiment reproduction

This package provides the training and optimization implementation used for the
paper's experiments. It runs independently of the multi-machine
scheduler. The case registry contains 590 experiments across five random seeds:
80 surrogate fits, 450 policy runs, and 60 direct-optimization runs.

## Installation

Run the commands below from the repository root in a Python 3.11 environment.
The experiments used Python 3.11.15 and PyTorch 2.11.0 with CUDA 12.8. Install
the matching PyTorch CUDA build for GPU execution, then the pinned dependencies:

```sh
python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r reproduction/requirements.txt
python reproduction/run.py verify
```

CPU execution is available with `--device cpu`. CUDA runs use fused Adam by
default; CPU runs use reference Adam. The backend can be specified explicitly
with `--adam-backend`. Device and backend are recorded with each result.
Numerical trajectories and wall-clock times can differ across hardware.

## Run experiments

Inspect the registered cases and prerequisites before starting a full run:

```sh
python reproduction/run.py list --kind policy --domain upper --seed 0
python reproduction/run.py plan --case policy/SAC/upper/g5/ST_PAC/N6000/s0
```

Train a source surrogate and fine-tune it using 6,000 upper-domain samples:

```sh
python reproduction/run.py run --case surrogate/cnn/upper/N6000/transfer/s0 --with-dependencies
```

Train target-specific and goal-conditioned policies:

```sh
python reproduction/run.py run --case policy/SAC/upper/g5/ST_P0/N6000/s0 --with-dependencies
python reproduction/run.py run --case policy/SAC/upper/g5/ST_PAC/N6000/s0 --with-dependencies
python reproduction/run.py run --case policy/SAC/upper/shared/ST_PAC/s0 --with-dependencies
python reproduction/run.py run --case policy/DDPG/upper/g5/ST_PAC/N6000/s0 --with-dependencies
```

`ST_P0` uses the transferred surrogate with a scratch policy. `ST_PAC` also
transfers the actor and critic hidden layers. `ST_PA` transfers only actor hidden
layers for the registered ablation. Each source/target pair uses the same seed.
The shared policy conditions on goals 3–7 and uses balanced goal schedules and
replay-goal relabeling.

Direct optimization uses the same transferred upper-domain surrogate:

```sh
python reproduction/run.py run --case optimizer/LBFGSB/upper/g5/s0 --with-dependencies
```

Other method identifiers are `BO`, `GA`, and `DE`. Their settings are in
[`spec/protocol.json`](spec/protocol.json). The supplied time allowance is
382.5083336830139 s, measured from the reference scratch-policy training runs
on an RTX 4080 SUPER. The runner applies that fixed allowance and records the
actual execution hardware; it does not recalibrate the allowance on another
GPU. Candidate selection excludes evaluations completed after the allowance.
Surrogate training is outside this optimization time allowance.

Full runs write to `reproduction/accepted/<case-id>/`. Dependencies are reused
only when their completion marker, code, protocol and artifact hashes agree.
An interrupted or incompatible output is not overwritten. Inspect and preserve
it before retrying in a clean directory or checkout. Run one process at a time
per checkout; this standalone entry point does not perform distributed locking.

For a short installation test:

```sh
python reproduction/run.py smoke --case policy/SAC/upper/shared/ST_PAC/s0
```

Smoke runs use one surrogate epoch, 25 policy episodes including 20 replay-fill
episodes, or a one-second optimizer allowance. They write only under
`reproduction/pilot/smoke/` and cannot satisfy full-run dependencies. They are
software tests, not research results.

## Data and experimental settings

The compressed arrays in `data/cache/` preserve the numerical inputs, record
order, and split assignments used for the experiments. Their hashes and the
training-only output scalers are in `locks/DATA_LOCK.json`. Source data contain
30,000 records; each target domain contains 6,000. Upper-domain budgets are
500, 1,000, 2,000, 4,000, and 6,000 **total** records, split 80/10/10 into
training/validation/test. Budget subsets are nested and retain a common
600-record target test set, disjoint from every training and validation set.
Thus, for example, N=6,000 uses 4,800 training records.

The surrogate and policy share the same cell encoding: normalized thickness
interleaved with `sqrt(r*r + c*c) / sqrt(R*R + C*C)`, where R=10, C=3,
r=1,…,10 and c=1,…,3. Unassigned cells have zero thickness and location entries.
The CNN reshapes the 60 entries to a 1×10×6 tensor. The fixed location vector
is supplied in `spec/location_golden.json`.

| Setting | Value |
| --- | --- |
| Surrogate epochs / batch size | 300 / 64 |
| Source and scratch surrogate LR | 5×10⁻⁴ |
| Transferred surrogate LR | 5×10⁻⁵ |
| Surrogate optimizer | Adam, weight decay 5×10⁻⁴ |
| Surrogate selection | Lowest validation MSE in mm²; earliest tie |
| SAC actor / critic / temperature LR | 3×10⁻⁴ / 3×10⁻⁴ / 3×10⁻⁴ |
| DDPG actor / critic LR | 1×10⁻⁴ / 1×10⁻³ |
| Policy episodes / steps | 1,500 / 30 |
| Uniform replay-fill episodes | First 20 of the 1,500; no gradient updates |
| Policy checkpoint evaluation | Episode 0 and every 25 episodes |
| Policy selection | Highest deterministic reward, or mean across five goals |

Surrogate transfer copies the convolutional and hidden fully connected weights
and BatchNorm affine parameters. It resets the output layer to the paired
scratch initialization and resets BatchNorm running statistics. All parameters
remain trainable. Both `best.pt` and `last.pt` are saved: policies use `best.pt`;
the final-epoch CNN/MLP comparison uses `last.pt` for both architectures.

The standalone runner evaluates both checkpoints explicitly and writes
`metrics_best.json`, `metrics_last.json`, and their corresponding common-test
predictions. `metrics.json` remains the original best-checkpoint evaluation.
To evaluate an existing run without training:

```sh
python reproduction/evaluate_surrogate.py --run-dir PATH_TO_RUN --checkpoint both --device cpu
```

An optional `--output-dir` keeps the saved run untouched. The evaluation records
the checkpoint hash, epoch, and completed training length. A short smoke run is
never labeled an epoch-300 result. Table S6/Fig. S5 use the ten source-domain
`last.pt` models, each trained for 300 epochs; policy environments use `best.pt`.

Policy transfer copies the selected source actor/critic hidden layers from the
same evaluation checkpoint, retaining the paired scratch output heads. Target
critics are synchronized; optimizers, replay memory, and SAC temperature are
initialized afresh. SAC and DDPG learning rates do not change on transfer.

## Outputs and metrics

Surrogate runs save checkpoints, scalers, learning-rate logs, validation/test
metrics, and common-test predictions. Policy runs save selected and final
actor/critic checkpoints, episode logs, deterministic evaluation records,
selected designs, and timing/configuration metadata.

The current manuscript's Average reward curves and early area under the curve
(Early AUC) use **training rewards** from `episode_log.csv`:

- `training_summary.json` gives trailing 50-observation means for plotting and
  the mean **unsmoothed** training reward in episodes 1–300 for normalized
  Early AUC. Each goal-conditioned policy contributes 60 observations per goal
  within those same 300 episodes. Incomplete early windows are reported as null.
  The CSV logs zero-based episode indices (0–1,499); summary axes use completed
  episode counts (1–1,500).
- `cmame_rt.reporting.aggregate_curves` reports the mean and sample standard
  deviation across independently trained policies after smoothing each run.
- `deterministic_auc.json` retains the separate trapezoidal integral of the
  deterministic evaluations at episodes 0,25,…,300. This is a diagnostic
  produced by the original training protocol, not the manuscript's
  training-reward Early AUC. The `evaluation.early_auc` entry in the frozen
  numerical protocol describes this diagnostic.

Selected-design displacements and rewards produced here are surrogate
predictions. Direct FEA and physical testing are separate validation steps;
these commands do not invoke Ansys or generate experimental measurements.

## Verification and provenance

```sh
python -m pip install -r reproduction/requirements-test.txt
python -m pytest reproduction/tests -q
```

Tests cover data splits and scalers, shared encoding, transfer initialization,
SAC/DDPG updates, goal relabeling, evaluation RNG isolation, optimizer settings,
and training-reward summaries. `SOURCE_MANIFEST.json` records the source and
published file hashes. The training algorithms are preserved from
the experiment implementation. The published reward implementation corrects
the separate `C2=0` branch to use `W=1`, matching the manuscript and original
coefficient experiment. Positive `C2` values are unchanged; the 590-case main
registry contains no `C2=0` runs. The correction updates the published protocol
hash but does not relabel hashes or results of saved experiments.
Packaging adds the standalone runner and
training-reward summaries, renames the deterministic-AUC output, and removes
machine scheduling and private-path metadata from the published specification.

This runner uses the checkpoints and scalers in each experiment's output
directory. Supplied models used for Table S9 are loaded through `src.env`, as
described in the [model and data guide](../docs/model_usage.md). Each model
requires its associated architecture, scaler and input encoding. Newly trained
checkpoints are excluded from Git.

Saved experiment outputs are provided separately in [../evidence/](../evidence/README.md).
The verifier reaggregates MAE, peak counts, the two S10 objectives, and Table S9
predictions from these artifacts. See the separate guides for the
[bilateral study](studies/bilateral/README.md) and
[coefficient study](studies/reward_coefficients/README.md). Their experiment-specific
surrogates, encoding, training schedules, and transfer implementations are
preserved; their results are not attributed to the September protocol.
