# S10 bilateral-objective experiments

The 105 cases comprise 30 source policies, 60 upper-domain policies and 15 BO
runs: three prescribed goals and five seeds. The objective is
`min(u_i-u_(i-1),u_i-u_(i+1))/(0.5+rho_A)`, with `rho_A=mean((action+1)/2)`.

From the repository root:

```sh
python reproduction/studies/bilateral/run.py list
python reproduction/studies/bilateral/run.py verify-init
python reproduction/studies/bilateral/run.py run --case upper/FGL5/SAC-TRL/s0 --with-dependencies
python reproduction/studies/bilateral/run.py run --case upper/FGL5/DDPG-TRL/s0 --with-dependencies
python reproduction/studies/bilateral/run.py run --case upper/FGL5/BO/s0
```

`--with-dependencies` trains the matching source policy first. SAC/DDPG require
CUDA. Run outputs and source checkpoints go under `generated/`. Smoke commands
use 25 episodes and write under `pilot/`:

```sh
python reproduction/studies/bilateral/run.py smoke --case upper/FGL5/SAC-TRL/s0
python reproduction/studies/bilateral/run.py smoke --case upper/FGL5/DDPG-TRL/s0
```

Only run one process per case. Preserve incomplete outputs before retrying.
BO retains its 200-call protocol; the integration test checks its objective
and evaluation count without a complete Gaussian-process optimization.

## Settings

The fixed source and upper-domain surrogate/scaler pairs are in `assets/`.
Their input encoding is implemented in `code/env60.py` and
`ddpg_c4/code/env60.py` for the corresponding experiment paths.

| Path | Settings |
| --- | --- |
| SAC source/scratch/transfer | Actor, critics and temperature LR 3e-4; batch 256; 1,500 training episodes after 20 replay-fill episodes |
| DDPG source/scratch | Actor LR 1e-4, critic LR 1e-3; batch 64; 1,500 training episodes after 20 replay-fill episodes |
| DDPG transfer | Actor LR 1e-4, critic LR 1e-3; batch 64; 1,500 total episodes including 20 replay-fill episodes |
| BO | `gp_minimize`, 200 total evaluations, 20 random initial points, `gp_hedge` |

Transfer copies actor and critic hidden layers fc1-fc4. Output heads, optimizer
states, replay memory and exploration states are initialized afresh. The DDPG
transfer implementation is in `ddpg_c4/`; the other paths are in `code/`.
Initial source/scratch tensors are checked against specified hashes.

Saved learning curves, selected designs and FEA responses are in
`evidence/bilateral.zip`. Arithmetic-objective S10 columns are evaluated using
the main policy/optimizer results. The two sets retain their specified
surrogates and episode/evaluation budgets.
