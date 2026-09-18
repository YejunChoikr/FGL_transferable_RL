# S10 bilateral-objective experiments

This package supplies the actual reward-training paths, not just an evaluation
function. The 105 registered cases comprise 30 source policies, 60 upper-domain
policies and 15 BO runs (three goals and five seeds).

From the repository root:

```sh
python reproduction/studies/bilateral/run.py list
python reproduction/studies/bilateral/run.py verify-init
python reproduction/studies/bilateral/run.py run --case upper/FGL5/SAC-TRL/s0 --with-dependencies
python reproduction/studies/bilateral/run.py run --case upper/FGL5/DDPG-TRL/s0 --with-dependencies
python reproduction/studies/bilateral/run.py run --case upper/FGL5/BO/s0
```

`--with-dependencies` trains the matching source policy first. SAC/DDPG require
CUDA. Outputs and source checkpoints go under `generated/`; they do not modify
the saved evidence. To test the neural execution paths with 25 episodes:

```sh
python reproduction/studies/bilateral/run.py smoke --case upper/FGL5/SAC-TRL/s0
python reproduction/studies/bilateral/run.py smoke --case upper/FGL5/DDPG-TRL/s0
```

Smoke outputs go under `pilot/` and are never used by the full-run commands.
Only run one process per case. Preserve existing incomplete outputs before
retrying. BO retains its full 200-call protocol; its automated integration test
checks objective wiring without fitting 180 Gaussian-process models.

## Preserved settings

The objective is `min(u_i-u_(i-1), u_i-u_(i+1)) / (0.5 + rho_A)` with
`rho_A=mean((action+1)/2)`. Source and upper surrogates/scalers are included under
`assets/`. Their original encoding is retained. These are separate from the
September common-encoding dataset and models.

| Path | Protocol |
| --- | --- |
| SAC source/scratch/transfer | Actor/critics/temperature LR 3e-4; batch 256; 1,500 training episodes after 20 replay-fill episodes |
| DDPG source/scratch | Actor LR 1e-4, critic LR 1e-3; batch 64; 1,500 training episodes after 20 replay-fill episodes |
| Final DDPG transfer (`ddpg_c4/`) | Native DDPG settings; actor and critic fc1–fc4 transferred; 1,500 total episodes including 20 replay-fill episodes |
| BO | `gp_minimize`, 200 total evaluations, 20 random initial points, `gp_hedge` |

The final DDPG transfer path uses the later C4 implementation that generated
the manuscript's values. The original C3 trainer remains in `code/run_ddpg.py`
because it also implements the source and scratch arms; the standalone runner
routes every upper DDPG transfer case to `ddpg_c4/`.

Initial scratch tensors are reconstructed and checked against the hashes
archived with the original source/target runs. Checkpoint serialization bytes
can differ while those tensors remain identical. The published configuration
omits machine scheduling and private paths; `SOURCE_MANIFEST.json` records
original and published hashes. New source checkpoints are trained by the
source commands; saved research learning curves, selections and FEA records
are in `evidence/bilateral.zip`.
