# S10 bilateral-objective experiments

This study defines 105 cases: 30 source policies, 60 upper-domain policies,
and 15 Bayesian-optimization runs for FGL 5, 6, and 7 over five seeds. It uses
the same `cmame_rt` policy, transfer, evaluation, and direct-optimization code
as the main experiment. The only changed quantity is the training and selection
objective:

```text
B = min(u_g - u_(g-1), u_g - u_(g+1)) / (0.5 + mean((a + 1) / 2))
```

Canonical Eq. (4) is still recorded separately for reporting.

From the repository root:

```sh
python reproduction/studies/bilateral/run.py list
python reproduction/studies/bilateral/run.py plan --case upper/FGL5/SAC-TRL/s0
python reproduction/studies/bilateral/run.py smoke --case upper/FGL5/SAC-TRL/s0
python reproduction/studies/bilateral/run.py run --case upper/FGL5/SAC-TRL/s0 --with-dependencies
```

`--with-dependencies` uses accepted main-experiment surrogate artifacts when
present, or generates them through the main registry. Each seed uses its paired
source N=30,000 CNN/scaler and transferred upper N=6,000 CNN/scaler; fixed policy
weights and separate S10 surrogates are not distributed.

SAC and DDPG use the main hyperparameters, initialization bank, input encoding,
and fused Adam CUDA path. Every policy run has 1,500 total episodes, including
20 uniform replay-prefill episodes, with deterministic evaluation at episode 0
and every 25 episodes through 1,500. The earliest checkpoint with the highest
bilateral evaluation reward is selected. Transfer copies actor and critic
hidden layers `fc1` through `fc4` from the matching goal, algorithm, and seed.
Output heads retain their paired scratch values; target networks are hard-copied
after transfer, while optimizers, replay, and random streams start fresh.

Full BO cases use the main BO operators and paired upper surrogate for exactly
382.5083336830139 seconds. They must run sequentially on an RTX 4080 SUPER while
the workstation is otherwise idle:

```sh
python reproduction/studies/bilateral/run.py run --case upper/FGL5/BO/s0 --with-dependencies --exclusive
```

Generated artifacts are ignored by Git. Selected responses are surrogate
predictions; the runner does not invoke Ansys or create recovered FEA values.
