# Original reward-coefficient study

This is the original coefficient-training implementation used for Fig. 5.
It uses its own source surrogate and calibration settings, preserved under
`assets/` and `configs/`. It is separate from the September joint sensitivity
cases in the main registry.

```sh
python reproduction/studies/reward_coefficients/run.py --task FGL5 --C1 0.5 --C2 0 --seed 0
python reproduction/studies/reward_coefficients/run.py --task FGL5 --C1 0.5 --C2 1 --seed 0
```

Run from the repository root with the reproduction dependencies and CUDA.
Outputs are written under this study's `generated/` directory. The numerator
is `u_i-C1*(u_(i-1)+u_(i+1))`. The denominator is **1 when C2=0** and
`0.5+C2*mean((action+1)/2)` otherwise. Substituting zero in the second expression
would incorrectly double the no-penalty baseline reward.

The original C1 grid is 0.1, 0.2, 0.3, 0.4, 0.5 at C2=0. The C2 grid is
0, 0.25, 0.5, 0.75, 1 at C1=0.5; its zero case reuses the first stage.
Each cell has three goals and five seeds. The saved histories and selected
designs are in `evidence/coefficients.zip`. Source and published file hashes
are recorded in `SOURCE_MANIFEST.json`.
