# Saved experiment evidence

These are saved experimental artifacts, not new runs made during code
publication. `INDEX.json` contains archive hashes and the mapping to manuscript
results. Each archive has a `MANIFEST.json` with published member hashes and,
for copied files, their original hashes. Private absolute paths are made
portable; numerical arrays and checkpoint tensors are retained.

| Artifact | Contents and use |
| --- | --- |
| `surrogates.zip` | 80 runs: training logs, configuration, best checkpoints, common-test predictions; ten source CNN/MLP runs also contain epoch-300 last checkpoints and predictions |
| `policies.zip` | 450 runs: episode rewards, deterministic evaluations, checkpoint selection, selected designs, configuration |
| `direct.zip` | 60 runs: first 1,500 evaluations for Fig. 11, full-time-allowance selected incumbent, counts and timing |
| `fea.zip` | Saved FEA displacement profiles and links for 700 requests, identified by case, goal and input hash |
| `bilateral.zip` | S10 source/target learning logs, selected designs, original FEA records, and the final 75 upper-domain cases |
| `coefficients.zip` | Original coefficient-study histories, selected designs, and analysis tables including the `C2=0` baseline |
| `table_s9.json` | Three fabricated-design inputs, original model/scaler hashes, predictions and FEA profiles |

## Recompute the reported quantities

From the repository root, with the reproduction dependencies installed:

```sh
python reproduction/verify_paper.py
```

The command checks archive and member hashes, reaggregates the metrics, verifies
policy/design/FEA connections, and evaluates the original model on the exact
Table S9 inputs. Results go to `reproduction/evidence_checks/`:

- `verified_metrics.json`: MAE, sample SD, peak counts, and S10 comparisons.
- `selected_design_fea_metrics.csv`: case-level metrics from selected-design FEA.

Reaggregation gives the following values:

| Quantity | Result |
| --- | --- |
| Surrogate MAE reduction, upper / lower / aspect-ratio domain | 35.3% / 29.8% / 36.4% |
| Shared transferred SAC policy, FEA peak success | 25/25 upper; 21/25 lower; 15/25 aspect-ratio domain |
| Source CNN, epoch-300 MAE, mean ± sample SD | 0.004344 ± 0.000159 mm |
| Source MLP, epoch-300 MAE, mean ± sample SD | 0.004965 ± 0.000602 mm |
| Table S9, maximum difference between saved and reevaluated predictions | Less than 0.000001 mm |

To evaluate the source last checkpoints, extract `surrogates.zip` into a local
directory and pass one extracted case directory to
`reproduction/evaluate_surrogate.py --run-dir ... --checkpoint last`. The main
frozen dataset is already included in `reproduction/data/`. Existing saved
predictions preserve the original run's arithmetic; reevaluation can differ
slightly with the hardware/backend.

## S10 provenance

The bilateral SAC, scratch DDPG and BO records originate from the fixed-surrogate
bilateral study. Its DDPG transfer arm was subsequently replaced by the
`A4_C4_NATIVE_OPT` experiment: actor and critic layers fc1–fc4 are transferred.
The 15 final DDPG transfer records and their selected designs are included in
`bilateral.zip`; the superseded C3 runs are not used in the final aggregation.
The executable paths are in `reproduction/studies/bilateral/`.

For FGL7, DDPG transfer seed 2, the original automated FEA record reports failure.
The final data package supplied a recovered nine-displacement profile through
`apply_recovered_failed_fea.py`. `recovered_fea_profiles.json` preserves that
profile, its source filename and hash. The five-run table is reproducible from
this recovery record; the original solver output supporting the recovery is
not included. Other final S10 FEA profiles are checked against the saved solver
summary records. All rewards, margins and ratios are recomputed from the nine
displacements and actions rather than copied from previously derived fields.

The arithmetic-objective columns of S10 are recomputed from the September
policy and optimizer archives. The two studies retain their actual protocols;
the archive does not imply a controlled comparison changing only the reward.

## Coverage

The saved logs allow reaggregation without training. They do not contain every
policy weight or the entire direct-optimization trajectory after evaluation
1,500. The selected incumbent under the full 382.5 s allowance is retained
separately. The training code can generate new policies, but numerical
trajectories and timing can differ with hardware and software.

These files do not execute Ansys or reproduce physical measurements. Their
presence establishes the traceable source of the stored numerical results,
not a new physical validation.
