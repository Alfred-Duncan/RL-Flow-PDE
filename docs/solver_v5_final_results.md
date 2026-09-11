### Main Result

Coarse: `0.740347 +/- 0.000000` trajectory Relative L2.

Best non-RL: RandomMacro at `0.708093 +/- 0.005420`.

BeamBC: `0.739403 +/- 0.000000`.

RL: `0.691026 +/- 0.008779`.

RL gain over strongest deployable baseline: `2.41%` mean trajectory-error reduction versus RandomMacro. RV-PI improves over BeamBC by `6.54% +/- 1.19%`.

Full-horizon vs immediate-only: RV-PI `0.691026 +/- 0.008779`; Immediate-Only PI `0.730245 +/- 0.000571`. This is a `5.37%` mean trajectory-error reduction for full-horizon returns.

3-seed: validation-selected RV-PI policies evaluated once on the fixed held-out test split for seeds `42`, `123`, and `2026`. The seed-level results are in `final_comparison_3seed.csv`; the aggregate table is `three_seed_summary.csv`.

The official operator input is called the official condition field `a(x,y)`. Spatial refinement is GT-free at deployment; GT is used only for train returns and held-out metrics. The evidence supports a benefit over the learned/myopic macro baselines and a positive but moderate mean benefit over stochastic RandomMacro; it does not support an unconditional large per-seed margin over every random rollout.
