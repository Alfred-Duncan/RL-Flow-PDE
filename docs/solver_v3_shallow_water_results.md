# Solver V3 Shallow-Water Diagnostic

Solver V3 uses the official Laplace Neural Operator shallow-water archives
(`72 x 256 x 256` trajectories), a shared 64-by-64 coarse FNO transition, and
a shared 48-by-48 local FNO corrector over an 8-by-8 native patch grid. The
closed-loop state after local refinement is fed into the next coarse transition.
Reflection padding is used only to construct boundary local-operator inputs;
out-of-domain correction halo values are discarded during native-domain blending.

## Fixed data split

Cases 0--229 are training cases, 230--249 are validation cases, and 250--299
are test cases. Patch-utility labels are constructed only from training cases;
the reported utility validation row is not used to fit its predictor.

## Completed diagnostic results

The autoregressive coarse rollout has validation trajectory relative L2
`0.70577` and test trajectory relative L2 `0.68967`. On the validation
single-transition check, applying all shared local corrections reduces the first
case from `0.07103` to `0.01341` relative L2.

The PatchUtilityNet is trained from 250,240 training patch transitions. Its
held-out validation Pearson correlation with true immediate local MSE gain is
`0.95511`, Spearman correlation is `0.77636`, and positive-gain sign accuracy
is `0.86232` over 21,760 validation patches.

The budget oracle uses 16 fixed validation cases, the same total local-call
budget for every planner, macro allocation counts `{0, 1, 2, 4}`, and beam
width 8. It is a privileged diagnostic only: true targets choose the patches
within each macro action and are never part of a deployable policy observation.

| Budget | Greedy budget oracle | Beam long-horizon oracle | Beam gain vs greedy | Positive cases |
| --- | ---: | ---: | ---: | ---: |
| 16 | 0.69761 | 0.66487 | 4.69% | 75.0% |
| 32 | 0.74479 | 0.65593 | 11.93% | 87.5% |
| 48 | 0.64861 | 0.65576 | -1.10% | 68.75% |

The B=32 result meets the predefined strong gate: at least 5% beam-over-greedy
gain and at least 65% positive cases. Seed-42 PPO evaluation therefore proceeds.
The B=48 result is retained as negative diagnostic evidence; the oracle does
not improve monotonically with a larger budget under this finite beam search.
