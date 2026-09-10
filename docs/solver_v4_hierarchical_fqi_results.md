# Solver V4 Hierarchical Refinement Gate

Solver V4 reuses the official shallow-water data and the frozen Solver V3
coarse/local neural-operator checkpoints. It does not retrain either operator.
The only new Phase 1 model is a set-aware selector with 64-dimensional patch
embeddings, two four-head Transformer layers, selected-set embeddings, and a
small global current/provisional-field CNN.

## Training protocol

All set-aware labels use only the 230 training cases. The final training set
contains 15,640 closed-loop physical states drawn from coarse-only, random,
independent-utility, and privileged-macro trajectories. For every state, a
random selected set of size zero through three is actually applied, and each
remaining patch label is the change in **full 256-by-256 next-frame MSE** after
that conditional local correction. The selector uses Huber regression plus
within-state pairwise ranking loss.

Validation uses all 20 held-out validation cases and 340 physical states. It
sequentially recomputes scores after every selected correction. Ground truth is
used only to compute validation labels and the privileged comparison, never as
a selector input.

## Selector result

| Selector | q | Top-1 overlap | Spearman | Pairwise accuracy | Immediate gain recovery | Global error after selection |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Set-aware | 1 | 0.3353 | 0.8024 | 0.8261 | 0.5240 | 0.7382 |
| Set-aware | 2 | 0.3662 | 0.8026 | 0.8423 | 0.0521 | 0.7651 |
| Set-aware | 4 | 0.2816 | 0.7243 | 0.8219 | -4.4222 | 0.7659 |
| Independent PatchUtility | 1 | 0.0000 | -0.3711 | 0.3537 | -1600.4758 | 0.7514 |
| Independent PatchUtility | 2 | 0.0059 | -0.3887 | 0.3466 | -659.0235 | 0.7949 |
| Independent PatchUtility | 4 | 0.0000 | -0.4605 | 0.3211 | -1530.3145 | 0.8189 |

The recovery metric is the deployed conditional global-MSE gain divided by the
privileged best conditional gain, averaged over sequential selections. Negative
values indicate that the selected local correction increased full-frame MSE;
large negative values occur when the privileged denominator is small.

## Gate decision

The predefined minimum recoveries are 85% for q=1, 80% for q=2, and 75% for
q=4. The set-aware selector improves substantially over the independent
selector, but reaches only 52.4%, 5.2%, and -442.2%, respectively. It therefore
does not pass the spatial-selector gate.

Consequently, the deployable macro oracle, BeamBC, rollout-guided FQI,
three-seed study, and FQI ablations are not run. The required premise for these
stages, a sufficiently accurate GT-free spatial selector, is not satisfied.
The existing privileged B=32 beam headroom remains an upper diagnostic rather
than evidence for a deployable hierarchical policy.
