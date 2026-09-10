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

## Frozen Multi-Patch Semantics Revision

The original V4 selector evaluation selected one patch, applied its correction,
then recomputed the local operator on the modified provisional field before the
next selection. That was inconsistent with a single physical transition and
with the local corrector's training distribution.

The v2 revision freezes all 64 local-corrector proposals from the same coarse
provisional field. Every selected set is then executed exactly once with the
canonical normalized cosine blend used by `HybridRefiner`. Conditional labels
are full-frame MSE differences between `ApplyFrozenSet(S)` and
`ApplyFrozenSet(S union {p})`; no local proposal is regenerated during either
training or sequential selection. V2 also uses V4 closed-loop feature statistics
and adds proposal descriptors and GT-free overlap/interference features.

| q | Aggregate gain/gap recovery | Positive improvement rate | Spearman | Pairwise accuracy |
| --- | ---: | ---: | ---: | ---: |
| 1 | 82.94% | 87.06% | 0.8529 | 0.8562 |
| 2 | 83.87% | 86.76% | 0.8975 | 0.8824 |
| 4 | 81.45% | 66.18% | 0.8516 | 0.8575 |

This resolves the earlier unstable q>1 gain-ratio diagnostic. The set-aware
selector now retains most of the privileged frozen-set improvement, although
q=4 misses the 70% positive-improvement target narrowly.

## Deployable Macro Diagnostic

With the frozen v2 selector, all spatial decisions are GT-free and both macro
policies use exactly 32 local calls over each 17-transition trajectory. On the
16-case validation diagnostic, deployable greedy macro allocation has trajectory
relative L2 `0.74522`; deployable beam allocation has `0.75200`. The beam versus
greedy relative gain is `-0.91%` and the positive-case rate is `68.75%`.

Thus, while the spatial selector is no longer the main bottleneck, the required
deployable long-horizon temporal-allocation headroom is not present in this
diagnostic. This is below the 2% continuation threshold, so FQI is not trained.
No PPO, FQI, or three-seed result is claimed.

The official input diagnostic finds that `inputs` and output t=0 share a shape
but are not equivalent (mean per-case correlation `-0.2981`). Future Markov-state
revision may need official input context; this round intentionally does not
change the solver state.
