# Paper Claims

- Supervised FM vs no refinement at 10 steps: 0.104699 +/- 0.0360001 vs 0.104686 +/- 0.0361394. In this run, supervised Flow Matching does not improve over LNO-only.
- RL-guided FM vs Supervised FM at 10 steps: 0.105023 +/- 0.0362686 vs 0.104699 +/- 0.0360001. In this run, advantage weighting does not improve the correction policy.
- Same-iteration and wall-time comparisons favor the gradient residual refinement baseline on this compact benchmark.
- Critic validation is mixed: Q Spearman = 0.3976, Advantage Spearman = 0.2594, while Pearson correlations are negative.
- RL-FM executes sequential correction steps after policy-improvement rounds, but those steps do not yield reliable monotonic error reduction.
- Results are reported as measured outcomes; unsupported improvements are explicitly rejected.
- GT is used for offline reward construction and evaluation only, not policy/critic inference inputs.
