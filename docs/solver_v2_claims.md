# Solver V2 Claims

- RL produces a functional iterative PDE solver, but long-horizon optimization does not yet provide a clear advantage over supervised correction.
- 10-step Base FNO Relative L2 mean: 0.347362.
- 10-step Supervised Neural Operator Corrector Relative L2 mean: 0.344349.
- 10-step RL Neural Operator Solver Relative L2 mean: 0.344349.
- Validation checkpoint selection is used for the full solver; in this run TD3 fine-tuning did not produce a checkpoint that improved over the supervised warm start.
- Physics residual is reported as a constraint diagnostic, not used as the core RL reward.
