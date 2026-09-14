# Paper-Safe Historical Diagnostic Summary

This file separates historical design diagnostics from final controlled RV-PI baselines. It does not alter the final method or its reported results.

## Safe for Main Text

- In earlier formulations of this solver, generic actor-critic optimization did not produce reliable validation improvements over supervised correction.
- The historical records motivate using verified continuation returns rather than allowing a learned critic to be the final long-horizon decision authority.
- The final macro action set is small (`{0, 1, 2, 4}`), so explicit solver continuation rollouts are computationally practical.
- Validation-based accept/rollback is motivated by observed closed-loop policy degradation after nonzero actor shifts.

These are design conclusions. They do not say that PPO, SAC, TD3, IQL, or actor-critic RL generally fails for PDE solving.

## Safe for Appendix

### Historical PPO diagnostic

Commit `327f4853bcd9b7e409bef0ae43cbb2c27d1521a9` contains a complete seed-42 shallow-water PPO diagnostic. On 20 validation cases, PPO trajectory relative L2 was `0.7769132614135742`, `0.8051373869180679`, and `0.8278388440608978` at budgets 16, 32, and 48. The corresponding GT-free random values were `0.7115214943885804`, `0.7114292874932289`, and `0.7203342616558075`. PPO did not pass the documented seed-42 feasibility condition, so no three-seed PPO study was run. This is a historical diagnostic, not a final RV-PI baseline.

### Historical TD3 diagnostic

Commit `de22e2db6d31ba693f0b018af8db742fe8b93b57` records a full-field neural-operator TD3 study. The supervised corrector reached 10-step relative L2 `0.344349`, while TD3 from scratch reached `0.348024`, `0.351344`, `0.370679`, and `0.417740` at 1, 2, 5, and 10 steps. Commit `1d178b52b4dff08fe7f27c2b5267f7cea9efb554` records a later latent-action Twin-Q formulation with full-cycle within-state critic Spearman `0.397`, pairwise accuracy `0.680`, and top-1 accuracy `0.288`; its best actual RL policy had 10-step relative L2 `0.334348` versus `0.307843` for the supervised latent corrector, and validation selected the supervised policy.

| Historical formulation | Scope | Strong recoverable evidence | Provenance |
| --- | --- | --- | --- |
| Discrete PPO | SWE, seed 42, 20 validation cases | PPO did not beat the stated baselines at B=16/32/48 | `327f4853bcd9b7e409bef0ae43cbb2c27d1521a9`; `results/solver_v3/shallow_water/rl_seed42.csv` |
| TD3 from scratch | Reaction-diffusion, 108 held-out samples | 10-step relative L2 `0.417740` versus supervised `0.344349` | `de22e2db6d31ba693f0b018af8db742fe8b93b57`; historical `docs/solver_v2_results.md` |
| Latent Twin-Q TD3+BC | Reaction-diffusion, three seeds | Critic `0.397/0.680/0.288` (Spearman/pairwise/top-1); validation chose supervised anchor | `1d178b52b4dff08fe7f27c2b5267f7cea9efb554`; `policy_selection.csv` |

## Do NOT Use in Paper

- Do not present the root IQL/Flow-Matching result as a final RV-PI baseline. It uses a different reaction-diffusion pipeline, continuous spectral actions, and a separate protocol.
- Do not claim SAC was evaluated. No recoverable SAC code or artifact exists.
- Do not claim rollout-guided FQI was trained in Solver V4. The selector/headroom diagnostic explicitly reports `fqi_run: false`.
- Do not use the Brusselator low-frequency/PCA oracle diagnostics as direct learned-policy comparisons; they are GT-based design tests and full RL training was not started.
- Do not portray the small, mixed three-seed continuous-latent rollout-verified result as robust evidence for the final method.
- Do not compare old PPO, TD3, IQL, oracle, or privileged beam numbers directly with final RV-PI unless the task, state, action space, budget, and evaluation protocol are controlled to match.

## Why Verified Rollouts Instead of Generic Actor-Critic RL

In earlier formulations of the present solver, generic actor-critic optimization did not produce reliable validation improvements. A continuous latent-action TD3+BC system could move the actor away from its supervised initialization, but the best actual policy was worse than the supervised corrector at long rollout horizons and validation consequently retained the supervised anchor. Critic ranking improved after within-state return and pairwise supervision, yet the recorded ranking quality was still not sufficiently dependable for direct long-horizon action selection. Separately, oracle checks showed that some candidate action spaces contained little useful continuation headroom, so an actor could not be expected to recover a robust benefit from them. The final macro formulation has only four refinement-count actions, `{0, 1, 2, 4}`, making explicit continuation evaluation practical. RV-PI therefore uses the learned policy to propose candidates, evaluates continuation returns with the solver, fits only to verified improvements, and accepts a policy update only when held-out validation supports it. This choice is a formulation-specific response to documented value-estimation and closed-loop distribution-shift risks, not a general claim about actor-critic RL for PDEs.

## Appendix Language

Historical PPO and TD3 experiments were retained as design diagnostics rather than final controlled baselines. In a seed-42 shallow-water PPO diagnostic, the learned discrete policy did not improve over the recorded GT-free random or gradient baselines at budgets 16, 32, or 48, so the planned three-seed extension was not run. In an earlier reaction-diffusion TD3 study, from-scratch TD3 degraded from 0.348024 at one step to 0.417740 at ten steps, while the corresponding supervised corrector reached 0.344349 at ten steps. A later latent-action Twin-Q version produced nonzero policy shifts but validation selected the supervised anchor, motivating solver-verified continuation returns in the final formulation.
