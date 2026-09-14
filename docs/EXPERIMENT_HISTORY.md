# Experiment History and Design Diagnostics

## 1. Purpose

This document records abandoned and intermediate sequential-decision formulations in the repository for reproducibility and design transparency. It is an evidence-recovery record, not a replacement for the final controlled RV-PI comparisons. The audit searched the current tree, every commit reachable from `main` and `origin/main`, commit messages, renamed/deleted paths, and unreachable objects. No additional branch was available; `git fsck --no-reflogs --unreachable` found no unreachable experiment artifact. The only deleted files were two early figure images, not code, configurations, or result tables.

Evidence grades: **A** = code, configuration, and numerical artifact; **B** = implementation plus partial artifact; **C** = proposal/code reference without a reliable quantitative run. A grade describes recoverability, not whether the method was successful.

## 2. Chronological Experiment Map

| Version/date | Algorithm or diagnostic | Formulation and outcome | Decision | Evidence | Commit |
| --- | --- | --- | --- | --- | --- |
| Initial, 2026-09-03 | IQL + Flow Matching / Twin-Q | Reaction-diffusion, continuous spectral correction actions. Root result tables show RL-FM at 10 steps worse than LNO-only and deterministic correction. | Historical separate-task diagnostic; not a final-method baseline. | A | `cf4a1c696a6d916013b5e8af5c7804f03453f556` |
| Solver V2, 2026-09-03 | TD3 from scratch, TD3+BC, TD3+BC without physics | Continuous neural-operator correction; from-scratch TD3 degraded with horizon. | Retained supervised anchor and later changed the formulation. | A | `de22e2db6d31ba693f0b018af8db742fe8b93b57` |
| Solver V2, 2026-09-04 | Latent-action TD3+BC / Twin-Q | A 32-D actor latent was decoded to a field correction. Early critic ranking was weak. | Strengthened critic supervision and diagnostics. | A | `4d8ec08603dbe1e879b22ee1210070184ed65ce3`, `ca5fecb787e8a17f6c36790776578e6bfe4c75ce` |
| Solver V2, 2026-09-08 | Within-state Twin-Q ranking | Grouped K-step returns, advantage and pairwise ranking losses; actual policy shift was rejected by validation selection. | Do not deploy generic TD3 actor updates. | A | `1d178b52b4dff08fe7f27c2b5267f7cea9efb554` |
| Solver V2, 2026-09-09 | Critic-screened rollout-verified policy improvement | Twin-Q screened continuous latent candidates, but PDE rollouts selected verified targets and validation accepted/rejected updates. | Historical precursor; aggregate gain was tiny and seed-level outcomes mixed. | A | `80da75468cac384cd1f422e8838bb82ca7e12014` |
| Solver V2, 2026-09-09 | Reaction-diffusion oracle/headroom | One-action and sequential greedy-versus-oracle checks. | Insufficient exploitable long-horizon structure under the stated gate. | A | `625642500b0582b880e69efef67d3b19a8d9f229`, `faaa539a8cca583bcf87372b634274b60df54f31` |
| Solver V2 Brusselator, 2026-09-09 | Low-frequency/PCA oracle headroom | Full-trajectory correction oracle, then PCA action-space checks. | Full RL training was not started. | A | `32c40df28cb61a58c1a603aaa3d33baae58b68e4`, `01c2599aee17bbae1d6cc72b881787a2e4256761` |
| Solver V3 SWE, 2026-09-10 | Discrete PPO | Patch-or-advance actor-critic over a local-refinement budget. | Seed-42 feasibility failed; no three-seed PPO run. | A | `327f4853bcd9b7e409bef0ae43cbb2c27d1521a9` |
| Solver V4 SWE, 2026-09-10 | Set-aware spatial selector and deployable macro headroom | GT-free selector was evaluated; deployable temporal beam did not beat greedy. | Rollout-guided FQI was explicitly not run. | A for diagnostic; C for FQI | `6c332e823891b64879292f7abbf5e98524ef4bb3`, `eb300588c76e01884d38cc9009288d2cd21529c2` |
| Solver V5, 2026-09-10 onward | Final RV-PI | Small discrete macro action set and solver-verified continuation returns. | Final controlled method; not modified by this audit. | Separate final evidence | `269d8a8520d356631f76a32618cf185629930e9b` onward |

## 3. PPO Diagnostics

### Formulation and provenance

The only recovered PPO run is the shallow-water V3 run in commit `327f4853bcd9b7e409bef0ae43cbb2c27d1521a9`:

- **Code/config:** `scripts/run_solver_v3_swe.py` (`policy_observation`, `collect_ppo_episode`, `ppo_advantages`, `train_ppo_seed42`), `src/solver_v3/models/patch_policy.py`, and `configs/solver_v3_swe.yaml`.
- **State:** a GT-free 32-by-32 downsampled two-field encoding of current and provisional numerical fields; 64 GT-free local descriptors; selected-patch mask; remaining-budget fraction; time fraction; selected-count context. The utility prior is frozen.
- **Action:** categorical action over 64 patch indices plus action 64, `advance`. Invalid, already selected, exhausted-budget, and per-step-limit actions are masked. The action is discrete.
- **Reward/horizon:** selecting a patch receives `-0.001`; advancing receives negative next-frame relative L2 against GT. Each episode has 17 physical transitions (`stride=4` over 72 frames), with at most six refinements per physical transition. GT is not an observation, but is used for the training return and reported metric.
- **Hyperparameters:** seed 42; budgets `[16, 32, 48]`; 20 updates; 16 training trajectories/update; four PPO epochs/update; minibatch 128; AdamW learning rate `3e-4`; gamma `0.99`; GAE lambda `0.95`; clip ratio `0.2`; entropy weight `0.01`; beam-width-8 privileged diagnostic.
- **Artifacts:** `results/solver_v3/shallow_water/ppo_seed42_training.csv`, `rl_seed42.csv`, `accuracy_compute_pareto.csv`, and `budget_oracle_headroom.csv`.

### Verified seed-42 outcome

All values below are validation mean trajectory relative L2 on 20 cases, lower is better.

| Budget | Random | Gradient heuristic | Supervised myopic | PPO |
| ---: | ---: | ---: | ---: | ---: |
| 16 | 0.7115214943885804 | 0.725046044588089 | 0.737413939833641 | 0.7769132614135742 |
| 32 | 0.7114292874932289 | 0.7228525310754776 | 0.7987825751304627 | 0.8051373869180679 |
| 48 | 0.7203342616558075 | 0.7375265121459961 | 0.8399438798427582 | 0.8278388440608978 |

The result is reproducibly supported by `rl_seed42.csv`. PPO did not beat both the GT-free gradient and supervised-myopic baselines at any budget. The documentation explicitly says that the seed-42 feasibility condition failed and the three-seed experiment was intentionally not run. The code itself is seed-42-specific (`train_ppo_seed42`, `evaluate_seed42`) and contains no executable three-seed gate; the decision to withhold a three-seed study is documented rather than encoded as a general gate. Evidence grade: **A** for the seed-42 diagnostic, but it is not a three-seed paper result.

## 4. TD3 / TD3+BC Diagnostics

### Original full-field neural-operator TD3

Commit `de22e2db6d31ba693f0b018af8db742fe8b93b57` introduced `scripts/run_solver_v2.py`, `src/solver_v2/rl/td3_bc.py`, `src/solver_v2/models/operator_actor.py`, `src/solver_v2/models/operator_critic.py`, and `configs/default.yaml`. The state is a residual-conditioned neural-operator representation. The deterministic actor outputs a continuous correction action, and a Twin-Q critic scores state-action pairs. The PDE reward combines GT relative-error reduction, physics-energy reduction, action cost, and a step cost. GT participates in rewards and offline labels, not in deployment-time actor inputs.

The initial 10-step relative-L2 evidence from `docs/solver_v2_results.md` at that commit is:

| Method | 1 step | 2 steps | 5 steps | 10 steps |
| --- | ---: | ---: | ---: | ---: |
| Supervised Neural Operator Corrector | 0.346581 | 0.345879 | 0.344928 | 0.344349 |
| TD3+BC with physics regularization (`RL Neural Operator Solver`) | 0.346581 | 0.345879 | 0.344928 | 0.344349 |
| TD3 from scratch | 0.348024 | 0.351344 | 0.370679 | 0.417740 |
| TD3+BC without physics regularization | 0.349086 | 0.351583 | 0.366267 | 0.413600 |

`RL Neural Operator Solver` is the documented TD3+BC model with scale-normalized physics regularization and validation checkpoint selection. Its reported trajectory values match the supervised corrector in this artifact. Thus the exact requested numbers are supported: the supervised 10-step value is `0.344349`; TD3 from scratch is `0.348024`, `0.351344`, `0.370679`, and `0.417740` at 1, 2, 5, and 10 steps. The failure is long-horizon degradation, not absence of a code path. Evidence grade: **A**.

### Latent-action TD3+BC and validation selection

Commits `4d8ec08603dbe1e879b22ee1210070184ed65ce3`, `ca5fecb787e8a17f6c36790776578e6bfe4c75ce`, and `1d178b52b4dff08fe7f27c2b5267f7cea9efb554` changed the action to a continuous 32-D latent `z`, decoded by `CorrectionOperatorDecoder` into a correction field. The actor receives five residual-conditioned field channels and four scalar inputs; the actor and critic widths are 32, state dimension is 96, and critic depth is two. The actor is warm-started by supervised correction. The critic is Twin-Q over `(state, z)` and receives grouped Monte-Carlo return, advantage, and within-state ranking supervision.

The final strengthened configuration uses seeds `[42, 123, 2026]`, actor/critic learning rates `5e-4`/`1e-3`, 50 TD3 epochs, 40 critic MC epochs, group batch size 8, 480 candidate groups, gamma `1.0`, tau `0.005`, policy delay 2, target noise `0.01` clipped at `0.02`, and TD3+BC weights `lambda_q=0.05`, BC from 10 to 2, physics `0.05`, and step `0.0005`. Validation ranking gates were Spearman `>0.5`, pairwise `>0.65`, and top-1 `>0.25`.

At commit `1d178b52b4dff08fe7f27c2b5267f7cea9efb554`, the recorded full-cycle within-state diagnostics are mean Spearman `0.397`, pairwise accuracy `0.680`, and top-1 accuracy `0.288`. The 10-step supervised latent corrector is `0.307843`; the best actual RL policy is `0.334348`; standalone TD3+BC is `0.307843`. The actual policy had nonzero shifts and actor updates, but validation selected the supervised policy for all three seeds: validation errors were `0.404368` versus `0.417994` (seed 42), `0.306576` versus `0.330655` (seed 123), and no actual RL actor was available for seed 2026. This is direct evidence that actor optimization was attempted, shifted the policy, and was rejected by validation selection. Evidence grade: **A**.

## 5. Critic-Ranking and Critic-Screened Rollout Diagnostics

The historical continuous-latent rollout-verified method is implemented in `src/solver_v2/rl/rollout_verified_pi.py` and attached in commit `80da75468cac384cd1f422e8838bb82ca7e12014`. It is distinct from final V5 RV-PI:

- It starts from the supervised latent actor, constructs 48 bounded latent candidates, has Twin-Q screen the candidates, evaluates the top 8 plus 3 random candidates with true five-step PDE returns, and trains only on verified positive-advantage targets.
- Its actor update has a supervised trust penalty and a latent trust-region projection. No critic gradient reaches that actor.
- It evaluates 96 training states/cycle, runs three cycles, and accepts a proposed policy only after validation trajectory error improves. GT is used in the candidate-return rollouts and validation, not in the state input.

`results/solver_v2/tables/final_rl_contribution.csv` records the following held-out outcome:

| Seed | Supervised error | Verified RL error | RL minus supervised | Accepted cycles |
| ---: | ---: | ---: | ---: | ---: |
| 42 | 0.37307792860600686 | 0.3717834423813555 | -0.0012944862246513367 | 1 |
| 123 | 0.278114404115412 | 0.278114404115412 | 0.0 | 0 |
| 2026 | 0.2723357135223018 | 0.27349580865767265 | 0.0011600951353708533 | 1 |

The historical documentation reports a three-seed mean advantage of only `0.000045` relative L2 with one of three positive test differences. It is evidence that validation acceptance/rollback was operational, but not evidence of a robust generic actor-critic gain. Evidence grade: **A**, appendix-only at most.

## 6. SAC / IQL / Other RL Variants

### IQL + Flow Matching

IQL is genuinely implemented and evaluated in the initial root pipeline, not in Solver V2. Provenance: `src/rl/iql.py`, `scripts/experiment.py` (`train_iql`, `train_awfm`, `evaluate_all`), `configs/default.yaml`, `results/tables/critic_results.csv`, `main_results.csv`, and `ablation.csv`; the code first appears in `cf4a1c696a6d916013b5e8af5c7804f03453f556`.

This is a different reaction-diffusion formulation: a 96-D state embedding, a 129-D continuous low-frequency spectral correction action (`2*8*8+1`), an IQL Twin-Q/value critic, and an advantage-weighted flow-matching policy. Offline transitions include GT-derived terminal and sequential corrections; the reward is log GT-error reduction plus physics improvement, action penalty, and step penalty. The configuration records seed 42, 96/24/36 train/validation/test cases, 160 critic epochs, 100 advantage-weighted FM epochs, gamma `0.95`, expectile `0.7`, beta `1.0`, and four sampled candidates.

The recoverable critic artifact reports Q-return Spearman `0.2766190364652664` over 1,980 samples. In `results/tables/main_results.csv`, 10-step RL-FM relative L2 is `0.120867 +/- 0.0506791`, versus `0.104686 +/- 0.0361394` for LNO-only and `0.102612 +/- 0.0363922` for the deterministic correction. The corresponding ablation has Full RL-FM `0.110951 +/- 0.0365086` and deterministic correction `0.0988408 +/- 0.0348457` on its 12-case subset. Evidence grade: **A** for a historical separate-task run; it is not an apples-to-apples V5 baseline.

### SAC

No SAC implementation, configuration, commit, result table, log, checkpoint metadata, deleted path, or reachable branch reference was found. SAC was not run in the recoverable history. Evidence grade: **C** (absence finding, not an experimental result).

### Planned rollout-guided FQI

Solver V4 contains a GT-free set-aware spatial selector and deployable macro headroom diagnostic, but no FQI result. `docs/solver_v4_hierarchical_fqi_results.md` and `scripts/run_solver_v4_swe.py` explicitly record `fqi_run: false`: the 16-case B=32 deployable beam trajectory error was `0.75200` versus greedy `0.74522`, a `-0.91%` relative gain, below the stated 2% continuation threshold. Do not describe FQI as trained or evaluated. Evidence grade: **C** for FQI; the selector/headroom diagnostic itself is **A**.

## 7. Long-Horizon Headroom Diagnostics

These diagnostics test whether a formulation contains enough exploitable sequential structure before asking RL to learn it. They do not establish that RL is intrinsically unsuitable for PDE solvers.

**Reaction-diffusion Solver V2 one-action oracle** (`results/solver_v2/tables/oracle_headroom_by_horizon.csv`):

| Horizon | Mean relative oracle gain | Median gain | Positive-gain rate | Action-different rate | Samples |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 5 | 0.0015070867485673 | 0 | 0.2111111111111111 | 0.2111111111111111 | 360 |
| 10 | 0.0058399197790638 | 0 | 0.35 | 0.3527777777777778 | 720 |
| 20 | 0.00542758434378029 | 0 | 0.34444444444444444 | 0.34444444444444444 | 1440 |

The three-seed sequential K=20 aggregate in `docs/solver_v2_results.md` is `0.0120479`, below the predefined 2% threshold; its positive-case rate is `0.569444` and action-different rate is `0.530556`. The per-seed table is `results/solver_v2/tables/sequential_oracle_headroom.csv`.
- **Official Brusselator low-frequency diagnostic:** the K=20 sequential oracle gain is `0.019371168917444406` with positive-case rate `0.921875`, below its predeclared 5% strong criterion. The PCA action-space follow-up did not fix this: PCA32 fixed-contraction K=20 gain is `-0.006271998159852846`, and all GreedyLocal PCA continuations are negative. Full RL training was not started. Sources: `results/solver_v2/brusselator/*headroom*.csv` and `docs/solver_v2_brusselator_results.md`.
- **SWE V3 privileged oracle:** B=16/32/48 beam-versus-greedy gains are `4.692747087121576%`, `11.929488932290576%`, and `-1.0816753392421619%` on 16 validation cases. B=32 cleared the 5%/65% diagnostic gate and motivated the seed-42 PPO check, whereas B=48 was retained as a negative finite-beam diagnostic. Source: `results/solver_v3/shallow_water/budget_oracle_headroom.csv`.
- **SWE V4 deployable macro diagnostic:** after a GT-free frozen-set selector, B=32 beam did not improve on deployable greedy (`-0.91%`), so rollout-guided FQI was not started.

## 8. Design Lessons That Led to RV-PI

- In earlier formulations, generic actor-critic optimization introduced a value-estimation and actor-update bottleneck: policy shifts occurred, but validation often selected the supervised anchor.
- Strong supervised/local correction did not imply that TD3 fine-tuning would preserve long-horizon closed-loop behavior.
- Within-state critic ranking improved but was insufficiently reliable for direct action selection in the latent-action formulation.
- Several diagnostics had too little usable long-horizon headroom, so a learned policy could not reasonably be expected to outperform greedy or supervised decisions.
- In the final macro formulation, the discrete action set is `{0, 1, 2, 4}`. Explicit continuation rollout is practical, avoiding a learned long-horizon value as the final decision authority.
- RV-PI therefore uses solver-verified candidate returns, and validation accept/rollback limits policy-induced closed-loop distribution shift.
