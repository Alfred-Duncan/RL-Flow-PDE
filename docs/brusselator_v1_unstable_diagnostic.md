# Brusselator V1 Unstable-Rollout Diagnostic

The archived V1 Brusselator results are retained as an instability diagnostic and are not used as paper-level cross-PDE evidence.

- The V1 coarse model achieved moderate one-step validation and test error.
- Its 38-step closed-loop rollouts diverged severely: validation trajectory relative L2 was 8.2505 and test trajectory relative L2 was 8.3774.
- The corresponding final-frame relative L2 values were 84.7149 on validation and 84.7666 on test.
- V1 RV-PI displayed positive relative behavior against several policy baselines on the completed seeds, but the underlying autoregressive solver is not sufficiently stable for a paper-level adaptive-refinement claim.

The V2 study therefore changes only the coarse neural-operator training objective, aligning it with the deployed autoregressive horizon before rebuilding all downstream components.
