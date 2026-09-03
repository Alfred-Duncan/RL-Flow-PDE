# Paper Claims

- RL-guided Flow Matching does not support a broad improvement claim at 10 steps: RL-FM 0.120867 +/- 0.0506791, Supervised FM 0.360872 +/- 0.718466, LNO 0.104686 +/- 0.0361394.
- The gradient residual baseline remains stronger than RL-FM at 10 steps in this run: Gradient 0.10281 +/- 0.035267, RL-FM 0.120867 +/- 0.0506791.
- Critic evaluation is reported on held-out transitions with Q Return Spearman = 0.2766 and Advantage Return Spearman = 0.0514.
- The final protocol uses one shared frozen state encoder after supervised FM, normalized actions, train-only state/reward statistics, sequential transition trajectories, and validation-calibrated STOP.
- GT is used for offline transition reward construction and evaluation only, not policy/critic inference inputs.
