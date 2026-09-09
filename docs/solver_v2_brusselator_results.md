# Solver V2: Official 3D Brusselator Diagnostic

This diagnostic preserves the Reaction-Diffusion results and adds the official LNO 3D_Brusselator NPZ only. The baseline is a newly trained FNO baseline, not LNO; the official LNO implementation was not adapted because its script is globally coupled and CUDA-hardwired.

- Official preprocessing: `(case, time, x, y) = (*, 39, 28, 28)`, then official `r=2` subsampling to `(*, 39, 14, 14)`.
- FNO baseline validation Relative L2: 0.138126; held-out test Relative L2: 0.137521.
- Action controllability: median correction difference 0.122892; median next-error difference 0.0212646.
- Rewards use ground-truth Relative L2 reduction plus a small action cost. This is a validation oracle diagnostic, not deployment-time reward.
- The solver state is the full trajectory field; solver iterations are separate from the 39 physical time indices.
- The fixed low-frequency correction basis uses no autoencoder. Its reference continuation is a fixed contraction to the FNO initial trajectory, not a learned policy or a reconstructed Brusselator residual.

## One-Action Headroom

|   Horizon |   MeanRelativeOracleGain |   MedianRelativeOracleGain |   PositiveGainRate |   ActionsDifferentRate |   Samples |
|----------:|-------------------------:|---------------------------:|-------------------:|-----------------------:|----------:|
|  5.000000 |                 0.001585 |                   0.000000 |           0.328125 |               0.328125 | 64.000000 |
| 10.000000 |                 0.006084 |                   0.000245 |           0.500000 |               0.500000 | 64.000000 |
| 20.000000 |                 0.009880 |                   0.010133 |           0.843750 |               0.843750 | 64.000000 |

## Sequential Headroom

|   Horizon |     Cases |   GreedyFinalErrorMean |   LongFinalErrorMean |   MedianCaseRelativeGain |   PositiveCaseRate |   ActionsDifferentRate |   AbsoluteSequentialGain |   RelativeSequentialGain |
|----------:|----------:|-----------------------:|---------------------:|-------------------------:|-------------------:|-----------------------:|-------------------------:|-------------------------:|
|  5.000000 | 64.000000 |               0.134086 |             0.134163 |                 0.000000 |           0.281250 |               0.178125 |                -0.000078 |                -0.000578 |
| 10.000000 | 64.000000 |               0.133342 |             0.132414 |                 0.005995 |           0.750000 |               0.309375 |                 0.000927 |                 0.006956 |
| 20.000000 | 64.000000 |               0.133494 |             0.130908 |                 0.017021 |           0.921875 |               0.546875 |                 0.002586 |                 0.019371 |

## K=20 Decision Stages

| Stage   |   ActionsDifferentRate |   MeanImmediateGain |   MeanTrajectoryGain |
|:--------|-----------------------:|--------------------:|---------------------:|
| Early   |               0.867188 |           -0.104952 |             0.002586 |
| Middle  |               0.738839 |           -0.027303 |             0.002586 |
| Late    |               0.080357 |           -0.000784 |             0.002586 |

Conclusion: weak headroom; do not start full RL training. The specified strong criterion is relative sequential gain >= 5% and positive-case rate >= 65%; observed K=20 values are 1.94% and 92.19%.
