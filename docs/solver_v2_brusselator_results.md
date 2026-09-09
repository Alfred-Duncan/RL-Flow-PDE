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

## Action-Space Headroom Diagnostic

This is a PCA correction oracle diagnostic, not a final RL solver. PCA uses only FNO residual corrections from the official training split, including residuals at alpha = 0, 0.25, 0.50, and 0.75 intermediate trajectories. Validation/test targets are not used to fit the PCA basis or candidate actions.

### PCA Basis Quality

|   LatentDim |   ExplainedVarianceRatio |   ReconstructionRelativeL2Mean |   ReconstructionRelativeL2Median |   TrainSamples |
|------------:|-------------------------:|-------------------------------:|---------------------------------:|---------------:|
|   16.000000 |                 0.993767 |                       0.108165 |                         0.114875 |    3200.000000 |
|   32.000000 |                 0.999909 |                       0.018001 |                         0.018763 |    3200.000000 |
|   64.000000 |                 0.999929 |                       0.016178 |                         0.017092 |    3200.000000 |
|   96.000000 |                 0.999935 |                       0.015537 |                         0.016637 |    3200.000000 |

### Action-Space K=20 Summary

| ActionSpace    |   LatentDim |   ExplainedVariance | Continuation     |   K20GreedyError |   K20LongError |   RelativeSequentialGain |   PositiveCaseRate |   ActionsDifferentRate |
|:---------------|------------:|--------------------:|:-----------------|-----------------:|---------------:|-------------------------:|-------------------:|-----------------------:|
| LowFrequency16 |          16 |          nan        | FixedContraction |         0.133494 |       0.130908 |                 0.019371 |           0.921875 |               0.546875 |
| PCA16          |          16 |            0.993767 | FixedContraction |         0.121610 |       0.122523 |                -0.007508 |           0.093750 |               0.568750 |
| PCA16          |          16 |            0.993767 | GreedyLocal      |         0.121610 |       0.125211 |                -0.029608 |           0.000000 |               0.841406 |
| PCA32          |          32 |            0.999909 | FixedContraction |         0.121398 |       0.122160 |                -0.006272 |           0.171875 |               0.566406 |
| PCA32          |          32 |            0.999909 | GreedyLocal      |         0.121398 |       0.125452 |                -0.033392 |           0.000000 |               0.825000 |
| PCA64          |          64 |            0.999929 | FixedContraction |         0.121397 |       0.122184 |                -0.006486 |           0.218750 |               0.564063 |
| PCA64          |          64 |            0.999929 | GreedyLocal      |         0.121397 |       0.125310 |                -0.032237 |           0.000000 |               0.839063 |
| PCA96          |          96 |            0.999935 | FixedContraction |         0.121276 |       0.122041 |                -0.006306 |           0.250000 |               0.539844 |
| PCA96          |          96 |            0.999935 | GreedyLocal      |         0.121276 |       0.125385 |                -0.033877 |           0.000000 |               0.830469 |

### One-Action Oracle

|   LatentDim |   Horizon |   MeanRelativeOracleGain |   MedianRelativeOracleGain |   PositiveGainRate |   ActionsDifferentRate |   Samples |
|------------:|----------:|-------------------------:|---------------------------:|-------------------:|-----------------------:|----------:|
|   16.000000 |  5.000000 |                 0.000043 |                   0.000000 |           0.093750 |               0.093750 | 64.000000 |
|   16.000000 | 10.000000 |                 0.000080 |                   0.000000 |           0.140625 |               0.140625 | 64.000000 |
|   16.000000 | 20.000000 |                 0.000071 |                   0.000000 |           0.171875 |               0.171875 | 64.000000 |
|   32.000000 |  5.000000 |                 0.000074 |                   0.000000 |           0.078125 |               0.078125 | 64.000000 |
|   32.000000 | 10.000000 |                 0.000110 |                   0.000000 |           0.078125 |               0.078125 | 64.000000 |
|   32.000000 | 20.000000 |                 0.000082 |                   0.000000 |           0.156250 |               0.156250 | 64.000000 |
|   64.000000 |  5.000000 |                 0.000019 |                   0.000000 |           0.093750 |               0.093750 | 64.000000 |
|   64.000000 | 10.000000 |                 0.000037 |                   0.000000 |           0.171875 |               0.171875 | 64.000000 |
|   64.000000 | 20.000000 |                 0.000047 |                   0.000000 |           0.265625 |               0.265625 | 64.000000 |
|   96.000000 |  5.000000 |                 0.000075 |                   0.000000 |           0.156250 |               0.156250 | 64.000000 |
|   96.000000 | 10.000000 |                 0.000169 |                   0.000000 |           0.281250 |               0.281250 | 64.000000 |
|   96.000000 | 20.000000 |                 0.000150 |                   0.000000 |           0.328125 |               0.328125 | 64.000000 |

### Sequential Oracle

|   LatentDim | Continuation     |   Horizon |   Cases |   GreedyFinalErrorMean |   LongFinalErrorMean |   MedianCaseRelativeGain |   PositiveCaseRate |   ActionsDifferentRate |   AbsoluteSequentialGain |   RelativeSequentialGain |
|------------:|:-----------------|----------:|--------:|-----------------------:|---------------------:|-------------------------:|-------------------:|-----------------------:|-------------------------:|-------------------------:|
|          16 | FixedContraction |        10 |      64 |               0.125584 |             0.125805 |                 0.000000 |           0.375000 |               0.246875 |                -0.000221 |                -0.001762 |
|          16 | FixedContraction |        20 |      64 |               0.121610 |             0.122523 |                -0.002678 |           0.093750 |               0.568750 |                -0.000913 |                -0.007508 |
|          16 | GreedyLocal      |        10 |      64 |               0.125584 |             0.127518 |                -0.011325 |           0.015625 |               0.698438 |                -0.001935 |                -0.015404 |
|          16 | GreedyLocal      |        20 |      64 |               0.121610 |             0.125211 |                -0.025103 |           0.000000 |               0.841406 |                -0.003601 |                -0.029608 |
|          32 | FixedContraction |        10 |      64 |               0.125863 |             0.126059 |                -0.000623 |           0.328125 |               0.265625 |                -0.000197 |                -0.001562 |
|          32 | FixedContraction |        20 |      64 |               0.121398 |             0.122160 |                -0.002231 |           0.171875 |               0.566406 |                -0.000761 |                -0.006272 |
|          32 | GreedyLocal      |        10 |      64 |               0.125863 |             0.128035 |                -0.013700 |           0.015625 |               0.710938 |                -0.002172 |                -0.017257 |
|          32 | GreedyLocal      |        20 |      64 |               0.121398 |             0.125452 |                -0.026216 |           0.000000 |               0.825000 |                -0.004054 |                -0.033392 |
|          64 | FixedContraction |        10 |      64 |               0.125570 |             0.125755 |                -0.000122 |           0.390625 |               0.240625 |                -0.000185 |                -0.001475 |
|          64 | FixedContraction |        20 |      64 |               0.121397 |             0.122184 |                -0.001709 |           0.218750 |               0.564063 |                -0.000787 |                -0.006486 |
|          64 | GreedyLocal      |        10 |      64 |               0.125570 |             0.127898 |                -0.012122 |           0.015625 |               0.700000 |                -0.002328 |                -0.018536 |
|          64 | GreedyLocal      |        20 |      64 |               0.121397 |             0.125310 |                -0.025303 |           0.000000 |               0.839063 |                -0.003913 |                -0.032237 |
|          96 | FixedContraction |        10 |      64 |               0.125736 |             0.125906 |                 0.000000 |           0.421875 |               0.214062 |                -0.000170 |                -0.001354 |
|          96 | FixedContraction |        20 |      64 |               0.121276 |             0.122041 |                -0.002399 |           0.250000 |               0.539844 |                -0.000765 |                -0.006306 |
|          96 | GreedyLocal      |        10 |      64 |               0.125736 |             0.127797 |                -0.014354 |           0.062500 |               0.679688 |                -0.002061 |                -0.016393 |
|          96 | GreedyLocal      |        20 |      64 |               0.121276 |             0.125385 |                -0.027514 |           0.000000 |               0.830469 |                -0.004108 |                -0.033877 |

### K=20 Decision Stages

|   LatentDim | Continuation     | Stage   |   ActionsDifferentRate |   MeanImmediateGain |   MeanTrajectoryGain |
|------------:|:-----------------|:--------|-----------------------:|--------------------:|---------------------:|
|          16 | FixedContraction | Early   |               0.549479 |           -0.005749 |            -0.000913 |
|          16 | FixedContraction | Middle  |               0.698661 |           -0.016964 |            -0.000913 |
|          16 | FixedContraction | Late    |               0.455357 |           -0.002887 |            -0.000913 |
|          16 | GreedyLocal      | Early   |               0.947917 |           -0.016042 |            -0.003601 |
|          16 | GreedyLocal      | Middle  |               0.946429 |           -0.017367 |            -0.003601 |
|          16 | GreedyLocal      | Late    |               0.645089 |           -0.007389 |            -0.003601 |
|          32 | FixedContraction | Early   |               0.570312 |           -0.007493 |            -0.000761 |
|          32 | FixedContraction | Middle  |               0.696429 |           -0.017063 |            -0.000761 |
|          32 | FixedContraction | Late    |               0.433036 |           -0.002474 |            -0.000761 |
|          32 | GreedyLocal      | Early   |               0.963542 |           -0.015527 |            -0.004054 |
|          32 | GreedyLocal      | Middle  |               0.899554 |           -0.016580 |            -0.004054 |
|          32 | GreedyLocal      | Late    |               0.631696 |           -0.008249 |            -0.004054 |
|          64 | FixedContraction | Early   |               0.544271 |           -0.008305 |            -0.000787 |
|          64 | FixedContraction | Middle  |               0.712054 |           -0.017231 |            -0.000787 |
|          64 | FixedContraction | Late    |               0.433036 |           -0.003093 |            -0.000787 |
|          64 | GreedyLocal      | Early   |               0.966146 |           -0.016886 |            -0.003913 |
|          64 | GreedyLocal      | Middle  |               0.946429 |           -0.017765 |            -0.003913 |
|          64 | GreedyLocal      | Late    |               0.622768 |           -0.006929 |            -0.003913 |
|          96 | FixedContraction | Early   |               0.539062 |           -0.006437 |            -0.000765 |
|          96 | FixedContraction | Middle  |               0.678571 |           -0.016022 |            -0.000765 |
|          96 | FixedContraction | Late    |               0.401786 |           -0.003254 |            -0.000765 |
|          96 | GreedyLocal      | Early   |               0.958333 |           -0.016235 |            -0.004108 |
|          96 | GreedyLocal      | Middle  |               0.921875 |           -0.016531 |            -0.004108 |
|          96 | GreedyLocal      | Late    |               0.629464 |           -0.008221 |            -0.004108 |

Best PCA K=20 result: PCA32 with FixedContraction, relative sequential gain -0.63%, positive-case rate 17.19%, and action-difference rate 56.64%.
LowFrequency16 remains the reference at 1.94% K=20 gain. PCA64 best gain is -0.65%. PCA64 does not clear 2.5%; the low-frequency basis is not the main headroom bottleneck, so do not start RL training and move the next benchmark diagnostic to official LNO Shallow Water.

GreedyLocal uses only the current state, the shared GT-free candidate generator, and immediate oracle reward during hypothetical continuation; it does not use a future-return selector. Actual sequential trajectories always replan at each solver step and apply bounded incremental PCA corrections.
