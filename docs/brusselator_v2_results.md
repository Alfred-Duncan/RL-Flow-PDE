# Brusselator V2 Results

## Protocol
The coarse FNO, local correction, selector, and BeamBC checkpoints are fixed after the V2 stability gate. Only policy optimization and evaluation use the three fixed seeds 42, 123, and 2026. Each method is evaluated on the same 100 held-out trajectories per seed.

## Three-Seed Comparison
| Method              |   MeanTrajectoryRelativeL2 |   StdTrajectoryRelativeL2 |   MeanFinalFrameRelativeL2 |   StdFinalFrameRelativeL2 |   MeanLocalCalls |
|:--------------------|---------------------------:|--------------------------:|---------------------------:|--------------------------:|-----------------:|
| BeamBC              |                   0.101283 |                  0.000000 |                   0.161804 |                  0.000000 |        76.000000 |
| CoarseOnly          |                   0.101647 |                  0.000000 |                   0.134663 |                  0.000000 |         0.000000 |
| GradientMacro       |                   0.109774 |                  0.000000 |                   0.239893 |                  0.000000 |        76.000000 |
| ImmediateOnlyPI     |                   0.101896 |                  0.001722 |                   0.140732 |                  0.018659 |        76.000000 |
| RVPI                |                   0.099542 |                  0.001997 |                   0.155273 |                  0.022721 |        76.000000 |
| RandomMacro         |                   0.105131 |                  0.000832 |                   0.156753 |                  0.023193 |        76.000000 |
| SetAwareMyopicMacro |                   0.109639 |                  0.000000 |                   0.232940 |                  0.000000 |        76.000000 |
| UniformMacro        |                   0.107448 |                  0.000000 |                   0.202439 |                  0.000000 |        76.000000 |

## Conclusion
Across the three fixed policy seeds, RVPI has the lowest mean trajectory relative L2. Its mean improvement is 2.31% versus ImmediateOnlyPI and 1.72% versus BeamBC.

The conclusion is limited to this stabilized Brusselator setup. It does not claim an advantage over the full-state oracle or imply that one-step surrogate accuracy alone is sufficient for long-horizon deployment.
