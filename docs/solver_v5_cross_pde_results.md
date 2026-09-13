# Cross-PDE Result

## Shallow Water
Frozen three-seed Shallow-Water V5 results are retained without retraining.

## Brusselator
Official data: a forcing-driven 2D field trajectory with 39 physical frames at 28x28 and one state channel. The adopted formulation has no separate static condition channel; the known time-varying scalar forcing is supplied at every transition.

| Method              |   MeanTrajectoryRelativeL2 |   StdTrajectoryRelativeL2 |   MeanFinalFrameRelativeL2 |   StdFinalFrameRelativeL2 |   MeanLocalCalls |
|:--------------------|---------------------------:|--------------------------:|---------------------------:|--------------------------:|-----------------:|
| BeamBC              |                  0.101283  |               0           |                   0.161804 |                 0         |               76 |
| CoarseOnly          |                  0.101647  |               0           |                   0.134663 |                 0         |                0 |
| GradientMacro       |                  0.109774  |               0           |                   0.239893 |                 0         |               76 |
| ImmediateOnlyPI     |                  0.101896  |               0.00172182  |                   0.140732 |                 0.0186595 |               76 |
| RVPI                |                  0.0995421 |               0.00199686  |                   0.155273 |                 0.0227211 |               76 |
| RandomMacro         |                  0.105131  |               0.000832046 |                   0.156753 |                 0.0231933 |               76 |
| SetAwareMyopicMacro |                  0.109639  |               0           |                   0.23294  |                 0         |               76 |
| UniformMacro        |                  0.107448  |               0           |                   0.202439 |                 0         |               76 |

RandomMacro schedule distribution: mean=0.103539, std=0.001663, median=0.103391, best=0.100604, worst=0.106951.

## Cross-PDE conclusion
Full-horizon rollout-based policy improvement outperformed immediate-only refinement policy improvement on the evaluated Brusselator seeds.
