### Main Result

Coarse: 0.740347 trajectory Relative L2.

Best non-RL: RandomMacro at 0.712588.

BeamBC: 0.739403.

RL: 0.685250.

RL gain over strongest deployable baseline: 3.84%.

Full-horizon vs immediate-only: RV-PI 0.685250; Immediate-Only PI 0.729946.

3-seed: this run reports seed 42 only. Additional seeds are run only when the seed-42 validation-selected RV-PI checkpoint provides a positive improvement.

The official operator input is called the official condition field a(x,y). Spatial refinement is GT-free at deployment; GT is used only for train returns and held-out metrics.
