# Method

RL-Flow-PDE refines an initial reaction-diffusion solution by encoding the current solution, PDE residual, IC/BC fields, and scalar physics context with a spectral operator encoder. A conditional Flow Matching policy samples low-frequency spectral corrections. An IQL-style Twin-Q and V critic scores candidate corrections, and inference applies the highest-value correction unless its advantage falls below a validation-calibrated stop margin.
