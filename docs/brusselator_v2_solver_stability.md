# Brusselator V2 Solver Stability

The V1 Brusselator FNO had moderate one-step accuracy but severe autoregressive drift over the deployed 38-step horizon. Its validation trajectory relative L2 was 8.2505, final-frame relative L2 was 84.7149, and the final prediction RMS was 79.8 times the target RMS.

V2 keeps the FNO2d architecture and data split unchanged. It replaces the V1 one-step plus scheduled two-step objective with a six-step fully autoregressive rollout loss, a 0.5-weight one-step anchor, and a small RMS explosion guard enabled because the validation diagnostic showed genuine amplitude growth.

On validation, V2 obtains one-step relative L2 of 0.3163, trajectory relative L2 of 0.0937, and final-frame relative L2 of 0.1082. The final prediction-to-target RMS ratio is 1.08. The coarse solver therefore passes the predeclared validation gate before any V2 local correction, selector, or policy is trained.

This establishes the relevant methodological point: one-step surrogate accuracy does not imply long-time learned-PDE reliability. Closed-loop error propagation and distribution shift must be controlled before evaluating adaptive correction timing.
