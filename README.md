# RL-Flow-PDE

Residual-conditioned advantage-weighted Flow Matching for sequential PDE refinement.

Default paper-mode uses the official Laplace Neural Operator `2D_Reac_diffusion` data file when it is present under:

`external/Laplace-Neural-Operator/2D_Reac_diffusion/Data/data.mat`

Run:

```bash
conda run --no-capture-output -n base python -u run_pipeline.py --mode paper
```

Outputs are written to `results/tables`, `results/figures`, and `docs`.
