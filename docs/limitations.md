# Limitations

The system is intentionally compact enough for a local 8 GB GPU. It uses the official LNO 2D_Reac_diffusion data when available and reports negative results when RL guidance does not beat supervised FM, LNO-only, or gradient refinement. Residual reduction and ground-truth accuracy improvement can disagree, so the residual/error quadrant table is part of the final evidence rather than a diagnostic afterthought.
