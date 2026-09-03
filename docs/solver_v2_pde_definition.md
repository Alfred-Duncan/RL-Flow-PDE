# Solver V2 PDE Definition

The official 2D_Reac_diffusion tensors provide source fields f(x,t), target solutions u(x,t), and one-dimensional x and t coordinate arrays. The data orientation used by the official training script is `(case, x, t)`. The bundled official arrays use 40 spatial points on x in [0, 2] and 20 temporal points on t in [0, 1].

V2 uses the reaction-diffusion residual

`R(u) = u_t - D u_xx - k u^2 - f(x,t)`

with `D = 1 - 0.95 / pi^2` and `k = 1`, matching the official note file. Boundary and initial values are taken directly from each target solution: `u[:,0]`, `u[0,:]`, and `u[-1,:]`. Every solver update is followed by hard IC/BC projection.
