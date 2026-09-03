from __future__ import annotations

import time

import torch
import torch.nn as nn
import torch.nn.functional as F


def gradient_residual_refine(env, u: torch.Tensor, steps: int, lr: float = 0.02) -> tuple[torch.Tensor, float, list[float]]:
    start = time.perf_counter()
    out = u.detach().clone()
    energies: list[float] = []
    for _ in range(steps):
        var = out.detach().clone().requires_grad_(True)
        energy = env.physics_energy(var)["energy"]
        grad = torch.autograd.grad(energy, var)[0]
        out = env.project(var - lr * grad / grad.abs().mean().clamp_min(1e-4)).detach()
        energies.append(float(energy.detach().cpu()))
    return out, time.perf_counter() - start, energies


class DeterministicCorrection(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden: int = 96):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_dim, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, action_dim))

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


def pinn_style_refine(env, u: torch.Tensor, steps: int, lr: float = 0.03, anchor: float = 1.0) -> tuple[torch.Tensor, float, list[float]]:
    start = time.perf_counter()
    delta = torch.zeros_like(u, requires_grad=True)
    opt = torch.optim.Adam([delta], lr=lr)
    energies: list[float] = []
    for _ in range(steps):
        cand = env.project(u + 0.1 * torch.tanh(delta))
        e = env.physics_energy(cand)
        loss = e["energy"] + anchor * (0.1 * torch.tanh(delta)).pow(2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        energies.append(float(e["energy"].detach().cpu()))
    return env.project(u + 0.1 * torch.tanh(delta.detach())), time.perf_counter() - start, energies
