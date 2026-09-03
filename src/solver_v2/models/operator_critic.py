from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.solver_v2.models.operator_blocks import FNOBlock, coordinate_grid


class OperatorQ(nn.Module):
    def __init__(self, in_channels: int = 4, scalar_dim: int = 4, width: int = 32, modes: int = 8, depth: int = 2):
        super().__init__()
        self.lift = nn.Conv2d(in_channels + 2, width, 1)
        self.blocks = nn.ModuleList([FNOBlock(width, modes, modes) for _ in range(depth)])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(nn.Linear(width + scalar_dim, width), nn.GELU(), nn.Linear(width, width), nn.GELU(), nn.Linear(width, 1))

    def forward(self, fields: torch.Tensor, scalars: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        # fields: [u_t, residual_t, source, ic_field, bc_field]; critic uses [u, residual, source, action].
        x = torch.stack([fields[:, 0], fields[:, 1], fields[:, 2], action], dim=1)
        b, _, nx, nt = x.shape
        x = torch.cat([x, coordinate_grid(b, nx, nt, x.device)], dim=1)
        h = self.lift(x)
        for block in self.blocks:
            h = block(h)
        pooled = self.pool(h).flatten(1)
        return self.head(torch.cat([pooled, scalars], dim=1)).squeeze(-1)


class TwinOperatorCritic(nn.Module):
    def __init__(self, in_channels: int = 4, scalar_dim: int = 4, width: int = 32, modes: int = 8, depth: int = 2):
        super().__init__()
        self.q1 = OperatorQ(in_channels, scalar_dim, width, modes, depth)
        self.q2 = OperatorQ(in_channels, scalar_dim, width, modes, depth)

    def forward(self, fields: torch.Tensor, scalars: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.q1(fields, scalars, action), self.q2(fields, scalars, action)

    def q_min(self, fields: torch.Tensor, scalars: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        q1, q2 = self.forward(fields, scalars, action)
        return torch.minimum(q1, q2)
