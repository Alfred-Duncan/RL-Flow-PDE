from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.solver_v2.models.operator_blocks import FNOBlock, coordinate_grid


class FNOInitializer(nn.Module):
    """FNO-style initializer G_psi(f, grid) -> u_0."""

    def __init__(self, width: int = 32, modes: int = 8, depth: int = 3):
        super().__init__()
        self.lift = nn.Conv2d(3, width, 1)
        self.blocks = nn.ModuleList([FNOBlock(width, modes, modes) for _ in range(depth)])
        self.proj = nn.Sequential(nn.Conv2d(width, width, 1), nn.GELU(), nn.Conv2d(width, 1, 1))

    def forward(self, source: torch.Tensor) -> torch.Tensor:
        if source.ndim == 3:
            source = source.unsqueeze(1)
        b, _, nx, nt = source.shape
        x = torch.cat([source, coordinate_grid(b, nx, nt, source.device)], dim=1)
        x = self.lift(x)
        for block in self.blocks:
            x = block(x)
        return self.proj(x).squeeze(1)


def initializer_loss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred, gt) + 0.1 * F.l1_loss(pred[:, :, 0], gt[:, :, 0])

