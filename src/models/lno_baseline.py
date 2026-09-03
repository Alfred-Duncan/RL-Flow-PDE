from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.operator_state_encoder import SpectralConv2d


class SmallLNOBaseline(nn.Module):
    """Compact LNO/FNO-style initial solution model for the official reaction-diffusion grid."""

    def __init__(self, width: int = 32, modes: int = 8):
        super().__init__()
        self.fc0 = nn.Linear(3, width)
        self.spec1 = SpectralConv2d(width, modes, modes)
        self.spec2 = SpectralConv2d(width, modes, modes)
        self.w1 = nn.Conv2d(width, width, 1)
        self.w2 = nn.Conv2d(width, width, 1)
        self.fc1 = nn.Linear(width, width)
        self.fc2 = nn.Linear(width, 1)

    def forward(self, source: torch.Tensor) -> torch.Tensor:
        if source.ndim == 3:
            source = source.unsqueeze(-1)
        b, nx, nt, _ = source.shape
        x = torch.linspace(0, 1, nx, device=source.device).view(1, nx, 1, 1).expand(b, nx, nt, 1)
        t = torch.linspace(0, 1, nt, device=source.device).view(1, 1, nt, 1).expand(b, nx, nt, 1)
        y = self.fc0(torch.cat([source, x, t], dim=-1)).permute(0, 3, 1, 2)
        y = F.gelu(self.spec1(y) + self.w1(y))
        y = F.gelu(self.spec2(y) + self.w2(y))
        y = self.fc2(F.gelu(self.fc1(y.permute(0, 2, 3, 1)))).squeeze(-1)
        return y
