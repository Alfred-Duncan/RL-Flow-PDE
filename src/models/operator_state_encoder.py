from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralConv2d(nn.Module):
    def __init__(self, width: int, modes1: int, modes2: int):
        super().__init__()
        self.width = int(width)
        self.modes1 = int(modes1)
        self.modes2 = int(modes2)
        scale = 1.0 / max(1, width * width)
        self.weights = nn.Parameter(scale * torch.randn(width, width, modes1, modes2, dtype=torch.cfloat))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        x_ft = torch.fft.rfft2(x)
        out_ft = torch.zeros(b, c, h, w // 2 + 1, dtype=torch.cfloat, device=x.device)
        m1 = min(self.modes1, h)
        m2 = min(self.modes2, w // 2 + 1)
        out_ft[:, :, :m1, :m2] = torch.einsum("bixy,ioxy->boxy", x_ft[:, :, :m1, :m2], self.weights[:, :, :m1, :m2])
        return torch.fft.irfft2(out_ft, s=(h, w)).real


class OperatorStateEncoder(nn.Module):
    def __init__(self, in_channels: int = 4, scalar_dim: int = 8, width: int = 32, modes: int = 8, state_dim: int = 96):
        super().__init__()
        self.lift = nn.Conv2d(in_channels, width, 1)
        self.spec1 = SpectralConv2d(width, modes, modes)
        self.spec2 = SpectralConv2d(width, modes, modes)
        self.local1 = nn.Conv2d(width, width, 1)
        self.local2 = nn.Conv2d(width, width, 1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.scalar = nn.Sequential(nn.Linear(scalar_dim, width), nn.GELU(), nn.Linear(width, width))
        self.out = nn.Sequential(nn.Linear(2 * width, state_dim), nn.LayerNorm(state_dim), nn.GELU())
        self.state_dim = state_dim

    def forward(self, fields: torch.Tensor, scalars: torch.Tensor) -> torch.Tensor:
        x = self.lift(fields)
        x = F.gelu(self.spec1(x) + self.local1(x))
        x = F.gelu(self.spec2(x) + self.local2(x))
        pooled = self.pool(x).flatten(1)
        s = self.scalar(scalars)
        return self.out(torch.cat([pooled, s], dim=1))


def state_tensors(u: torch.Tensor, residual: torch.Tensor, ic: torch.Tensor, bc: torch.Tensor, scalars: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if u.ndim == 2:
        u = u.unsqueeze(0)
        residual = residual.unsqueeze(0)
        ic = ic.unsqueeze(0)
        bc = bc.unsqueeze(0)
        scalars = scalars.unsqueeze(0)
    ic_field = ic[:, :, None].expand_as(u)
    bc_field = bc
    return torch.stack([u, residual, ic_field, bc_field], dim=1), scalars
