from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.modes1 = int(modes1)
        self.modes2 = int(modes2)
        scale = 1.0 / max(1, in_channels * out_channels)
        self.weight = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        x_ft = torch.fft.rfft2(x)
        out_ft = torch.zeros(b, self.out_channels, h, w // 2 + 1, dtype=torch.cfloat, device=x.device)
        m1 = min(self.modes1, h)
        m2 = min(self.modes2, w // 2 + 1)
        out_ft[:, :, :m1, :m2] = torch.einsum("bixy,ioxy->boxy", x_ft[:, :, :m1, :m2], self.weight[:, :, :m1, :m2])
        return torch.fft.irfft2(out_ft, s=(h, w)).real


class FNOBlock(nn.Module):
    def __init__(self, width: int, modes1: int, modes2: int):
        super().__init__()
        self.spectral = SpectralConv2d(width, width, modes1, modes2)
        self.local = nn.Conv2d(width, width, 1)
        self.norm = nn.GroupNorm(1, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.norm(self.spectral(x) + self.local(x)))


def coordinate_grid(batch: int, nx: int, nt: int, device: torch.device) -> torch.Tensor:
    gx = torch.linspace(0.0, 1.0, nx, device=device).view(1, 1, nx, 1).expand(batch, 1, nx, nt)
    gt = torch.linspace(0.0, 1.0, nt, device=device).view(1, 1, 1, nt).expand(batch, 1, nx, nt)
    return torch.cat([gx, gt], dim=1)


class ScalarFiLM(nn.Module):
    def __init__(self, scalar_dim: int, width: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(scalar_dim, width), nn.GELU(), nn.Linear(width, 2 * width))

    def forward(self, x: torch.Tensor, scalars: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.net(scalars).chunk(2, dim=1)
        return x * (1.0 + 0.1 * gamma[:, :, None, None]) + 0.1 * beta[:, :, None, None]

