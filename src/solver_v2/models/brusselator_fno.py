from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralConv3d(nn.Module):
    def __init__(self, channels: int, modes_t: int, modes_x: int, modes_y: int):
        super().__init__()
        self.modes_t, self.modes_x, self.modes_y = modes_t, modes_x, modes_y
        scale = 1.0 / (channels * channels)
        shape = (channels, channels, modes_t, modes_x, modes_y)
        self.weights = nn.ParameterList([nn.Parameter(scale * torch.randn(*shape, dtype=torch.cfloat)) for _ in range(4)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, nt, nx, ny = x.shape
        x_ft = torch.fft.rfftn(x, dim=(-3, -2, -1))
        out_ft = torch.zeros(batch, channels, nt, nx, ny // 2 + 1, dtype=torch.cfloat, device=x.device)
        mt, mx, my = min(self.modes_t, nt), min(self.modes_x, nx), min(self.modes_y, ny // 2 + 1)
        quadrants = [
            (slice(0, mt), slice(0, mx), self.weights[0]),
            (slice(-mt, None), slice(0, mx), self.weights[1]),
            (slice(0, mt), slice(-mx, None), self.weights[2]),
            (slice(-mt, None), slice(-mx, None), self.weights[3]),
        ]
        for t_slice, x_slice, weight in quadrants:
            out_ft[:, :, t_slice, x_slice, :my] = torch.einsum(
                "bctxy,cotxy->botxy", x_ft[:, :, t_slice, x_slice, :my], weight[:, :, :mt, :mx, :my]
            )
        return torch.fft.irfftn(out_ft, s=(nt, nx, ny), dim=(-3, -2, -1))


class BrusselatorFNO(nn.Module):
    """Strictly an FNO baseline, not the official LNO implementation."""

    def __init__(self, width: int, modes_t: int, modes_x: int, modes_y: int, depth: int):
        super().__init__()
        self.lift = nn.Conv3d(4, width, 1)
        self.spectral = nn.ModuleList([SpectralConv3d(width, modes_t, modes_x, modes_y) for _ in range(depth)])
        self.pointwise = nn.ModuleList([nn.Conv3d(width, width, 1) for _ in range(depth)])
        self.project = nn.Sequential(nn.Conv3d(width, width, 1), nn.GELU(), nn.Conv3d(width, 1, 1))

    def forward(self, forcing: torch.Tensor) -> torch.Tensor:
        if forcing.ndim == 2:
            forcing = forcing[:, :, None, None].expand(-1, -1, 14, 14)
        batch, nt, nx, ny = forcing.shape
        t = torch.linspace(0.0, 1.0, nt, device=forcing.device).view(1, nt, 1, 1).expand(batch, -1, nx, ny)
        x = torch.linspace(0.0, 1.0, nx, device=forcing.device).view(1, 1, nx, 1).expand(batch, nt, -1, ny)
        y = torch.linspace(0.0, 1.0, ny, device=forcing.device).view(1, 1, 1, ny).expand(batch, nt, nx, -1)
        h = self.lift(torch.stack([forcing, t, x, y], dim=1))
        for spectral, pointwise in zip(self.spectral, self.pointwise):
            h = F.gelu(spectral(h) + pointwise(h))
        return self.project(h).squeeze(1)
