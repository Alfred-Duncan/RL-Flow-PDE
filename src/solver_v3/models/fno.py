from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralConv2d(nn.Module):
    def __init__(self, channels: int, modes: int):
        super().__init__()
        self.modes = modes
        scale = 1.0 / (channels * channels)
        shape = (channels, channels, modes, modes)
        self.weights = nn.ParameterList([nn.Parameter(scale * torch.randn(*shape, dtype=torch.cfloat)) for _ in range(2)])

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        batch, channels, nx, ny = value.shape
        transform = torch.fft.rfft2(value)
        output = torch.zeros(batch, channels, nx, ny // 2 + 1, dtype=torch.cfloat, device=value.device)
        mx, my = min(self.modes, nx), min(self.modes, ny // 2 + 1)
        output[:, :, :mx, :my] = torch.einsum("bcxy,coxy->boxy", transform[:, :, :mx, :my], self.weights[0][:, :, :mx, :my])
        output[:, :, -mx:, :my] = torch.einsum("bcxy,coxy->boxy", transform[:, :, -mx:, :my], self.weights[1][:, :, :mx, :my])
        return torch.fft.irfft2(output, s=(nx, ny))


class FNO2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, width: int, modes: int, depth: int):
        super().__init__()
        self.lift = nn.Conv2d(in_channels + 2, width, 1)
        self.spectral = nn.ModuleList([SpectralConv2d(width, modes) for _ in range(depth)])
        self.pointwise = nn.ModuleList([nn.Conv2d(width, width, 1) for _ in range(depth)])
        self.project = nn.Sequential(nn.Conv2d(width, width, 1), nn.GELU(), nn.Conv2d(width, out_channels, 1))

    def forward(self, fields: torch.Tensor) -> torch.Tensor:
        batch, _, nx, ny = fields.shape
        x = torch.linspace(-1.0, 1.0, nx, device=fields.device, dtype=fields.dtype).view(1, 1, nx, 1).expand(batch, -1, -1, ny)
        y = torch.linspace(-1.0, 1.0, ny, device=fields.device, dtype=fields.dtype).view(1, 1, 1, ny).expand(batch, -1, nx, -1)
        hidden = self.lift(torch.cat([fields, x, y], dim=1))
        for spectral, pointwise in zip(self.spectral, self.pointwise):
            hidden = F.gelu(spectral(hidden) + pointwise(hidden))
        return self.project(hidden)
