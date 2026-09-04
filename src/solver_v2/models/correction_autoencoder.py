from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.solver_v2.data.normalization import bounded_delta
from src.solver_v2.models.operator_blocks import FNOBlock, ScalarFiLM, coordinate_grid


class NeuralOperatorStateEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 5,
        scalar_dim: int = 4,
        width: int = 32,
        modes: int = 8,
        depth: int = 2,
        out_dim: int = 96,
    ):
        super().__init__()
        self.lift = nn.Conv2d(in_channels + 2, width, 1)
        self.film = ScalarFiLM(scalar_dim, width)
        self.blocks = nn.ModuleList([FNOBlock(width, modes, modes) for _ in range(depth)])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Linear(width + scalar_dim, out_dim),
            nn.GELU(),
            nn.LayerNorm(out_dim),
            nn.Linear(out_dim, out_dim),
            nn.GELU(),
        )

    def forward(self, fields: torch.Tensor, scalars: torch.Tensor) -> torch.Tensor:
        b, _, nx, nt = fields.shape
        x = torch.cat([fields, coordinate_grid(b, nx, nt, fields.device)], dim=1)
        h = self.film(self.lift(x), scalars)
        for block in self.blocks:
            h = block(h)
        pooled = self.pool(h).flatten(1)
        return self.head(torch.cat([pooled, scalars], dim=1))


class CorrectionEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 5,
        scalar_dim: int = 4,
        latent_dim: int = 32,
        width: int = 32,
        modes: int = 8,
        depth: int = 2,
    ):
        super().__init__()
        self.lift = nn.Conv2d(in_channels + 1 + 2, width, 1)
        self.film = ScalarFiLM(scalar_dim, width)
        self.blocks = nn.ModuleList([FNOBlock(width, modes, modes) for _ in range(depth)])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Linear(width + scalar_dim, width),
            nn.GELU(),
            nn.LayerNorm(width),
            nn.Linear(width, latent_dim),
        )

    def forward(self, fields: torch.Tensor, scalars: torch.Tensor, delta_u: torch.Tensor) -> torch.Tensor:
        if delta_u.ndim == 3:
            delta_u = delta_u.unsqueeze(1)
        b, _, nx, nt = fields.shape
        x = torch.cat([fields, delta_u, coordinate_grid(b, nx, nt, fields.device)], dim=1)
        h = self.film(self.lift(x), scalars)
        for block in self.blocks:
            h = block(h)
        pooled = self.pool(h).flatten(1)
        return self.head(torch.cat([pooled, scalars], dim=1))


class CorrectionOperatorDecoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 5,
        scalar_dim: int = 4,
        latent_dim: int = 32,
        width: int = 32,
        modes: int = 8,
        depth: int = 3,
    ):
        super().__init__()
        del in_channels
        self.latent_dim = int(latent_dim)
        self.width = int(width)
        # Only residual/source/coordinates enter the shortcut path. The main field
        # representation is seeded and modulated by z so z cannot be ignored.
        self.state_lift = nn.Conv2d(2 + 2, width, 1)
        self.z_seed = nn.Sequential(nn.Linear(latent_dim, width), nn.GELU(), nn.Linear(width, width))
        self.z_film = nn.Sequential(nn.Linear(latent_dim, 2 * width), nn.GELU(), nn.Linear(2 * width, 2 * width))
        self.film = ScalarFiLM(scalar_dim, width)
        self.blocks = nn.ModuleList([FNOBlock(width, modes, modes) for _ in range(depth)])
        self.proj = nn.Sequential(nn.Conv2d(width, width, 1), nn.GELU(), nn.Conv2d(width, 1, 1))

    def forward(self, fields: torch.Tensor, scalars: torch.Tensor, z: torch.Tensor, stats: dict, temperature: float = 1.0) -> torch.Tensor:
        b, _, nx, nt = fields.shape
        if z.ndim > 2:
            z = z.view(z.shape[0], -1)
        if z.ndim == 1:
            z = z.unsqueeze(0)
        state_cond = torch.cat([fields[:, 1:3], coordinate_grid(b, nx, nt, fields.device)], dim=1)
        state_h = self.state_lift(state_cond)
        z_h = self.z_seed(z).view(b, self.width, 1, 1).expand(b, self.width, nx, nt)
        gamma, beta = self.z_film(z).chunk(2, dim=1)
        h = 0.25 * state_h + z_h
        h = h * (1.0 + gamma[:, :, None, None]) + beta[:, :, None, None]
        h = self.film(h, scalars)
        for block in self.blocks:
            h = block(h)
        return bounded_delta(self.proj(h).squeeze(1), stats, temperature)


class CorrectionAutoencoder(nn.Module):
    def __init__(self, encoder: CorrectionEncoder, decoder: CorrectionOperatorDecoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, fields: torch.Tensor, scalars: torch.Tensor, delta_u: torch.Tensor, stats: dict, temperature: float = 1.0) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(fields, scalars, delta_u)
        recon = self.decoder(fields, scalars, z, stats, temperature)
        return recon, z


def latent_stats(z: torch.Tensor) -> dict[str, torch.Tensor]:
    return {
        "latent_mean": z.mean(dim=0).detach().cpu(),
        "latent_std": z.std(dim=0).clamp_min(1e-4).detach().cpu(),
    }
