from __future__ import annotations

import torch
import torch.nn as nn

from src.solver_v2.data.normalization import bounded_delta
from src.solver_v2.models.operator_blocks import FNOBlock, ScalarFiLM, coordinate_grid


class DeterministicNeuralOperatorActor(nn.Module):
    """Residual-conditioned deterministic neural operator pi_theta(s) -> delta_u."""

    def __init__(self, in_channels: int = 5, scalar_dim: int = 4, width: int = 32, modes: int = 8, depth: int = 3):
        super().__init__()
        self.lift = nn.Conv2d(in_channels + 2, width, 1)
        self.film = ScalarFiLM(scalar_dim, width)
        self.blocks = nn.ModuleList([FNOBlock(width, modes, modes) for _ in range(depth)])
        self.proj = nn.Sequential(nn.Conv2d(width, width, 1), nn.GELU(), nn.Conv2d(width, 1, 1))

    def raw(self, fields: torch.Tensor, scalars: torch.Tensor) -> torch.Tensor:
        b, _, nx, nt = fields.shape
        x = torch.cat([fields, coordinate_grid(b, nx, nt, fields.device)], dim=1)
        x = self.film(self.lift(x), scalars)
        for block in self.blocks:
            x = block(x)
        return self.proj(x).squeeze(1)

    def forward(self, fields: torch.Tensor, scalars: torch.Tensor, stats: dict, temperature: float = 1.0) -> torch.Tensor:
        return bounded_delta(self.raw(fields, scalars), stats, temperature)


class UnboundedActorAdapter(nn.Module):
    def __init__(self, actor: DeterministicNeuralOperatorActor, stats: dict, temperature: float = 1.0):
        super().__init__()
        self.actor = actor
        self.stats = stats
        self.temperature = float(temperature)

    def forward(self, fields: torch.Tensor, scalars: torch.Tensor) -> torch.Tensor:
        return self.actor(fields, scalars, self.stats, self.temperature)

