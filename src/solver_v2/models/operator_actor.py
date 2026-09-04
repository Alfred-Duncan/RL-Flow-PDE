from __future__ import annotations

import torch
import torch.nn as nn

from src.solver_v2.models.correction_autoencoder import NeuralOperatorStateEncoder


class DeterministicNeuralOperatorActor(nn.Module):
    """Residual-conditioned deterministic neural operator pi_theta(s) -> latent correction z."""

    def __init__(
        self,
        in_channels: int = 5,
        scalar_dim: int = 4,
        width: int = 32,
        modes: int = 8,
        depth: int = 3,
        latent_dim: int = 32,
        state_dim: int = 96,
    ):
        super().__init__()
        self.encoder = NeuralOperatorStateEncoder(in_channels, scalar_dim, width, modes, depth, state_dim)
        self.head = nn.Sequential(
            nn.Linear(state_dim, state_dim),
            nn.GELU(),
            nn.LayerNorm(state_dim),
            nn.Linear(state_dim, latent_dim),
        )

    def forward(self, fields: torch.Tensor, scalars: torch.Tensor, stats: dict | None = None, temperature: float = 1.0) -> torch.Tensor:
        del stats, temperature
        return self.head(self.encoder(fields, scalars))


class UnboundedActorAdapter(nn.Module):
    def __init__(self, actor: DeterministicNeuralOperatorActor, stats: dict, temperature: float = 1.0):
        super().__init__()
        self.actor = actor
        self.stats = stats
        self.temperature = float(temperature)

    def forward(self, fields: torch.Tensor, scalars: torch.Tensor) -> torch.Tensor:
        return self.actor(fields, scalars, self.stats, self.temperature)
