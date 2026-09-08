from __future__ import annotations

import torch
import torch.nn as nn

from src.solver_v2.models.correction_autoencoder import NeuralOperatorStateEncoder


class OperatorQ(nn.Module):
    def __init__(
        self,
        in_channels: int = 5,
        scalar_dim: int = 4,
        width: int = 32,
        modes: int = 8,
        depth: int = 2,
        latent_dim: int = 32,
        state_dim: int = 96,
    ):
        super().__init__()
        self.encoder = NeuralOperatorStateEncoder(in_channels, scalar_dim, width, modes, depth, state_dim)
        self.action_encoder = nn.Sequential(
            nn.Linear(latent_dim, width),
            nn.GELU(),
            nn.LayerNorm(width),
            nn.Linear(width, width),
            nn.GELU(),
        )
        self.action_to_state = nn.Linear(width, state_dim)
        self.action_projection = nn.Linear(width, state_dim)
        self.head = nn.Sequential(
            nn.Linear(state_dim + width + 2 * state_dim, width),
            nn.GELU(),
            nn.LayerNorm(width),
            nn.Linear(width, width),
            nn.GELU(),
            nn.Linear(width, 1),
        )

    def forward(self, fields: torch.Tensor, scalars: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        state = self.encoder(fields, scalars)
        latent = self.action_encoder(action.reshape(action.shape[0], -1))
        action_state = self.action_to_state(latent)
        projection = self.action_projection(latent)
        interaction = torch.cat([state, latent, state * action_state, torch.abs(state - projection)], dim=1)
        return self.head(interaction).squeeze(-1)


class TwinOperatorCritic(nn.Module):
    def __init__(
        self,
        in_channels: int = 5,
        scalar_dim: int = 4,
        width: int = 32,
        modes: int = 8,
        depth: int = 2,
        latent_dim: int = 32,
        state_dim: int = 96,
    ):
        super().__init__()
        self.q1 = OperatorQ(in_channels, scalar_dim, width, modes, depth, latent_dim, state_dim)
        self.q2 = OperatorQ(in_channels, scalar_dim, width, modes, depth, latent_dim, state_dim)

    def forward(self, fields: torch.Tensor, scalars: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.q1(fields, scalars, action), self.q2(fields, scalars, action)

    def q_min(self, fields: torch.Tensor, scalars: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        q1, q2 = self.forward(fields, scalars, action)
        return torch.minimum(q1, q2)
