from __future__ import annotations

import torch
import torch.nn as nn


class PatchUtilityNet(nn.Module):
    """GT-free utility predictor used for the supervised myopic allocation baseline."""

    def __init__(self, feature_dim: int):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(feature_dim, 96), nn.GELU(), nn.Linear(96, 48), nn.GELU(), nn.Linear(48, 1))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(-1)


class AdaptivePatchPolicy(nn.Module):
    """Discrete patch/advance actor-critic with a frozen supervised utility prior."""

    def __init__(self, feature_dim: int, utility_prior: PatchUtilityNet):
        super().__init__()
        self.utility_prior = utility_prior
        for parameter in self.utility_prior.parameters():
            parameter.requires_grad_(False)
        self.field_encoder = nn.Sequential(
            nn.Conv2d(2, 16, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(16, 16, 3, stride=2, padding=1), nn.GELU(),
        )
        self.patch_head = nn.Sequential(nn.Linear(feature_dim + 16 + 16 + 3, 96), nn.GELU(), nn.Linear(96, 1))
        self.advance_head = nn.Sequential(nn.Linear(16 + 3, 48), nn.GELU(), nn.Linear(48, 1))
        self.value_head = nn.Sequential(nn.Linear(16 + 3, 48), nn.GELU(), nn.Linear(48, 1))

    def forward(self, field: torch.Tensor, patch_features: torch.Tensor, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # field: (batch, 2, 32, 32), patch_features: (batch, 64, features).
        spatial = self.field_encoder(field).flatten(2).transpose(1, 2)
        pooled = spatial.mean(1)
        global_features = torch.cat([pooled, context], dim=1)
        shared = global_features[:, None].expand(-1, 64, -1)
        residual = self.patch_head(torch.cat([patch_features, spatial, shared], dim=-1)).squeeze(-1)
        utility_logits = self.utility_prior(patch_features.flatten(0, 1)).view(field.shape[0], 64)
        patch_logits = utility_logits + residual
        advance = self.advance_head(global_features)
        return torch.cat([patch_logits, advance], dim=1), self.value_head(global_features).squeeze(-1)
