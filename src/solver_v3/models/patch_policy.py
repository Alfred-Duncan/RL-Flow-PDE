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
