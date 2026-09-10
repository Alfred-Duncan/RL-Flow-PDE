from __future__ import annotations

import torch
import torch.nn as nn


class MacroPolicy(nn.Module):
    """Four-action policy for per-transition refinement counts {0,1,2,4}."""

    def __init__(self, selector_width: int = 64, field_channels: int = 6):
        super().__init__()
        self.field_encoder = nn.Sequential(
            nn.Conv2d(field_channels, 24, 5, stride=2, padding=2), nn.GELU(),
            nn.Conv2d(24, 48, 3, stride=2, padding=1), nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(
            nn.Linear(48 + 5 + selector_width + 3, 128), nn.GELU(),
            nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 4),
        )

    def forward(self, fields: torch.Tensor, selector_stats: torch.Tensor, selector_embedding: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        visual = self.field_encoder(fields).flatten(1)
        return self.head(torch.cat([visual, selector_stats, selector_embedding, context], dim=1))
