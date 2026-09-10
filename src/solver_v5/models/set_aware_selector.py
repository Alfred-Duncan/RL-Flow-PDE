from __future__ import annotations

import torch
import torch.nn as nn


class SetAwareSelectorV5(nn.Module):
    """Frozen-proposal marginal-gain selector with conditional solver context."""

    def __init__(self, feature_dim: int, field_channels: int = 6, width: int = 64, heads: int = 4, layers: int = 2):
        super().__init__()
        self.field_encoder = nn.Sequential(
            nn.Conv2d(field_channels, 24, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(24, width, 3, stride=2, padding=1), nn.GELU(),
        )
        self.feature_encoder = nn.Sequential(nn.Linear(feature_dim, width), nn.GELU(), nn.Linear(width, width))
        self.mask_embedding = nn.Embedding(2, width)
        self.context_encoder = nn.Sequential(nn.Linear(3, width), nn.GELU(), nn.Linear(width, width))
        layer = nn.TransformerEncoderLayer(width, heads, 2 * width, dropout=0.0, batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=layers)
        self.score = nn.Sequential(nn.Linear(2 * width, width), nn.GELU(), nn.Linear(width, 1))

    def encode(self, fields: torch.Tensor, patch_features: torch.Tensor, selected_mask: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        spatial = self.field_encoder(fields).flatten(2).transpose(1, 2)
        token = self.feature_encoder(patch_features) + spatial + self.mask_embedding(selected_mask.long())
        return self.transformer(token + self.context_encoder(context)[:, None])

    def forward(self, fields: torch.Tensor, patch_features: torch.Tensor, selected_mask: torch.Tensor, context: torch.Tensor, return_embedding: bool = False):
        encoded = self.encode(fields, patch_features, selected_mask, context)
        pooled = encoded.mean(1, keepdim=True).expand_as(encoded)
        scores = self.score(torch.cat([encoded, pooled], dim=-1)).squeeze(-1)
        return (scores, encoded.mean(1)) if return_embedding else scores
