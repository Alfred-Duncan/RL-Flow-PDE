from __future__ import annotations

import torch
import torch.nn as nn


class SetAwareSelector(nn.Module):
    """Scores conditional patch gains given a previously selected patch set."""

    def __init__(self, feature_dim: int, width: int = 64, heads: int = 4, layers: int = 2):
        super().__init__()
        self.field_encoder = nn.Sequential(
            nn.Conv2d(3, 24, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(24, width, 3, stride=2, padding=1), nn.GELU(),
        )
        self.feature_encoder = nn.Sequential(nn.Linear(feature_dim, width), nn.GELU(), nn.Linear(width, width))
        self.mask_embedding = nn.Embedding(2, width)
        self.context_encoder = nn.Sequential(nn.Linear(2, width), nn.GELU(), nn.Linear(width, width))
        layer = nn.TransformerEncoderLayer(d_model=width, nhead=heads, dim_feedforward=2 * width, dropout=0.0, batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=layers)
        self.score = nn.Sequential(nn.Linear(2 * width, width), nn.GELU(), nn.Linear(width, 1))

    def forward(self, fields: torch.Tensor, patch_features: torch.Tensor, selected_mask: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """fields is Bx3x32x32; patch_features and selected_mask have 64 patches."""
        spatial = self.field_encoder(fields).flatten(2).transpose(1, 2)
        patch = self.feature_encoder(patch_features) + spatial + self.mask_embedding(selected_mask.long())
        global_context = self.context_encoder(context)
        encoded = self.transformer(patch + global_context[:, None])
        pooled = encoded.mean(1, keepdim=True).expand_as(encoded)
        return self.score(torch.cat([encoded, pooled], dim=-1)).squeeze(-1)
