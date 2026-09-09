from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class PCACorrectionBasis:
    """Train-fitted PCA action space with no mean-correction shortcut at inference."""

    mean_delta: torch.Tensor
    components: torch.Tensor
    latent_mean: torch.Tensor
    latent_std: torch.Tensor
    max_step_rms: float
    lower: float
    upper: float

    @property
    def max_dim(self) -> int:
        return int(self.components.shape[1])

    def decode(self, actions: torch.Tensor, latent_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
        if latent_dim > self.max_dim:
            raise ValueError(f"Requested PCA dimension {latent_dim}, but only {self.max_dim} components are available.")
        flat = actions[:, :latent_dim] @ self.components[:, :latent_dim].T
        raw = flat.reshape(-1, 39, 14, 14)
        raw_rms = torch.sqrt(torch.mean(raw.square(), dim=(1, 2, 3))).clamp_min(1e-8)
        eta = torch.minimum(torch.ones_like(raw_rms), torch.full_like(raw_rms, self.max_step_rms) / raw_rms)
        return raw * eta[:, None, None, None], eta

    def advance(self, current: torch.Tensor, actions: torch.Tensor, latent_dim: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply a bounded incremental action: u_(k+1) = u_k + eta * PCA_Decode(z)."""
        if current.ndim == 3:
            current = current.unsqueeze(0)
        delta, eta = self.decode(actions, latent_dim)
        next_u = (current + delta).clamp(self.lower, self.upper)
        correction_norm = torch.sqrt(torch.mean((next_u - current).square(), dim=(1, 2, 3)))
        return next_u, correction_norm, eta

    def fixed_contraction(self, current: torch.Tensor, baseline: torch.Tensor, retention: float) -> torch.Tensor:
        """The existing data-free reference continuation, retained for comparison."""
        return (baseline + retention * (current - baseline)).clamp(self.lower, self.upper)
