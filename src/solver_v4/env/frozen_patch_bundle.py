from __future__ import annotations

from dataclasses import dataclass

import torch

from src.solver_v3.env.hybrid_rollout import HybridRefiner


@dataclass
class FrozenPatchBundle:
    """All local proposals generated once from one coarse physical transition."""

    current: torch.Tensor
    provisional: torch.Tensor
    patch_inputs: torch.Tensor
    corrections: torch.Tensor
    contributions: torch.Tensor
    weights: torch.Tensor
    refiner: HybridRefiner

    @classmethod
    @torch.no_grad()
    def create(cls, refiner: HybridRefiner, current: torch.Tensor, time_fraction: float) -> "FrozenPatchBundle":
        _, provisional = refiner.coarse_next(current, time_fraction)
        patches = list(range(64))
        patch_inputs = refiner.patch_inputs(current, provisional, patches, time_fraction)
        corrections = refiner.local_model(patch_inputs)
        contributions, weights = refiner.correction_canvases(provisional, patches, corrections)
        return cls(current, provisional, patch_inputs, corrections, contributions, weights, refiner)

    def apply_set(self, selected_patches: list[int]) -> torch.Tensor:
        """Apply a selected proposal set once with the canonical normalized blend."""
        if not selected_patches:
            return self.provisional
        selected = torch.as_tensor(selected_patches, device=self.provisional.device, dtype=torch.long)
        output = self.contributions[selected].sum(0, keepdim=True)
        weights = self.weights[selected].sum(0, keepdim=True)
        return self.provisional + output / weights.clamp_min(1e-6)

    def apply_single_from_base(self, base_set: list[int], patch: int) -> torch.Tensor:
        return self.apply_set(base_set + [patch])

    def candidate_fields(self, selected_patches: list[int]) -> torch.Tensor:
        """All 64 set-union candidates without recomputing local proposals."""
        if selected_patches:
            selected = torch.as_tensor(selected_patches, device=self.provisional.device, dtype=torch.long)
            output = self.contributions[selected].sum(0, keepdim=True)
            weights = self.weights[selected].sum(0, keepdim=True)
        else:
            output = torch.zeros_like(self.provisional)
            weights = torch.zeros_like(self.provisional[:, :1])
        return self.provisional + (output + self.contributions) / (weights + self.weights).clamp_min(1e-6)
