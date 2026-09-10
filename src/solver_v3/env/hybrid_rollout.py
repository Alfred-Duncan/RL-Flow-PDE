from __future__ import annotations

import torch
import torch.nn.functional as F


def relative_l2(value: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.linalg.vector_norm((value - target).flatten(1), dim=1) / torch.linalg.vector_norm(target.flatten(1), dim=1).clamp_min(1e-8)


class HybridRefiner:
    """Closed-loop coarse transition plus shared local neural-operator patch refinement."""

    def __init__(self, coarse_model: torch.nn.Module, local_model: torch.nn.Module, stride: int, core: int, halo: int, mean: torch.Tensor | None = None, std: torch.Tensor | None = None):
        self.coarse_model, self.local_model = coarse_model, local_model
        self.stride, self.core, self.halo = stride, core, halo
        self.mean, self.std = mean, std

    def coarse_next(self, native_current: torch.Tensor, time_fraction: float) -> tuple[torch.Tensor, torch.Tensor]:
        coarse_current = F.interpolate(native_current, size=(64, 64), mode="bilinear", align_corners=False)
        time = torch.full_like(coarse_current[:, :1], float(time_fraction))
        model_current = coarse_current if self.mean is None else (coarse_current - self.mean.to(native_current.device)) / self.std.to(native_current.device)
        coarse_next = self.coarse_model(torch.cat([model_current, time], dim=1))
        if self.mean is not None:
            coarse_next = coarse_next * self.std.to(native_current.device) + self.mean.to(native_current.device)
        provisional = F.interpolate(coarse_next, size=(256, 256), mode="bilinear", align_corners=False)
        return coarse_next, provisional

    @staticmethod
    def _flat_cosine_window(size: int, halo: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        ramp = torch.sin(torch.linspace(0.0, torch.pi / 2.0, halo + 1, device=device, dtype=dtype))[1:]
        one = torch.ones(size - 2 * halo, device=device, dtype=dtype)
        axis = torch.cat([ramp, one, ramp.flip(0)])
        return axis[:, None] * axis[None, :]

    def patch_bounds(self, patch: int) -> tuple[int, int, int, int]:
        row, col = divmod(patch, 8)
        return row * self.core - self.halo, (row + 1) * self.core + self.halo, col * self.core - self.halo, (col + 1) * self.core + self.halo

    def extract_patch(self, value: torch.Tensor, patch: int) -> torch.Tensor:
        top, bottom, left, right = self.patch_bounds(patch)
        padded = F.pad(value, (self.halo, self.halo, self.halo, self.halo), mode="reflect")
        return padded[:, :, top + self.halo : bottom + self.halo, left + self.halo : right + self.halo]

    def patch_inputs(self, current: torch.Tensor, provisional: torch.Tensor, patches: list[int], time_fraction: float) -> torch.Tensor:
        """Build local-operator inputs for a set of patches in one batched call."""
        coarse_current = F.interpolate(current, size=(64, 64), mode="bilinear", align_corners=False)
        coarse_current = F.interpolate(coarse_current, size=(256, 256), mode="bilinear", align_corners=False)
        inputs = []
        for patch in patches:
            current_patch, provisional_patch = self.extract_patch(current, patch), self.extract_patch(provisional, patch)
            coarse_patch = self.extract_patch(coarse_current, patch)
            difference = provisional_patch - current_patch
            time = torch.full_like(provisional_patch[:, :1], float(time_fraction))
            inputs.append(torch.cat([current_patch, provisional_patch, coarse_patch, difference, time], dim=1))
        return torch.cat(inputs, dim=0)

    def blend_corrections(self, provisional: torch.Tensor, patches: list[int], corrections: torch.Tensor) -> torch.Tensor:
        """Cosine-blend batched local corrections into their native patch locations."""
        if not patches:
            return provisional
        output, weights = torch.zeros_like(provisional), torch.zeros_like(provisional[:, :1])
        for patch, correction in zip(patches, corrections.split(1, dim=0)):
            top, bottom, left, right = self.patch_bounds(patch)
            window = self._flat_cosine_window(bottom - top, self.halo, provisional.device, provisional.dtype)[None, None]
            # Reflection is only an input-boundary rule.  Corrections outside the
            # physical domain are discarded, so boundary pixels never receive a
            # duplicate reflected contribution during blending.
            row_start, row_end = max(top, 0), min(bottom, provisional.shape[-2])
            col_start, col_end = max(left, 0), min(right, provisional.shape[-1])
            patch_rows = slice(row_start - top, row_end - top)
            patch_cols = slice(col_start - left, col_end - left)
            output[:, :, row_start:row_end, col_start:col_end] += correction[:, :, patch_rows, patch_cols] * window[:, :, patch_rows, patch_cols]
            weights[:, :, row_start:row_end, col_start:col_end] += window[:, :, patch_rows, patch_cols]
        return provisional + output / weights.clamp_min(1e-6)

    def apply_patches(self, current: torch.Tensor, provisional: torch.Tensor, patches: list[int], time_fraction: float) -> torch.Tensor:
        if not patches:
            return provisional
        corrections = self.local_model(self.patch_inputs(current, provisional, patches, time_fraction))
        return self.blend_corrections(provisional, patches, corrections)
