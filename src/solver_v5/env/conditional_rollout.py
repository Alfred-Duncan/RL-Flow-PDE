from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


def relative_l2(value: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.linalg.vector_norm((value - target).flatten(1), dim=1) / torch.linalg.vector_norm(target.flatten(1), dim=1).clamp_min(1e-8)


class ConditionalRefiner:
    """Condition/history global transition and frozen local correction proposals."""

    def __init__(self, coarse_model: torch.nn.Module, local_model: torch.nn.Module, core: int, halo: int, state_mean: torch.Tensor, state_std: torch.Tensor, condition_mean: torch.Tensor, condition_std: torch.Tensor):
        self.coarse_model, self.local_model = coarse_model, local_model
        self.core, self.halo = core, halo
        self.state_mean, self.state_std = state_mean, state_std
        self.condition_mean, self.condition_std = condition_mean, condition_std

    def _state_norm(self, value: torch.Tensor) -> torch.Tensor:
        return (value - self.state_mean.to(value.device)) / self.state_std.to(value.device)

    def _condition_norm(self, value: torch.Tensor) -> torch.Tensor:
        return (value - self.condition_mean.to(value.device)) / self.condition_std.to(value.device)

    @torch.no_grad()
    def coarse_next(self, condition: torch.Tensor, previous: torch.Tensor, current: torch.Tensor, time_fraction: float) -> tuple[torch.Tensor, torch.Tensor]:
        size = (64, 64)
        condition64 = F.interpolate(condition, size=size, mode="bilinear", align_corners=False)
        previous64 = F.interpolate(previous, size=size, mode="bilinear", align_corners=False)
        current64 = F.interpolate(current, size=size, mode="bilinear", align_corners=False)
        time = torch.full_like(current64[:, :1], float(time_fraction))
        fields = torch.cat([self._condition_norm(condition64), self._state_norm(previous64), self._state_norm(current64), self._state_norm(current64 - previous64), time], dim=1)
        coarse = self.coarse_model(fields) * self.state_std.to(current.device) + self.state_mean.to(current.device)
        return coarse, F.interpolate(coarse, size=(256, 256), mode="bilinear", align_corners=False)

    @staticmethod
    def _window(size: int, halo: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        ramp = torch.sin(torch.linspace(0.0, torch.pi / 2.0, halo + 1, device=device, dtype=dtype))[1:]
        axis = torch.cat([ramp, torch.ones(size - 2 * halo, device=device, dtype=dtype), ramp.flip(0)])
        return axis[:, None] * axis[None, :]

    def patch_bounds(self, patch: int) -> tuple[int, int, int, int]:
        row, col = divmod(patch, 8)
        return row * self.core - self.halo, (row + 1) * self.core + self.halo, col * self.core - self.halo, (col + 1) * self.core + self.halo

    def extract_patch(self, value: torch.Tensor, patch: int) -> torch.Tensor:
        top, bottom, left, right = self.patch_bounds(patch)
        padded = F.pad(value, (self.halo, self.halo, self.halo, self.halo), mode="reflect")
        return padded[:, :, top + self.halo:bottom + self.halo, left + self.halo:right + self.halo]

    def local_inputs(self, condition: torch.Tensor, previous: torch.Tensor, current: torch.Tensor, provisional: torch.Tensor, patches: list[int], time_fraction: float) -> torch.Tensor:
        rows = []
        for patch in patches:
            a, prev, now, nxt = (self.extract_patch(value, patch) for value in (condition, previous, current, provisional))
            time = torch.full_like(nxt[:, :1], float(time_fraction))
            rows.append(torch.cat([a, prev, now, nxt, nxt - now, now - prev, time], dim=1))
        return torch.cat(rows)

    def correction_canvases(self, provisional: torch.Tensor, corrections: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        contributions = torch.zeros((64, *provisional.shape[1:]), device=provisional.device, dtype=provisional.dtype)
        weights = torch.zeros((64, 1, *provisional.shape[-2:]), device=provisional.device, dtype=provisional.dtype)
        for patch, correction in enumerate(corrections.split(1, 0)):
            top, bottom, left, right = self.patch_bounds(patch)
            r0, r1, c0, c1 = max(top, 0), min(bottom, 256), max(left, 0), min(right, 256)
            rs, cs = slice(r0 - top, r1 - top), slice(c0 - left, c1 - left)
            window = self._window(bottom - top, self.halo, provisional.device, provisional.dtype)[None, None]
            contributions[patch:patch + 1, :, r0:r1, c0:c1] += correction[:, :, rs, cs] * window[:, :, rs, cs]
            weights[patch:patch + 1, :, r0:r1, c0:c1] += window[:, :, rs, cs]
        return contributions, weights


@dataclass
class FrozenConditionalBundle:
    condition: torch.Tensor
    previous: torch.Tensor
    current: torch.Tensor
    provisional: torch.Tensor
    local_inputs: torch.Tensor
    corrections: torch.Tensor
    contributions: torch.Tensor
    weights: torch.Tensor
    refiner: ConditionalRefiner

    @classmethod
    @torch.no_grad()
    def create(cls, refiner: ConditionalRefiner, condition: torch.Tensor, previous: torch.Tensor, current: torch.Tensor, time_fraction: float) -> "FrozenConditionalBundle":
        _, provisional = refiner.coarse_next(condition, previous, current, time_fraction)
        inputs = refiner.local_inputs(condition, previous, current, provisional, list(range(64)), time_fraction)
        corrections = refiner.local_model(inputs)
        contributions, weights = refiner.correction_canvases(provisional, corrections)
        return cls(condition, previous, current, provisional, inputs, corrections, contributions, weights, refiner)

    def apply_set(self, selected: list[int]) -> torch.Tensor:
        if not selected:
            return self.provisional
        ids = torch.tensor(selected, device=self.provisional.device)
        return self.provisional + self.contributions[ids].sum(0, keepdim=True) / self.weights[ids].sum(0, keepdim=True).clamp_min(1e-6)

    def candidate_fields(self, selected: list[int]) -> torch.Tensor:
        if selected:
            ids = torch.tensor(selected, device=self.provisional.device)
            value, weight = self.contributions[ids].sum(0, keepdim=True), self.weights[ids].sum(0, keepdim=True)
        else:
            value, weight = torch.zeros_like(self.provisional), torch.zeros_like(self.provisional[:, :1])
        return self.provisional + (value + self.contributions) / (weight + self.weights).clamp_min(1e-6)
