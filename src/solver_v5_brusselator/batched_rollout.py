from __future__ import annotations

from dataclasses import dataclass

import torch

from src.solver_v5_brusselator.env import BrusselatorRefiner


@dataclass
class BatchedFrozenBundle:
    """Batched counterpart to ``FrozenBundle`` used only by rollout evaluation.

    The established single-trajectory implementation is intentionally left
    untouched.  Inputs are ordered patch-major then sample-major, matching the
    legacy order exactly when the batch size is one.
    """

    force: torch.Tensor
    previous: torch.Tensor
    current: torch.Tensor
    provisional: torch.Tensor
    corrections: torch.Tensor
    contributions: torch.Tensor
    weights: torch.Tensor
    refiner: BrusselatorRefiner

    @classmethod
    @torch.no_grad()
    def create(
        cls,
        refiner: BrusselatorRefiner,
        force: torch.Tensor,
        previous: torch.Tensor,
        current: torch.Tensor,
        time: float | torch.Tensor,
    ) -> "BatchedFrozenBundle":
        batch = current.shape[0]
        device = current.device
        force = force.to(device).reshape(batch)
        if not torch.is_tensor(time):
            time = torch.full((batch,), float(time), device=device, dtype=current.dtype)
        else:
            time = time.to(device=device, dtype=current.dtype).reshape(batch)

        normalized_force = ((force - refiner.force_mean.to(device)) / refiner.force_std.to(device))
        force_field = normalized_force[:, None, None, None].expand_as(current[:, :1])
        time_field = time[:, None, None, None].expand_as(current[:, :1])
        provisional = refiner.coarse(
            torch.cat([force_field, refiner.norm(previous), refiner.norm(current), refiner.norm(current - previous), time_field], dim=1)
        ) * refiner.std.to(device) + refiner.mean.to(device)
        provisional = refiner.stabilize(provisional)

        patch_rows = []
        for patch in range(refiner.grid * refiner.grid):
            prev_patch, current_patch, provisional_patch = (
                refiner.extract(value, patch) for value in (previous, current, provisional)
            )
            patch_rows.append(
                torch.cat(
                    [
                        force[:, None, None, None].expand_as(provisional_patch[:, :1]),
                        prev_patch,
                        current_patch,
                        provisional_patch,
                        current_patch - prev_patch,
                        provisional_patch - current_patch,
                        time[:, None, None, None].expand_as(provisional_patch[:, :1]),
                    ],
                    dim=1,
                )
            )
        local_inputs = torch.cat(patch_rows, dim=0)
        raw_corrections = torch.nan_to_num(refiner.local(local_inputs))
        patches = refiner.grid * refiner.grid
        corrections = raw_corrections.reshape(patches, batch, *raw_corrections.shape[1:]).permute(1, 0, 2, 3, 4).contiguous()

        height, width = provisional.shape[-2:]
        contributions = torch.zeros((batch, patches, *provisional.shape[1:]), device=device, dtype=provisional.dtype)
        weights = torch.zeros((batch, patches, 1, height, width), device=device, dtype=provisional.dtype)
        for patch in range(patches):
            top, bottom, left, right = refiner.bounds(patch)
            row_start, row_end = max(top, 0), min(bottom, height)
            col_start, col_end = max(left, 0), min(right, width)
            row_slice, col_slice = slice(row_start - top, row_end - top), slice(col_start - left, col_end - left)
            window = refiner.window(bottom - top, device, provisional.dtype)[None, None]
            contributions[:, patch, :, row_start:row_end, col_start:col_end] += corrections[:, patch, :, row_slice, col_slice] * window[:, :, row_slice, col_slice]
            weights[:, patch, :, row_start:row_end, col_start:col_end] += window[:, :, row_slice, col_slice]
        return cls(force, previous, current, provisional, corrections, contributions, weights, refiner)

    def apply_sets(self, selected: list[list[int]]) -> torch.Tensor:
        """Apply a separate selected patch set to every element in the batch."""
        batch, patches = self.contributions.shape[:2]
        if len(selected) != batch:
            raise ValueError(f"Expected {batch} selected sets, got {len(selected)}")
        mask = torch.zeros((batch, patches), device=self.provisional.device, dtype=self.provisional.dtype)
        for row, patch_ids in enumerate(selected):
            if patch_ids:
                mask[row, torch.as_tensor(patch_ids, device=mask.device)] = 1.0
        contribution = (self.contributions * mask[:, :, None, None, None]).sum(1)
        weight = (self.weights * mask[:, :, None, None, None]).sum(1)
        return self.refiner.stabilize(self.provisional + contribution / weight.clamp_min(1e-6))


def selector_fields_batch(bundle: BatchedFrozenBundle) -> torch.Tensor:
    force = bundle.force[:, None, None, None].expand_as(bundle.current[:, :1])
    return torch.cat([force, bundle.previous, bundle.current, bundle.provisional, bundle.current - bundle.previous, bundle.provisional - bundle.current], dim=1)


def selector_features_batch(bundle: BatchedFrozenBundle, selected_mask: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
    """Return B x P x 9 selector features, preserving the legacy feature layout."""
    batch, patches = selected_mask.shape
    flattened = bundle.corrections.flatten(2)
    row = torch.arange(bundle.refiner.grid, device=flattened.device, dtype=flattened.dtype).repeat_interleave(bundle.refiner.grid)
    col = torch.arange(bundle.refiner.grid, device=flattened.device, dtype=flattened.dtype).repeat(bundle.refiner.grid)
    row = (row / max(bundle.refiner.grid - 1, 1))[None].expand(batch, -1)
    col = (col / max(bundle.refiner.grid - 1, 1))[None].expand(batch, -1)
    provisional_rms = bundle.provisional.flatten(1).square().mean(1).sqrt()[:, None].expand(batch, patches)
    return torch.stack(
        [
            flattened.mean(2),
            flattened.std(2),
            flattened.square().mean(2).sqrt(),
            flattened.abs().amax(2),
            provisional_rms,
            row,
            col,
            selected_mask.to(flattened.dtype),
            time[:, None].to(flattened.dtype).expand(batch, patches),
        ],
        dim=2,
    )


@torch.no_grad()
def select_batch(selector, selector_stats: dict[str, torch.Tensor], bundle: BatchedFrozenBundle, counts: torch.Tensor, time: torch.Tensor):
    """Run greedy patch selection for many trajectories in parallel."""
    batch, patches = bundle.contributions.shape[:2]
    counts = counts.to(bundle.current.device, dtype=torch.long).reshape(batch)
    selected_mask = torch.zeros((batch, patches), device=bundle.current.device, dtype=torch.bool)
    chosen = [[] for _ in range(batch)]
    fields = selector_fields_batch(bundle)
    max_count = int(counts.max().item()) if batch else 0
    embedding = None
    for selection_index in range(max_count + 1):
        features = selector_features_batch(bundle, selected_mask, time)
        context = torch.stack([time, selected_mask.float().mean(1) * 49.0 / 4.0, torch.ones_like(time)], dim=1)
        score, embedding = selector(
            fields,
            (features - selector_stats["mean"]) / selector_stats["std"],
            selected_mask,
            context,
            return_embedding=True,
        )
        active = selection_index < counts
        if not active.any():
            break
        score = score.masked_fill(selected_mask, -torch.inf)
        indices = score.argmax(1)
        for row in torch.nonzero(active, as_tuple=False).flatten().tolist():
            patch = int(indices[row])
            selected_mask[row, patch] = True
            chosen[row].append(patch)
    return chosen, embedding


@torch.no_grad()
def observe_batch(selector, selector_stats: dict[str, torch.Tensor], bundle: BatchedFrozenBundle, remaining: torch.Tensor, step: int):
    batch = bundle.current.shape[0]
    time = torch.full((batch,), step / 38, device=bundle.current.device, dtype=bundle.current.dtype)
    _, embedding = select_batch(selector, selector_stats, bundle, torch.zeros(batch, device=time.device, dtype=torch.long), time)
    features = selector_features_batch(bundle, torch.zeros((batch, bundle.contributions.shape[1]), device=time.device, dtype=torch.bool), time)
    fields = selector_fields_batch(bundle)
    context = torch.stack([torch.full_like(time, step / 37), remaining / 76.0, torch.full_like(time, (37 - step) / 37)], dim=1)
    scores = selector(fields, (features - selector_stats["mean"]) / selector_stats["std"], torch.zeros_like(features[:, :, 0], dtype=torch.bool), torch.stack([time, torch.zeros_like(time), torch.ones_like(time)], dim=1))
    stats = torch.stack([scores.topk(1).values.mean(1), scores.topk(2).values.mean(1), scores.topk(4).values.mean(1), scores.std(1), (scores > 0).float().mean(1)], dim=1)
    return fields, stats, embedding, context


def rvpi_actions_batch(policy, observations, allowed: list[list[int]], action_values: tuple[int, ...] = (0, 1, 2, 4)) -> torch.Tensor:
    logits = policy(*observations)
    selected = []
    for row, choices in enumerate(allowed):
        mask = torch.tensor([action in choices for action in action_values], device=logits.device)
        selected.append(action_values[int(logits[row].masked_fill(~mask, -torch.inf).argmax())])
    return torch.tensor(selected, device=logits.device, dtype=torch.long)


@torch.no_grad()
def batched_future_return(data, refiner, selector, selector_stats, policy, states: list[dict], initial_actions: list[int], feasible, immediate: bool = False) -> list[float]:
    """Evaluate same-step candidate continuations together without changing policy semantics."""
    if not states:
        return []
    steps = {int(state["step"]) for state in states}
    if len(steps) != 1:
        raise ValueError("Batched rollout requires states from one physical time step")
    device = next(refiner.coarse.parameters()).device
    step = steps.pop()
    cases = [int(state["case"]) for state in states]
    previous = torch.cat([state["previous"].to(device) for state in states])
    current = torch.cat([state["current"].to(device) for state in states])
    remaining = torch.tensor([int(state["remaining"]) + int(state["q"]) for state in states], device=device)
    initial_actions = torch.tensor(initial_actions, device=device, dtype=torch.long)
    numerator = torch.zeros(len(states), device=device)
    denominator = torch.zeros(len(states), device=device)
    for time_index in range(step, 38):
        force = torch.stack([data.forcing(case, time_index) for case in cases]).to(device)
        target = torch.stack([data.frame(case, time_index + 1) for case in cases]).to(device)
        time = torch.full((len(states),), time_index / 38, device=device, dtype=current.dtype)
        bundle = BatchedFrozenBundle.create(refiner, force, previous, current, time)
        allowed = [feasible(int(remaining[row]), 37 - time_index) for row in range(len(states))]
        actions = initial_actions if time_index == step else rvpi_actions_batch(policy, observe_batch(selector, selector_stats, bundle, remaining.float(), time_index), allowed)
        chosen, _ = select_batch(selector, selector_stats, bundle, actions, time)
        next_value = bundle.apply_sets(chosen)
        numerator += (next_value - target).square().flatten(1).sum(1)
        denominator += target.square().flatten(1).sum(1)
        previous, current, remaining = current, next_value, remaining - actions
        if immediate:
            break
    values = -numerator / denominator.clamp_min(1e-12)
    return torch.nan_to_num(values, nan=-1e6, posinf=-1e6, neginf=-1e6).cpu().tolist()
