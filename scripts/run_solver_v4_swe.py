from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_solver_v3_swe import (
    config as v3_config,
    coarse_model,
    local_model,
    patch_features_from_inputs,
    train_stats,
)
from src.solver_v3.data.official_shallow_water import OfficialShallowWater
from src.solver_v3.env.hybrid_rollout import HybridRefiner, relative_l2
from src.solver_v3.models.patch_policy import PatchUtilityNet
from src.solver_v4.models.set_aware_selector import SetAwareSelector
from src.solver_v4.env.frozen_patch_bundle import FrozenPatchBundle
from src.utils.seed import get_device, set_seed


def config() -> dict:
    with open(ROOT / "configs" / "solver_v4_swe.yaml", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def output_paths() -> tuple[Path, Path]:
    result = ROOT / "results" / "solver_v4" / "shallow_water"
    checkpoint = ROOT / "checkpoints" / "solver_v4" / "shallow_water"
    result.mkdir(parents=True, exist_ok=True)
    checkpoint.mkdir(parents=True, exist_ok=True)
    return result, checkpoint


def load_v3_system(c: dict, device: torch.device) -> tuple[OfficialShallowWater, HybridRefiner, PatchUtilityNet, dict[str, torch.Tensor]]:
    c3 = v3_config()
    data = OfficialShallowWater(ROOT / str(c["data"]["official_dir"]), int(c["data"]["train_cases"]), int(c["data"]["validation_cases"]), int(c["data"]["test_cases"]))
    stats = train_stats(data, c3)
    source = ROOT / "checkpoints" / "solver_v3" / "shallow_water"
    coarse_path, local_path, utility_path = source / "coarse_transition.pt", source / "local_corrector.pt", source / "patch_utility.pt"
    for path in (coarse_path, local_path, utility_path):
        if not path.exists():
            raise FileNotFoundError(f"Required Solver V3 checkpoint is missing: {path}")
    coarse = coarse_model(c3, data.channels, device)
    coarse.load_state_dict(torch.load(coarse_path, map_location=device)["model"])
    local = local_model(c3, data.channels, device)
    local.load_state_dict(torch.load(local_path, map_location=device)["model"])
    refiner = HybridRefiner(coarse.eval(), local.eval(), int(c["data"]["stride"]), int(c["model"]["patch_core"]), int(c["model"]["patch_halo"]), stats["mean"], stats["std"])
    saved = torch.load(utility_path, map_location=device)
    utility = PatchUtilityNet(int(saved["feature_dim"])).to(device)
    utility.load_state_dict(saved["model"]); utility.eval()
    utility_stats = {key: saved[key].to(device) for key in ("feature_mean", "feature_std", "utility_mean", "utility_std")}
    return data, refiner, utility, utility_stats


def normalized_features(features: torch.Tensor, utility_stats: dict[str, torch.Tensor]) -> torch.Tensor:
    return (features - utility_stats["feature_mean"]) / utility_stats["feature_std"]


def field_tensor(current: torch.Tensor, provisional: torch.Tensor) -> torch.Tensor:
    return F.interpolate(torch.cat([current, provisional, provisional - current], dim=1), size=(32, 32), mode="bilinear", align_corners=False)


@torch.no_grad()
def candidate_marginal_gains(refiner: HybridRefiner, current: torch.Tensor, base: torch.Tensor, target: torch.Tensor, time_fraction: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """GT train/validation label: full-frame MSE decrease after one conditional patch."""
    patches = list(range(64))
    inputs = refiner.patch_inputs(current, base, patches, time_fraction)
    corrections = refiner.local_model(inputs)
    candidates = base.expand(64, -1, -1, -1).clone()
    for patch, correction in enumerate(corrections.split(1, dim=0)):
        top, bottom, left, right = refiner.patch_bounds(patch)
        r0, r1 = max(top, 0), min(bottom, base.shape[-2])
        c0, c1 = max(left, 0), min(right, base.shape[-1])
        rs, cs = slice(r0 - top, r1 - top), slice(c0 - left, c1 - left)
        window = refiner._flat_cosine_window(bottom - top, refiner.halo, base.device, base.dtype)[None, None, rs, cs]
        candidates[patch : patch + 1, :, r0:r1, c0:c1] += correction[:, :, rs, cs] * (window > 0).to(base.dtype)
    base_error = F.mse_loss(base, target)
    errors = ((candidates - target.expand_as(candidates)) ** 2).flatten(1).mean(1)
    features = patch_features_from_inputs(inputs, current, time_fraction)
    return base_error - errors, features, corrections


@torch.no_grad()
def apply_selected(refiner: HybridRefiner, current: torch.Tensor, provisional: torch.Tensor, selected: list[int], time_fraction: float) -> torch.Tensor:
    if not selected:
        return provisional
    inputs = refiner.patch_inputs(current, provisional, selected, time_fraction)
    return refiner.blend_corrections(provisional, selected, refiner.local_model(inputs))


def source_action(source: str, features: torch.Tensor, gains: torch.Tensor, utility: PatchUtilityNet, utility_stats: dict[str, torch.Tensor], generator: np.random.Generator) -> list[int]:
    if source == "coarse":
        return []
    q = int(generator.choice([0, 1, 2, 4]))
    if q == 0:
        return []
    if source == "random":
        return generator.choice(64, q, replace=False).tolist()
    if source == "independent":
        scores = utility(normalized_features(features, utility_stats))
    else:
        scores = gains
    return torch.topk(scores, q).indices.tolist()


@torch.no_grad()
def build_selector_dataset(data: OfficialShallowWater, refiner: HybridRefiner, utility: PatchUtilityNet, utility_stats: dict[str, torch.Tensor], c: dict, checkpoint: Path) -> dict[str, torch.Tensor]:
    # Versioned because this phase deliberately expands from a source-balanced
    # pilot to all available training trajectories without changing labels.
    cache = checkpoint / "selector_train_labels_all_train.pt"
    if cache.exists():
        return torch.load(cache, map_location="cpu")
    generator = np.random.default_rng(int(c["seed"]))
    source_names = ("coarse", "random", "independent", "privileged")
    source_cases = {name: generator.choice(data.case_ids("train"), int(c["training"]["selector_cases_per_source"]), replace=False).tolist() for name in source_names}
    rows: dict[str, list[torch.Tensor]] = {key: [] for key in ("fields", "features", "mask", "context", "gains")}
    stride = int(c["data"]["stride"])
    for source in source_names:
        for case in tqdm(source_cases[source], desc=f"v4:selector-labels:{source}"):
            state = data.frame(case, 0).unsqueeze(0).to(next(refiner.coarse_model.parameters()).device)
            for time_index in range(0, 72 - stride, stride):
                target = data.frame(case, time_index + stride).unsqueeze(0).to(state.device)
                _, provisional = refiner.coarse_next(state, time_index / 71.0)
                initial_gains, initial_features, _ = candidate_marginal_gains(refiner, state, provisional, target, time_index / 71.0)
                q_target = int(generator.choice([1, 2, 4]))
                selected_size = int(generator.integers(0, min(3, q_target) + 1))
                if selected_size:
                    if source == "random":
                        selected = generator.choice(64, selected_size, replace=False).tolist()
                    elif source == "independent":
                        selected = torch.topk(utility(normalized_features(initial_features, utility_stats)), selected_size).indices.tolist()
                    else:
                        selected = torch.topk(initial_gains, selected_size).indices.tolist()
                else:
                    selected = []
                base = apply_selected(refiner, state, provisional, selected, time_index / 71.0)
                gains, features, _ = candidate_marginal_gains(refiner, state, base, target, time_index / 71.0)
                mask = torch.zeros(64, dtype=torch.bool, device=state.device)
                mask[selected] = True
                rows["fields"].append(field_tensor(state, base).cpu())
                rows["features"].append(normalized_features(features, utility_stats).cpu())
                rows["mask"].append(mask.cpu())
                rows["context"].append(torch.tensor([q_target / 4.0, selected_size / 4.0]))
                rows["gains"].append(gains.cpu())
                next_selected = source_action(source, initial_features, initial_gains, utility, utility_stats, generator)
                state = apply_selected(refiner, state, provisional, next_selected, time_index / 71.0)
    dataset = {
        "fields": torch.cat(rows["fields"]),
        "features": torch.stack(rows["features"]),
        "mask": torch.stack(rows["mask"]),
        "context": torch.stack(rows["context"]),
        "gains": torch.stack(rows["gains"]),
    }
    torch.save(dataset, cache)
    return dataset


def selector_loss(scores: torch.Tensor, targets: torch.Tensor, selected: torch.Tensor, pair_weight: float) -> torch.Tensor:
    valid = ~selected
    target_mean = targets[valid].mean()
    target_std = targets[valid].std().clamp_min(1e-7)
    standardized = (targets - target_mean) / target_std
    # Preserve Huber regression while emphasizing the rare high-value patches
    # that determine a sequential top-q decision.
    regression_weight = (1.0 + standardized.clamp_min(0.0)).detach()
    regression = (F.huber_loss(scores, standardized, reduction="none") * regression_weight * valid).sum() / valid.sum().clamp_min(1)
    differences = targets[:, :, None] - targets[:, None, :]
    score_differences = scores[:, :, None] - scores[:, None, :]
    pair_mask = (differences.abs() > 1e-8) & valid[:, :, None] & valid[:, None, :]
    all_pairs = F.softplus(-differences.sign()[pair_mask] * score_differences[pair_mask]).mean()
    masked_target = targets.masked_fill(~valid, -torch.inf)
    best = masked_target.argmax(1)
    batch = torch.arange(scores.shape[0], device=scores.device)
    best_delta = scores[batch, best, None] - scores
    best_target_delta = targets[batch, best, None] - targets
    top_mask = valid & (torch.arange(64, device=scores.device)[None] != best[:, None]) & (best_target_delta.abs() > 1e-8)
    top_pairs = F.softplus(-best_target_delta.sign()[top_mask] * best_delta[top_mask]).mean()
    ranking = 0.5 * all_pairs + 0.5 * top_pairs
    return regression + pair_weight * ranking


def train_selector(dataset: dict[str, torch.Tensor], c: dict, device: torch.device, checkpoint: Path) -> SetAwareSelector:
    path = checkpoint / "set_aware_selector.pt"
    model = SetAwareSelector(int(dataset["features"].shape[-1]), int(c["model"]["selector_dim"]), int(c["model"]["selector_heads"]), int(c["model"]["selector_layers"])).to(device)
    if path.exists():
        saved = torch.load(path, map_location=device)
        if saved.get("loss_version") == 3:
            model.load_state_dict(saved["model"])
            return model.eval()
    set_seed(int(c["seed"]))
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(c["training"]["selector_lr"]), weight_decay=1e-5)
    count, batch = len(dataset["gains"]), int(c["training"]["selector_batch_size"])
    for _ in tqdm(range(int(c["training"]["selector_epochs"])), desc="v4:set-selector"):
        for indices in torch.randperm(count).split(batch):
            fields = dataset["fields"][indices].to(device)
            features = dataset["features"][indices].to(device)
            mask = dataset["mask"][indices].to(device)
            context = dataset["context"][indices].to(device)
            gains = dataset["gains"][indices].to(device)
            loss = selector_loss(model(fields, features, mask, context), gains, mask, float(c["training"]["pairwise_weight"]))
            optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
    torch.save({"model": model.state_dict(), "feature_dim": int(dataset["features"].shape[-1]), "loss_version": 3}, path)
    return model.eval()


def rank_metrics(scores: torch.Tensor, gains: torch.Tensor, valid: torch.Tensor) -> tuple[float, float, float]:
    s, g = scores[valid].detach().cpu().numpy(), gains[valid].detach().cpu().numpy()
    pearson = float(np.corrcoef(s, g)[0, 1]) if s.std() and g.std() else 0.0
    spearman = float(pd.Series(s).rank().corr(pd.Series(g).rank()))
    pairwise = float(np.mean((np.sign(s[:, None] - s[None, :]) == np.sign(g[:, None] - g[None, :]))[np.triu_indices(len(s), 1)]))
    return pearson, spearman, pairwise


@torch.no_grad()
def choose_set_aware(refiner: HybridRefiner, selector: SetAwareSelector, current: torch.Tensor, provisional: torch.Tensor, target: torch.Tensor, time_fraction: float, count: int, utility_stats: dict[str, torch.Tensor], privileged: bool) -> tuple[torch.Tensor, list[int], list[float], list[float], list[float]]:
    selected: list[int] = []
    overlaps: list[float] = []; spearman: list[float] = []; recovery: list[float] = []
    base = provisional
    for _ in range(count):
        gains, features, _ = candidate_marginal_gains(refiner, current, base, target, time_fraction)
        mask = torch.zeros(64, dtype=torch.bool, device=base.device); mask[selected] = True
        fields = field_tensor(current, base)
        context = torch.tensor([[count / 4.0, len(selected) / 4.0]], device=base.device)
        scores = selector(fields, normalized_features(features, utility_stats).unsqueeze(0), mask.unsqueeze(0), context)[0]
        valid = ~mask
        true_index = int(gains.masked_fill(mask, -torch.inf).argmax())
        chosen = true_index if privileged else int(scores.masked_fill(mask, -torch.inf).argmax())
        overlaps.append(float(chosen == true_index)); spearman.append(rank_metrics(scores, gains, valid)[1])
        best_gain = float(gains[true_index])
        recovery.append(float(gains[chosen] / gains[true_index]) if best_gain > 1e-12 else 1.0)
        base = apply_selected(refiner, current, base, [chosen], time_fraction)
        selected.append(chosen)
    return base, selected, overlaps, spearman, recovery


@torch.no_grad()
def validate_selector(data: OfficialShallowWater, refiner: HybridRefiner, selector: SetAwareSelector, utility_stats: dict[str, torch.Tensor], c: dict, result: Path) -> pd.DataFrame:
    rows = []
    stride = int(c["data"]["stride"])
    for count in (1, 2, 4):
        overlap, spearman, pairwise, recovery, errors = [], [], [], [], []
        for case in tqdm(data.case_ids("val"), desc=f"v4:selector-validation:q{count}"):
            state = data.frame(case, 0).unsqueeze(0).to(next(refiner.coarse_model.parameters()).device)
            for time_index in range(0, 72 - stride, stride):
                target = data.frame(case, time_index + stride).unsqueeze(0).to(state.device)
                _, provisional = refiner.coarse_next(state, time_index / 71.0)
                deployed, _, o, s, r = choose_set_aware(refiner, selector, state, provisional, target, time_index / 71.0, count, utility_stats, privileged=False)
                overlap.extend(o); spearman.extend(s); recovery.extend(r); errors.append(float(relative_l2(deployed, target)[0]))
                # Pairwise metric for the same conditional state used by the selector.
                gains, features, _ = candidate_marginal_gains(refiner, state, provisional, target, time_index / 71.0)
                scores = selector(field_tensor(state, provisional), normalized_features(features, utility_stats).unsqueeze(0), torch.zeros(1, 64, dtype=torch.bool, device=state.device), torch.tensor([[count / 4.0, 0.0]], device=state.device))[0]
                pairwise.append(rank_metrics(scores, gains, torch.ones(64, dtype=torch.bool, device=state.device))[2])
                state = deployed
        rows.append({"Q": count, "States": len(errors), "TopKOverlap": float(np.mean(overlap)), "Spearman": float(np.mean(spearman)), "PairwiseAccuracy": float(np.mean(pairwise)), "ImmediateGainRecovery": float(np.mean(recovery)), "GlobalErrorAfterSelection": float(np.mean(errors))})
    table = pd.DataFrame(rows)
    table.to_csv(result / "set_selector_validation.csv", index=False)
    return table


@torch.no_grad()
def validate_independent_selector(data: OfficialShallowWater, refiner: HybridRefiner, utility: PatchUtilityNet, utility_stats: dict[str, torch.Tensor], c: dict, result: Path) -> None:
    """Same conditional global-gain protocol for the V3 independent utility baseline."""
    rows, stride = [], int(c["data"]["stride"])
    for count in (1, 2, 4):
        overlap, spearman, pairwise, recovery, errors = [], [], [], [], []
        for case in tqdm(data.case_ids("val"), desc=f"v4:independent-validation:q{count}"):
            state = data.frame(case, 0).unsqueeze(0).to(next(refiner.coarse_model.parameters()).device)
            for time_index in range(0, 72 - stride, stride):
                target = data.frame(case, time_index + stride).unsqueeze(0).to(state.device)
                _, base = refiner.coarse_next(state, time_index / 71.0)
                selected = []
                for _ in range(count):
                    gains, features, _ = candidate_marginal_gains(refiner, state, base, target, time_index / 71.0)
                    mask = torch.zeros(64, dtype=torch.bool, device=state.device); mask[selected] = True
                    scores = utility(normalized_features(features, utility_stats))
                    true_index = int(gains.masked_fill(mask, -torch.inf).argmax())
                    chosen = int(scores.masked_fill(mask, -torch.inf).argmax())
                    valid = ~mask
                    _, sp, pw = rank_metrics(scores, gains, valid)
                    overlap.append(float(chosen == true_index)); spearman.append(sp); pairwise.append(pw)
                    best_gain = float(gains[true_index])
                    recovery.append(float(gains[chosen] / gains[true_index]) if best_gain > 1e-12 else 1.0)
                    base = apply_selected(refiner, state, base, [chosen], time_index / 71.0)
                    selected.append(chosen)
                errors.append(float(relative_l2(base, target)[0])); state = base
        rows.append({"Q": count, "States": len(errors), "TopKOverlap": float(np.mean(overlap)), "Spearman": float(np.mean(spearman)), "PairwiseAccuracy": float(np.mean(pairwise)), "ImmediateGainRecovery": float(np.mean(recovery)), "GlobalErrorAfterSelection": float(np.mean(errors))})
    pd.DataFrame(rows).to_csv(result / "independent_selector_validation.csv", index=False)


def v2_paths() -> tuple[Path, Path]:
    result = ROOT / "results" / "solver_v4" / "shallow_water" / "selector_semantics_v2"
    checkpoint = ROOT / "checkpoints" / "solver_v4" / "shallow_water"
    result.mkdir(parents=True, exist_ok=True)
    return result, checkpoint


def overlap_features(selected: list[int], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    rows = torch.arange(8, device=device).repeat_interleave(8)
    cols = torch.arange(8, device=device).repeat(8)
    if not selected:
        return torch.stack([torch.ones(64, device=device, dtype=dtype), torch.zeros(64, device=device, dtype=dtype), torch.zeros(64, device=device, dtype=dtype), torch.zeros(64, device=device, dtype=dtype), torch.zeros(64, device=device, dtype=dtype)], dim=1)
    chosen = torch.as_tensor(selected, device=device)
    dr, dc = (rows[:, None] - rows[chosen]).abs(), (cols[:, None] - cols[chosen]).abs()
    distance = torch.sqrt((dr.float() ** 2 + dc.float() ** 2).min(1).values) / np.sqrt(98.0)
    adjacent = ((dr + dc) == 1).sum(1).to(dtype)
    halo_overlap = ((dr <= 1) & (dc <= 1) & ((dr + dc) > 0)).sum(1).to(dtype)
    overlap_area = ((48 - 32 * dr).clamp_min(0) * (48 - 32 * dc).clamp_min(0)).sum(1).to(dtype) / float(48 * 48)
    cosine_strength = (overlap_area / max(len(selected), 1)).clamp_max(1.0)
    return torch.stack([distance.to(dtype), adjacent, overlap_area, halo_overlap, cosine_strength], dim=1)


def correction_features(bundle: FrozenPatchBundle, time_fraction: float, selected: list[int]) -> torch.Tensor:
    base = patch_features_from_inputs(bundle.patch_inputs, bundle.current, time_fraction)
    correction, provisional = bundle.corrections, bundle.patch_inputs[:, bundle.current.shape[1] : 2 * bundle.current.shape[1]]
    dx, dy = correction[..., 1:] - correction[..., :-1], correction[..., 1:, :] - correction[..., :-1, :]
    high = correction - F.avg_pool2d(correction, 3, stride=1, padding=1)
    flat = correction.flatten(1)
    corr_rms = flat.square().mean(1).sqrt()
    proposal = torch.stack([flat.mean(1), flat.std(1), flat.abs().mean(1), corr_rms, flat.abs().amax(1), dx.flatten(1).abs().mean(1), dy.flatten(1).abs().mean(1), high.flatten(1).abs().mean(1), provisional.flatten(1).square().mean(1).sqrt(), corr_rms / provisional.flatten(1).square().mean(1).sqrt().clamp_min(1e-8)], dim=1)
    return torch.cat([base, proposal, overlap_features(selected, base.device, base.dtype)], dim=1)


def frozen_gains(bundle: FrozenPatchBundle, target: torch.Tensor, selected: list[int]) -> tuple[torch.Tensor, float, torch.Tensor]:
    base = bundle.apply_set(selected)
    base_error = F.mse_loss(base, target)
    candidates = bundle.candidate_fields(selected)
    errors = ((candidates - target.expand_as(candidates)) ** 2).flatten(1).mean(1)
    gains = base_error - errors
    if selected:
        gains[torch.as_tensor(selected, device=gains.device)] = -torch.inf
    return gains, float(base_error), base


def source_frozen_set(source: str, bundle: FrozenPatchBundle, target: torch.Tensor, utility: PatchUtilityNet, utility_stats: dict[str, torch.Tensor], time_fraction: float, generator: np.random.Generator) -> list[int]:
    if source == "coarse":
        return []
    count = int(generator.choice([0, 1, 2, 4]))
    if not count:
        return []
    if source == "random":
        return generator.choice(64, count, replace=False).tolist()
    if source == "independent":
        base = patch_features_from_inputs(bundle.patch_inputs, bundle.current, time_fraction)
        return torch.topk(utility(normalized_features(base, utility_stats)), count).indices.tolist()
    selected = []
    for _ in range(count):
        gains, _, _ = frozen_gains(bundle, target, selected)
        selected.append(int(gains.argmax()))
    return selected


@torch.no_grad()
def build_frozen_dataset(data: OfficialShallowWater, refiner: HybridRefiner, utility: PatchUtilityNet, utility_stats: dict[str, torch.Tensor], c: dict, checkpoint: Path) -> dict[str, torch.Tensor]:
    cache = checkpoint / "selector_train_labels_frozen_semantics_v2.pt"
    if cache.exists():
        return torch.load(cache, map_location="cpu")
    generator, stride = np.random.default_rng(int(c["seed"])), int(c["data"]["stride"])
    rows = {key: [] for key in ("fields", "raw_features", "mask", "context", "gains")}
    sources = ("coarse", "random", "independent", "privileged")
    for source in sources:
        for case in tqdm(data.case_ids("train"), desc=f"v4:v2-labels:{source}"):
            state = data.frame(case, 0).unsqueeze(0).to(next(refiner.coarse_model.parameters()).device)
            for time_index in range(0, 72 - stride, stride):
                target = data.frame(case, time_index + stride).unsqueeze(0).to(state.device)
                bundle = FrozenPatchBundle.create(refiner, state, time_index / 71.0)
                q_target = int(generator.choice([1, 2, 4], p=c["training"]["q_target_probabilities"]))
                size = int(generator.choice([0, 1, 2, 3], p=c["training"]["selected_size_probabilities"]))
                selected = generator.choice(64, size, replace=False).tolist() if size else []
                gains, _, _ = frozen_gains(bundle, target, selected)
                mask = torch.zeros(64, dtype=torch.bool, device=state.device); mask[selected] = True
                rows["fields"].append(field_tensor(state, bundle.provisional).cpu())
                rows["raw_features"].append(correction_features(bundle, time_index / 71.0, selected).cpu())
                rows["mask"].append(mask.cpu())
                rows["context"].append(torch.tensor([q_target / 4.0, size / 4.0]))
                rows["gains"].append(gains.cpu())
                state = bundle.apply_set(source_frozen_set(source, bundle, target, utility, utility_stats, time_index / 71.0, generator))
    data_rows = {"fields": torch.cat(rows["fields"]), "raw_features": torch.stack(rows["raw_features"]), "mask": torch.stack(rows["mask"]), "context": torch.stack(rows["context"]), "gains": torch.stack(rows["gains"])}
    torch.save(data_rows, cache)
    return data_rows


def v2_loss(scores: torch.Tensor, gains: torch.Tensor, selected: torch.Tensor, mean: torch.Tensor, std: torch.Tensor, noise_threshold: torch.Tensor, c: dict) -> torch.Tensor:
    valid = ~selected
    safe_gains = torch.where(valid, gains, mean)
    targets = (safe_gains - mean) / std
    regression = F.huber_loss(scores[valid], targets[valid])
    difference, score_difference = targets[:, :, None] - targets[:, None, :], scores[:, :, None] - scores[:, None, :]
    pairs = valid[:, :, None] & valid[:, None, :] & (difference.abs() > 1e-8)
    best = safe_gains.masked_fill(~valid, -torch.inf).amax(1)
    state_weight = torch.where(best >= noise_threshold, torch.ones_like(best), torch.full_like(best, 0.25))
    pair_loss = F.softplus(-difference.sign() * score_difference)
    pairwise = (pair_loss * pairs * state_weight[:, None, None]).sum() / pairs.sum().clamp_min(1)
    top = gains.masked_fill(~valid, -torch.inf).topk(8, dim=1).indices
    batch = torch.arange(scores.shape[0], device=scores.device)[:, None]
    top_scores, top_targets = scores[batch, top], targets[batch, top]
    top_difference = top_scores[:, :, None] - scores[:, None, :]
    target_difference = top_targets[:, :, None] - targets[:, None, :]
    top_pairs = valid[:, None, :] & (target_difference > 0)
    top_loss = (F.relu(0.1 - top_difference) * top_pairs * state_weight[:, None, None]).sum() / top_pairs.sum().clamp_min(1)
    return regression + float(c["training"]["pairwise_weight"]) * pairwise + float(c["training"]["top_weight"]) * top_loss


def train_selector_v2(dataset: dict[str, torch.Tensor], c: dict, device: torch.device, checkpoint: Path) -> tuple[SetAwareSelector, dict[str, torch.Tensor]]:
    path = checkpoint / "set_aware_selector_v2.pt"
    raw = dataset["raw_features"]
    statistics = {"feature_mean": raw.mean((0, 1)), "feature_std": raw.std((0, 1)).clamp_min(1e-6), "gain_mean": dataset["gains"][torch.isfinite(dataset["gains"])].mean(), "gain_std": dataset["gains"][torch.isfinite(dataset["gains"])].std().clamp_min(1e-8)}
    model = SetAwareSelector(raw.shape[-1], int(c["model"]["selector_dim"]), int(c["model"]["selector_heads"]), int(c["model"]["selector_layers"])).to(device)
    if path.exists():
        saved = torch.load(path, map_location=device)
        if saved.get("feature_version") == "frozen_semantics_v2":
            model.load_state_dict(saved["model"])
            return model.eval(), {key: saved[key].to(device) for key in statistics}
    set_seed(int(c["seed"])); optimizer = torch.optim.AdamW(model.parameters(), lr=float(c["training"]["selector_lr"]), weight_decay=1e-5)
    threshold = dataset["gains"].masked_fill(dataset["mask"], -torch.inf).amax(1).quantile(0.1).to(device)
    count, batch = len(dataset["gains"]), int(c["training"]["selector_batch_size"])
    for _ in tqdm(range(int(c["training"]["selector_v2_epochs"])), desc="v4:set-selector-v2"):
        for indices in torch.randperm(count).split(batch):
            fields, features = dataset["fields"][indices].to(device), ((raw[indices] - statistics["feature_mean"]) / statistics["feature_std"]).to(device)
            mask, context, gains = dataset["mask"][indices].to(device), dataset["context"][indices].to(device), dataset["gains"][indices].to(device)
            loss = v2_loss(model(fields, features, mask, context), gains, mask, statistics["gain_mean"].to(device), statistics["gain_std"].to(device), threshold, c)
            optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
    torch.save({"model": model.state_dict(), **{key: value.cpu() for key, value in statistics.items()}, "feature_version": "frozen_semantics_v2"}, path)
    return model.eval(), {key: value.to(device) for key, value in statistics.items()}


def selector_scores_v2(model: SetAwareSelector, stats: dict[str, torch.Tensor], bundle: FrozenPatchBundle, time_fraction: float, selected: list[int], q: int) -> torch.Tensor:
    mask = torch.zeros(64, dtype=torch.bool, device=bundle.provisional.device); mask[selected] = True
    features = (correction_features(bundle, time_fraction, selected) - stats["feature_mean"]) / stats["feature_std"]
    context = torch.tensor([[q / 4.0, len(selected) / 4.0]], device=bundle.provisional.device)
    return model(field_tensor(bundle.current, bundle.provisional), features.unsqueeze(0), mask.unsqueeze(0), context)[0]


@torch.no_grad()
def select_frozen(bundle: FrozenPatchBundle, target: torch.Tensor, q: int, time_fraction: float, model: SetAwareSelector | None, stats: dict[str, torch.Tensor] | None, utility: PatchUtilityNet | None, utility_stats: dict[str, torch.Tensor] | None, privileged: bool) -> tuple[list[int], dict[str, float]]:
    selected, overlaps, spearman, pairwise = [], [], [], []
    initial_error = float(F.mse_loss(bundle.provisional, target))
    if q == 0:
        return selected, {"initial_error": initial_error, "final_error": initial_error, "overlap": 1.0, "spearman": 1.0, "pairwise": 1.0}
    for _ in range(q):
        gains, _, _ = frozen_gains(bundle, target, selected)
        valid = torch.isfinite(gains)
        true = int(gains.argmax())
        if privileged:
            chosen = true
        elif model is not None:
            scores = selector_scores_v2(model, stats, bundle, time_fraction, selected, q)
            chosen = int(scores.masked_fill(~valid, -torch.inf).argmax())
            _, sp, pw = rank_metrics(scores, gains, valid); spearman.append(sp); pairwise.append(pw)
        else:
            base = patch_features_from_inputs(bundle.patch_inputs, bundle.current, time_fraction)
            scores = utility(normalized_features(base, utility_stats))
            chosen = int(scores.masked_fill(~valid, -torch.inf).argmax())
            _, sp, pw = rank_metrics(scores, gains, valid); spearman.append(sp); pairwise.append(pw)
        overlaps.append(float(chosen == true)); selected.append(chosen)
    final = bundle.apply_set(selected)
    final_error = float(F.mse_loss(final, target))
    return selected, {"initial_error": initial_error, "final_error": final_error, "overlap": float(np.mean(overlaps)), "spearman": float(np.mean(spearman)) if spearman else 1.0, "pairwise": float(np.mean(pairwise)) if pairwise else 1.0}


@torch.no_grad()
def validate_selector_v2(data: OfficialShallowWater, refiner: HybridRefiner, model: SetAwareSelector, stats: dict[str, torch.Tensor], utility: PatchUtilityNet, utility_stats: dict[str, torch.Tensor], c: dict, result: Path) -> pd.DataFrame:
    rows, stride = [], int(c["data"]["stride"])
    for q in (1, 2, 4):
        aggregates = {key: [] for key in ("overlap", "spearman", "pairwise", "pred", "priv", "initial", "regret", "positive")}
        for case in tqdm(data.case_ids("val"), desc=f"v4:v2-selector-validation:q{q}"):
            state = data.frame(case, 0).unsqueeze(0).to(next(refiner.coarse_model.parameters()).device)
            for time_index in range(0, 72 - stride, stride):
                target = data.frame(case, time_index + stride).unsqueeze(0).to(state.device)
                bundle = FrozenPatchBundle.create(refiner, state, time_index / 71.0)
                selected, pred = select_frozen(bundle, target, q, time_index / 71.0, model, stats, None, None, False)
                _, priv = select_frozen(bundle, target, q, time_index / 71.0, None, None, None, None, True)
                aggregates["overlap"].append(pred["overlap"]); aggregates["spearman"].append(pred["spearman"]); aggregates["pairwise"].append(pred["pairwise"])
                aggregates["pred"].append(pred["final_error"]); aggregates["priv"].append(priv["final_error"]); aggregates["initial"].append(pred["initial_error"]); aggregates["regret"].append(pred["final_error"] - priv["final_error"]); aggregates["positive"].append(pred["final_error"] < pred["initial_error"])
                state = bundle.apply_set(selected)
        initial, pred, priv = np.asarray(aggregates["initial"]), np.asarray(aggregates["pred"]), np.asarray(aggregates["priv"])
        rows.append({"Q": q, "States": len(pred), "Top1Overlap": np.mean(aggregates["overlap"]), "Spearman": np.mean(aggregates["spearman"]), "PairwiseAccuracy": np.mean(aggregates["pairwise"]), "AggregateGainRecovery": (initial - pred).sum() / max((initial - priv).sum(), 1e-12), "AggregateGapRecovery": (initial - pred).sum() / max((initial - priv).sum(), 1e-12), "MeanSetRegret": np.mean(aggregates["regret"]), "MedianSetRegret": np.median(aggregates["regret"]), "PositiveImprovementRate": np.mean(aggregates["positive"]), "PredictedFinalError": np.mean(pred), "PrivilegedFinalError": np.mean(priv), "ProvisionalError": np.mean(initial)})
    table = pd.DataFrame(rows); table.to_csv(result / "set_selector_validation_v2.csv", index=False)
    return table


def is_budget_representable(remaining_budget: int, remaining_steps: int, actions: tuple[int, ...] = (0, 1, 2, 4)) -> bool:
    reachable = {0}
    for _ in range(remaining_steps):
        reachable = {used + action for used in reachable for action in actions if used + action <= remaining_budget}
    return remaining_budget in reachable


@torch.no_grad()
def macro_rollout(data: OfficialShallowWater, refiner: HybridRefiner, model: SetAwareSelector, stats: dict[str, torch.Tensor], c: dict, case: int, beam: bool) -> tuple[float, int]:
    actions, budget, stride = tuple(int(x) for x in c["oracle"]["macro_actions"]), int(c["oracle"]["budget"]), int(c["data"]["stride"])
    times, device = list(range(0, 72 - stride, stride)), next(refiner.coarse_model.parameters()).device
    initial = data.frame(case, 0).unsqueeze(0).to(device)
    branches = [(initial, budget, 0.0, [])]
    for step, time_index in enumerate(times):
        target = data.frame(case, time_index + stride).unsqueeze(0).to(device)
        candidates = []
        for state, remaining, score, trace in branches:
            feasible = [q for q in actions if q <= remaining and is_budget_representable(remaining - q, len(times) - step - 1, actions)]
            if not beam:
                quota = remaining / (len(times) - step)
                feasible = [min(feasible, key=lambda q: (abs(q - quota), -q))]
            for q in feasible:
                bundle = FrozenPatchBundle.create(refiner, state, time_index / 71.0)
                selected, _ = select_frozen(bundle, target, q, time_index / 71.0, model, stats, None, None, False)
                next_state = bundle.apply_set(selected)
                candidates.append((next_state, remaining - q, score + float(relative_l2(next_state, target)[0]), trace + [(q, selected)]))
        candidates.sort(key=lambda row: row[2])
        branches = candidates[: int(c["oracle"]["beam_width"])] if beam else candidates[:1]
    best = branches[0]
    targets = torch.stack([data.frame(case, time_index).to(device) for time_index in [0] + [t + stride for t in times]])
    # The beam score is the sum of per-frame relative errors; report trajectory L2 consistently with earlier tables.
    states = [initial]
    state = initial
    for (q, selected), time_index in zip(best[3], times):
        bundle = FrozenPatchBundle.create(refiner, state, time_index / 71.0)
        state = bundle.apply_set(selected); states.append(state)
    trajectory = torch.cat(states)
    error = float(torch.linalg.vector_norm(trajectory - targets) / torch.linalg.vector_norm(targets).clamp_min(1e-8))
    return error, budget - best[1]


def evaluate_deployable_macro(data: OfficialShallowWater, refiner: HybridRefiner, model: SetAwareSelector, stats: dict[str, torch.Tensor], c: dict, result: Path) -> pd.DataFrame:
    cases, rows = data.case_ids("val")[: int(c["oracle"]["validation_cases"])], []
    greedy, beam, used, positive = [], [], [], 0
    for case in tqdm(cases, desc="v4:v2-deployable-macro"):
        greedy_error, _ = macro_rollout(data, refiner, model, stats, c, case, beam=False)
        beam_error, calls = macro_rollout(data, refiner, model, stats, c, case, beam=True)
        greedy.append(greedy_error); beam.append(beam_error); used.append(calls); positive += beam_error < greedy_error
    table = pd.DataFrame([{"Budget": int(c["oracle"]["budget"]), "Cases": len(cases), "GreedyError": np.mean(greedy), "BeamError": np.mean(beam), "RelativeGain": 100.0 * (np.mean(greedy) - np.mean(beam)) / max(np.mean(greedy), 1e-12), "PositiveCaseRate": 100.0 * positive / len(cases), "MeanBudgetUsed": np.mean(used)}])
    table.to_csv(result / "deployable_macro_headroom_v2.csv", index=False)
    return table


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("selector-v2",), default="selector-v2")
    parser.parse_args()
    c, device = config(), get_device(str(config()["device"]))
    result, checkpoint = v2_paths()
    data, refiner, utility, utility_stats = load_v3_system(c, device)
    dataset = build_frozen_dataset(data, refiner, utility, utility_stats, c, checkpoint)
    selector, statistics = train_selector_v2(dataset, c, device, checkpoint)
    table = validate_selector_v2(data, refiner, selector, statistics, utility, utility_stats, c, result)
    rows = table.set_index("Q")
    noncatastrophic = rows.loc[1, "AggregateGapRecovery"] > 0 and rows.loc[2, "AggregateGapRecovery"] > 0 and rows.loc[4, "AggregateGapRecovery"] > -0.1
    macro = evaluate_deployable_macro(data, refiner, selector, statistics, c, result) if noncatastrophic else None
    payload = {"stage": "frozen_selector_v2", "training_states": len(dataset["gains"]), "selector": table.to_dict(orient="records"), "deployable_macro": None if macro is None else macro.to_dict(orient="records"), "fqi_run": False}
    (result / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("Solver V4 frozen-semantics selector stage completed.")


if __name__ == "__main__":
    main()
