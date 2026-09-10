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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("selector",), default="selector")
    parser.parse_args()
    c, device = config(), get_device(str(config()["device"]))
    result, checkpoint = output_paths()
    data, refiner, utility, utility_stats = load_v3_system(c, device)
    dataset = build_selector_dataset(data, refiner, utility, utility_stats, c, checkpoint)
    selector = train_selector(dataset, c, device, checkpoint)
    table = validate_selector(data, refiner, selector, utility_stats, c, result)
    validate_independent_selector(data, refiner, utility, utility_stats, c, result)
    (result / "summary.json").write_text(json.dumps({"stage": "selector", "training_states": len(dataset["gains"]), "selector_gate": table.to_dict(orient="records")}, indent=2), encoding="utf-8")
    print("Solver V4 selector stage completed.")


if __name__ == "__main__":
    main()
