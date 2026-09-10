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

from src.solver_v3.data.official_shallow_water import OfficialShallowWater
from src.solver_v3.env.hybrid_rollout import HybridRefiner, relative_l2
from src.solver_v3.models.fno import FNO2d
from src.solver_v3.models.patch_policy import AdaptivePatchPolicy, PatchUtilityNet
from src.utils.seed import get_device, set_seed


def config() -> dict:
    with open(ROOT / "configs" / "solver_v3_swe.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def paths() -> tuple[Path, Path]:
    result = ROOT / "results" / "solver_v3" / "shallow_water"
    checkpoint = ROOT / "checkpoints" / "solver_v3" / "shallow_water"
    result.mkdir(parents=True, exist_ok=True)
    checkpoint.mkdir(parents=True, exist_ok=True)
    return result, checkpoint


def normalized(x: torch.Tensor, stats: dict[str, torch.Tensor]) -> torch.Tensor:
    return (x - stats["mean"].to(x.device)) / stats["std"].to(x.device)


def train_stats(data: OfficialShallowWater, c: dict) -> dict[str, torch.Tensor]:
    frames = []
    for case_id in data.case_ids("train")[:16]:
        frames.append(data.trajectory(case_id)[::8])
    values = torch.cat(frames)
    return {"mean": values.mean(), "std": values.std().clamp_min(1e-6)}


def coarse_model(c: dict, channels: int, device: torch.device) -> FNO2d:
    m = c["model"]
    return FNO2d(channels + 1, channels, int(m["coarse_width"]), int(m["coarse_modes"]), int(m["coarse_depth"])).to(device)


def local_model(c: dict, channels: int, device: torch.device) -> FNO2d:
    m = c["model"]
    return FNO2d(4 * channels + 1, channels, int(m["local_width"]), int(m["local_modes"]), int(m["local_depth"])).to(device)


def train_coarse(data: OfficialShallowWater, c: dict, stats: dict[str, torch.Tensor], device: torch.device, checkpoint: Path) -> FNO2d:
    model = coarse_model(c, data.channels, device)
    if checkpoint.exists():
        model.load_state_dict(torch.load(checkpoint, map_location=device)["model"])
        return model.eval()
    set_seed(int(c["seed"]))
    opt = torch.optim.AdamW(model.parameters(), lr=float(c["training"]["lr"]), weight_decay=1e-5)
    stride, batch = int(c["data"]["stride"]), int(c["training"]["batch_size"])
    transitions = [(case_id, t) for case_id in data.case_ids("train") for t in range(0, 72 - stride, stride)]
    for _ in tqdm(range(int(c["training"]["coarse_epochs"])), desc="v3:coarse"):
        order = torch.randperm(len(transitions)).tolist()
        for start in range(0, len(order), batch):
            rows = [transitions[index] for index in order[start : start + batch]]
            current = torch.stack([data.frame(case, t) for case, t in rows]).to(device)
            target = torch.stack([data.frame(case, t + stride) for case, t in rows]).to(device)
            current = F.interpolate(current, size=(64, 64), mode="bilinear", align_corners=False)
            target = F.interpolate(target, size=(64, 64), mode="bilinear", align_corners=False)
            time_channel = torch.tensor([t / 71.0 for _, t in rows], device=device).view(-1, 1, 1, 1).expand(-1, 1, 64, 64)
            # CUDA FFT/einsum does not support ComplexHalf for the spectral weights.
            with torch.autocast(device_type=device.type, enabled=False):
                pred = model(torch.cat([normalized(current, stats), time_channel], dim=1))
                loss = F.mse_loss(pred, normalized(target, stats))
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    torch.save({"model": model.state_dict(), "stats": {k: v.cpu() for k, v in stats.items()}}, checkpoint)
    return model.eval()


def rollout_coarse(data: OfficialShallowWater, model: FNO2d, stats: dict[str, torch.Tensor], c: dict, case_id: int, device: torch.device) -> torch.Tensor:
    stride = int(c["data"]["stride"])
    gt = data.trajectory(case_id).to(device)
    state = gt[0:1]
    frames = [state]
    for t in range(0, 72 - stride, stride):
        coarse = F.interpolate(state, size=(64, 64), mode="bilinear", align_corners=False)
        time_channel = torch.full_like(coarse[:, :1], t / 71.0)
        next_coarse = model(torch.cat([normalized(coarse, stats), time_channel], dim=1))
        state = F.interpolate(next_coarse * stats["std"].to(device) + stats["mean"].to(device), size=(256, 256), mode="bilinear", align_corners=False)
        frames.append(state)
    return torch.cat(frames)


def write_coarse_results(data: OfficialShallowWater, model: FNO2d, stats: dict[str, torch.Tensor], c: dict, device: torch.device, result: Path) -> None:
    rows = []
    for split in ("val", "test"):
        started, trajectory_errors, final_errors, frame_errors = time.perf_counter(), [], [], []
        for case_id in tqdm(data.case_ids(split), desc=f"v3:coarse-rollout:{split}"):
            rollout = rollout_coarse(data, model, stats, c, case_id, device)
            gt = data.trajectory(case_id)[::int(c["data"]["stride"])].to(device)
            errors = relative_l2(rollout, gt)
            trajectory_errors.append(float(torch.linalg.vector_norm(rollout - gt) / torch.linalg.vector_norm(gt).clamp_min(1e-8))); final_errors.append(float(errors[-1])); frame_errors.append(float(errors.mean()))
        rows.append({"Split": split, "TrajectoryRelativeL2": np.mean(trajectory_errors), "FinalFrameRelativeL2": np.mean(final_errors), "MeanFrameRelativeL2": np.mean(frame_errors), "Cases": len(trajectory_errors), "WallTime": time.perf_counter() - started})
    pd.DataFrame(rows).to_csv(result / "global_coarse_baseline.csv", index=False)


def train_local(data: OfficialShallowWater, coarse: FNO2d, c: dict, stats: dict[str, torch.Tensor], device: torch.device, checkpoint: Path) -> FNO2d:
    model = local_model(c, data.channels, device)
    if checkpoint.exists():
        model.load_state_dict(torch.load(checkpoint, map_location=device)["model"])
        return model.eval()
    coarse.eval()
    for parameter in coarse.parameters(): parameter.requires_grad_(False)
    refiner = HybridRefiner(coarse, model, int(c["data"]["stride"]), int(c["model"]["patch_core"]), int(c["model"]["patch_halo"]), stats["mean"], stats["std"])
    opt = torch.optim.AdamW(model.parameters(), lr=float(c["training"]["lr"]), weight_decay=1e-5)
    stride = int(c["data"]["stride"]); transitions = [(case, t) for case in data.case_ids("train") for t in range(0, 72 - stride, stride)]
    patches_per = int(c["training"]["local_patches_per_transition"])
    for _ in tqdm(range(int(c["training"]["local_epochs"])), desc="v3:local"):
        for index in torch.randperm(len(transitions)).tolist():
            case, t = transitions[index]
            current, target = data.frame(case, t).unsqueeze(0).to(device), data.frame(case, t + stride).unsqueeze(0).to(device)
            with torch.no_grad(): _, provisional = refiner.coarse_next(current, t / 71.0)
            coarse_current = F.interpolate(F.interpolate(current, size=(64, 64), mode="bilinear", align_corners=False), size=(256, 256), mode="bilinear", align_corners=False)
            selected = torch.randperm(64)[:patches_per].tolist()
            inputs, labels = [], []
            for patch in selected:
                current_patch, provisional_patch = refiner.extract_patch(current, patch), refiner.extract_patch(provisional, patch)
                inputs.append(torch.cat([current_patch, provisional_patch, refiner.extract_patch(coarse_current, patch), provisional_patch - current_patch, torch.full_like(provisional_patch[:, :1], t / 71.0)], dim=1))
                labels.append(refiner.extract_patch(target - provisional, patch))
            prediction = model(torch.cat(inputs)); label = torch.cat(labels)
            loss = F.mse_loss(prediction, label)
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    torch.save({"model": model.state_dict()}, checkpoint)
    return model.eval()


def local_predictions(refiner: HybridRefiner, current: torch.Tensor, provisional: torch.Tensor, time_fraction: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Return all 64 local corrections and GT-free patch descriptors in one FNO pass."""
    patches = list(range(64))
    inputs = refiner.patch_inputs(current, provisional, patches, time_fraction)
    corrections = refiner.local_model(inputs)
    return corrections, patch_features_from_inputs(inputs, current, time_fraction)


def patch_features_from_inputs(inputs: torch.Tensor, current: torch.Tensor, time_fraction: float) -> torch.Tensor:
    """Numerical patch descriptors available at deployment without target information."""
    channels = current.shape[1]
    current_patch, provisional_patch = inputs[:, :channels], inputs[:, channels : 2 * channels]
    coarse_patch = inputs[:, 2 * channels : 3 * channels]
    temporal = provisional_patch - current_patch
    dx = provisional_patch[..., 1:] - provisional_patch[..., :-1]
    dy = provisional_patch[..., 1:, :] - provisional_patch[..., :-1, :]
    smooth = F.avg_pool2d(provisional_patch, 3, stride=1, padding=1)
    rows = torch.arange(8, device=current.device, dtype=current.dtype).repeat_interleave(8) / 7.0
    cols = torch.arange(8, device=current.device, dtype=current.dtype).repeat(8) / 7.0
    boundary = ((rows == 0) | (rows == 1) | (rows == 6) | (rows == 7) | (cols == 0) | (cols == 1) | (cols == 6) | (cols == 7)).to(current.dtype)
    def mean(value: torch.Tensor) -> torch.Tensor:
        return value.flatten(1).mean(1)
    def std(value: torch.Tensor) -> torch.Tensor:
        return value.flatten(1).std(1)
    features = torch.stack([
        mean(provisional_patch), std(provisional_patch), mean(current_patch),
        mean(temporal.abs()), std(temporal), mean(dx.abs()), mean(dy.abs()),
        mean((provisional_patch - smooth).abs()), mean((provisional_patch - coarse_patch).abs()),
        provisional_patch.flatten(1).abs().amax(1), rows, cols,
        torch.full_like(rows, float(time_fraction)), boundary,
    ], dim=1)
    return features


def immediate_patch_gains(refiner: HybridRefiner, provisional: torch.Tensor, target: torch.Tensor, corrections: torch.Tensor) -> torch.Tensor:
    """Privileged validation/train label: core-patch MSE reduction from one correction."""
    before, after = [], []
    core = slice(refiner.halo, refiner.halo + refiner.core)
    for patch, correction in enumerate(corrections.split(1, dim=0)):
        provisional_patch = refiner.extract_patch(provisional, patch)[:, :, core, core]
        target_patch = refiner.extract_patch(target, patch)[:, :, core, core]
        before.append(F.mse_loss(provisional_patch, target_patch, reduction="none").mean())
        after.append(F.mse_loss(provisional_patch + correction[:, :, core, core], target_patch, reduction="none").mean())
    return torch.stack(before) - torch.stack(after)


@torch.no_grad()
def build_utility_dataset(data: OfficialShallowWater, refiner: HybridRefiner, c: dict, cases: list[int], device: torch.device, desc: str) -> tuple[torch.Tensor, torch.Tensor]:
    features, utilities = [], []
    stride = int(c["data"]["stride"])
    for case in tqdm(cases, desc=desc):
        for time_index in range(0, 72 - stride, stride):
            current = data.frame(case, time_index).unsqueeze(0).to(device)
            target = data.frame(case, time_index + stride).unsqueeze(0).to(device)
            _, provisional = refiner.coarse_next(current, time_index / 71.0)
            corrections, feature = local_predictions(refiner, current, provisional, time_index / 71.0)
            features.append(feature.cpu())
            utilities.append(immediate_patch_gains(refiner, provisional, target, corrections).cpu())
    return torch.cat(features), torch.cat(utilities)


def correlations(prediction: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    pearson = float(np.corrcoef(prediction, target)[0, 1]) if prediction.std() > 0 and target.std() > 0 else 0.0
    spearman = float(pd.Series(prediction).rank().corr(pd.Series(target).rank())) if prediction.size else 0.0
    return pearson, spearman


def train_patch_utility(data: OfficialShallowWater, refiner: HybridRefiner, c: dict, device: torch.device, checkpoint: Path, result: Path) -> PatchUtilityNet:
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location=device)
        model = PatchUtilityNet(int(saved["feature_dim"])).to(device)
        model.load_state_dict(saved["model"])
        return model.eval()
    train_features, train_utility = build_utility_dataset(data, refiner, c, data.case_ids("train"), device, "v3:utility-train-labels")
    val_features, val_utility = build_utility_dataset(data, refiner, c, data.case_ids("val"), device, "v3:utility-val-labels")
    feature_mean, feature_std = train_features.mean(0), train_features.std(0).clamp_min(1e-6)
    utility_mean, utility_std = train_utility.mean(), train_utility.std().clamp_min(1e-8)
    model = PatchUtilityNet(train_features.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(c["training"]["lr"]), weight_decay=1e-5)
    x = ((train_features - feature_mean) / feature_std).to(device)
    y = ((train_utility - utility_mean) / utility_std).to(device)
    for _ in tqdm(range(int(c["training"]["utility_epochs"])), desc="v3:patch-utility"):
        for start in torch.randperm(len(x)).split(2048):
            prediction = model(x[start])
            loss = F.huber_loss(prediction, y[start])
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    model.eval()
    rows = []
    for split, features, target in (("train", train_features, train_utility), ("val", val_features, val_utility)):
        with torch.no_grad():
            prediction = model(((features - feature_mean) / feature_std).to(device)).cpu() * utility_std + utility_mean
        pred_np, target_np = prediction.numpy(), target.numpy()
        pearson, spearman = correlations(pred_np, target_np)
        rows.append({"Split": split, "Samples": len(target_np), "MSE": float(np.mean((pred_np - target_np) ** 2)), "Pearson": pearson, "Spearman": spearman, "PositiveSignAccuracy": float(np.mean((pred_np > 0) == (target_np > 0)))})
    pd.DataFrame(rows).to_csv(result / "patch_utility_results.csv", index=False)
    torch.save({"model": model.state_dict(), "feature_dim": train_features.shape[1], "feature_mean": feature_mean, "feature_std": feature_std, "utility_mean": utility_mean, "utility_std": utility_std}, checkpoint)
    return model


@torch.no_grad()
def select_gt_patches(refiner: HybridRefiner, current: torch.Tensor, target: torch.Tensor, time_fraction: float, count: int) -> tuple[torch.Tensor, list[int]]:
    _, provisional = refiner.coarse_next(current, time_fraction)
    if count <= 0:
        return provisional, []
    corrections, _ = local_predictions(refiner, current, provisional, time_fraction)
    gains = immediate_patch_gains(refiner, provisional, target, corrections)
    selected = torch.topk(gains, min(count, 64)).indices.tolist()
    return refiner.blend_corrections(provisional, selected, corrections[selected]), selected


def trajectory_error(states: list[torch.Tensor], targets: list[torch.Tensor]) -> tuple[float, float]:
    predicted, target = torch.cat(states), torch.cat(targets)
    return float(torch.linalg.vector_norm(predicted - target) / torch.linalg.vector_norm(target).clamp_min(1e-8)), float(relative_l2(predicted[-1:], target[-1:])[0])


@torch.no_grad()
def greedy_rollout(data: OfficialShallowWater, refiner: HybridRefiner, c: dict, case: int, budget: int, reservation: bool) -> tuple[float, float, int]:
    stride = int(c["data"]["stride"])
    time_indices = list(range(0, 72 - stride, stride))
    steps = len(time_indices)
    state = data.frame(case, 0).unsqueeze(0).to(next(refiner.coarse_model.parameters()).device)
    states, targets, remaining, used = [state], [state], budget, 0
    macros = [int(value) for value in c["oracle"]["macro_actions"]]
    for step, time_index in enumerate(time_indices):
        target = data.frame(case, time_index + stride).unsqueeze(0).to(state.device)
        steps_left = steps - step
        if reservation:
            quota = remaining / steps_left
            feasible = [q for q in macros if q <= min(remaining, int(c["oracle"]["max_refinements_per_time"]))]
            # Reserve enough calls for future physical steps: choose the smallest
            # macro action that still meets the remaining per-step quota.
            above_quota = [q for q in feasible if q >= quota]
            count = min(above_quota) if above_quota else max(feasible, default=0)
        else:
            # Fixed schedule spends the same integer number of calls as evenly as possible.
            count = min(remaining, int(np.ceil(remaining / steps_left)))
        state, selected = select_gt_patches(refiner, state, target, time_index / 71.0, count)
        remaining -= len(selected); used += len(selected)
        states.append(state); targets.append(target)
    error, final = trajectory_error(states, targets)
    return error, final, used


@torch.no_grad()
def beam_rollout(data: OfficialShallowWater, refiner: HybridRefiner, c: dict, case: int, budget: int) -> tuple[float, float, int]:
    stride, width = int(c["data"]["stride"]), int(c["oracle"]["beam_width"])
    initial = data.frame(case, 0).unsqueeze(0).to(next(refiner.coarse_model.parameters()).device)
    # Each branch stores its closed-loop native state, remaining budget and accumulated trajectory error.
    beam = [(initial, budget, 0.0, [initial], [initial], 0)]
    macros = [int(value) for value in c["oracle"]["macro_actions"]]
    for step, time_index in enumerate(range(0, 72 - stride, stride)):
        target = data.frame(case, time_index + stride).unsqueeze(0).to(initial.device)
        candidates = []
        for state, remaining, score, states, targets, used in beam:
            allowed = [q for q in macros if q <= remaining and q <= int(c["oracle"]["max_refinements_per_time"])]
            _, provisional = refiner.coarse_next(state, time_index / 71.0)
            corrections, _ = local_predictions(refiner, state, provisional, time_index / 71.0)
            gains = immediate_patch_gains(refiner, provisional, target, corrections)
            for count in allowed:
                selected = torch.topk(gains, count).indices.tolist() if count else []
                next_state = refiner.blend_corrections(provisional, selected, corrections[selected]) if selected else provisional
                frame_error = float(relative_l2(next_state, target)[0])
                candidates.append((next_state, remaining - len(selected), score + frame_error, states + [next_state], targets + [target], used + len(selected)))
        candidates.sort(key=lambda row: row[2])
        beam = candidates[:width]
    best = beam[0]
    error, final = trajectory_error(best[3], best[4])
    return error, final, best[5]


def evaluate_budget_oracles(data: OfficialShallowWater, refiner: HybridRefiner, c: dict, result: Path) -> None:
    cases = data.case_ids("val")[:16]
    rows = []
    for budget in c["oracle"]["budgets"]:
        started = time.perf_counter()
        coarse, uniform, greedy, beam, positive, usage = [], [], [], [], 0, []
        for case in tqdm(cases, desc=f"v3:oracle:B{budget}"):
            coarse_error, _, _ = greedy_rollout(data, refiner, c, case, 0, reservation=False)
            uniform_error, _, _ = greedy_rollout(data, refiner, c, case, int(budget), reservation=False)
            greedy_error, _, greedy_used = greedy_rollout(data, refiner, c, case, int(budget), reservation=True)
            beam_error, _, beam_used = beam_rollout(data, refiner, c, case, int(budget))
            coarse.append(coarse_error); uniform.append(uniform_error); greedy.append(greedy_error); beam.append(beam_error)
            positive += beam_error < greedy_error
            usage.append(beam_used)
        greedy_mean, beam_mean, uniform_mean = float(np.mean(greedy)), float(np.mean(beam)), float(np.mean(uniform))
        rows.append({"Budget": int(budget), "Cases": len(cases), "CoarseOnlyError": float(np.mean(coarse)), "UniformGTGreedyError": uniform_mean, "GreedyBudgetOracleError": greedy_mean, "BeamLongHorizonError": beam_mean, "Beam_vs_Greedy_RelativeGain": 100.0 * (greedy_mean - beam_mean) / max(greedy_mean, 1e-12), "Beam_vs_Uniform_RelativeGain": 100.0 * (uniform_mean - beam_mean) / max(uniform_mean, 1e-12), "PositiveCaseRate": 100.0 * positive / len(cases), "BudgetUsage": float(np.mean(usage)), "WallTime": time.perf_counter() - started})
    pd.DataFrame(rows).to_csv(result / "budget_oracle_headroom.csv", index=False)


def load_utility_bundle(checkpoint: Path, device: torch.device) -> tuple[PatchUtilityNet, dict[str, torch.Tensor]]:
    saved = torch.load(checkpoint, map_location=device)
    model = PatchUtilityNet(int(saved["feature_dim"])).to(device)
    model.load_state_dict(saved["model"])
    model.eval()
    return model, {key: saved[key].to(device) for key in ("feature_mean", "feature_std", "utility_mean", "utility_std")}


def policy_observation(refiner: HybridRefiner, current: torch.Tensor, provisional: torch.Tensor, time_fraction: float, remaining: int, budget: int, selected: torch.Tensor, utility_stats: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    inputs = refiner.patch_inputs(current, provisional, list(range(64)), time_fraction)
    features = patch_features_from_inputs(inputs, current, time_fraction)
    features = (features - utility_stats["feature_mean"]) / utility_stats["feature_std"]
    field = F.interpolate(torch.cat([current, provisional], dim=1), size=(32, 32), mode="bilinear", align_corners=False)
    context = torch.tensor([[remaining / max(budget, 1), 1.0 - time_fraction, float(selected.sum()) / 6.0]], device=current.device, dtype=current.dtype)
    return field, features.unsqueeze(0), context, inputs


def action_mask(selected: torch.Tensor, remaining: int, at_limit: bool) -> torch.Tensor:
    mask = torch.zeros(65, dtype=torch.bool, device=selected.device)
    if remaining <= 0 or at_limit:
        mask[:64] = True
    else:
        mask[:64] = selected
    return mask


@torch.no_grad()
def collect_ppo_episode(data: OfficialShallowWater, refiner: HybridRefiner, policy: AdaptivePatchPolicy, utility_stats: dict[str, torch.Tensor], c: dict, case: int, budget: int, sample: bool) -> tuple[list[dict[str, torch.Tensor | float | bool]], tuple[float, float, int]]:
    device, stride = next(policy.parameters()).device, int(c["data"]["stride"])
    state = data.frame(case, 0).unsqueeze(0).to(device)
    states, targets, transitions, remaining, used = [state], [state], [], budget, 0
    time_indices = list(range(0, 72 - stride, stride))
    for step, time_index in enumerate(time_indices):
        target = data.frame(case, time_index + stride).unsqueeze(0).to(device)
        _, provisional = refiner.coarse_next(state, time_index / 71.0)
        selected = torch.zeros(64, dtype=torch.bool, device=device)
        while True:
            field, features, context, inputs = policy_observation(refiner, state, provisional, time_index / 71.0, remaining, budget, selected, utility_stats)
            logits, value = policy(field, features, context)
            invalid = action_mask(selected, remaining, int(selected.sum()) >= int(c["oracle"]["max_refinements_per_time"]))
            logits = logits.masked_fill(invalid[None], -1e9)
            distribution = torch.distributions.Categorical(logits=logits)
            action = int(distribution.sample().item()) if sample else int(logits.argmax(1).item())
            record: dict[str, torch.Tensor | float | bool] = {"field": field.cpu(), "features": features.cpu(), "context": context.cpu(), "invalid": invalid.cpu(), "action": torch.tensor(action), "logprob": distribution.log_prob(torch.tensor([action], device=device)).cpu(), "value": value.cpu(), "done": False}
            if action < 64:
                correction = refiner.local_model(inputs[action : action + 1])
                provisional = refiner.blend_corrections(provisional, [action], correction)
                selected[action] = True; remaining -= 1; used += 1
                record["reward"] = -float(c["ppo"]["refinement_penalty"])
            else:
                record["reward"] = -float(relative_l2(provisional, target)[0])
                record["done"] = step == len(time_indices) - 1
                state = provisional; states.append(state); targets.append(target)
            transitions.append(record)
            if action == 64:
                break
    trajectory, final = trajectory_error(states, targets)
    return transitions, (trajectory, final, used)


def ppo_advantages(transitions: list[dict[str, torch.Tensor | float | bool]], gamma: float, gae_lambda: float) -> tuple[torch.Tensor, torch.Tensor]:
    values = torch.tensor([float(torch.as_tensor(row["value"]).item()) for row in transitions])
    rewards = torch.tensor([float(row["reward"]) for row in transitions])
    dones = torch.tensor([bool(row["done"]) for row in transitions])
    advantages = torch.zeros_like(rewards)
    running = torch.tensor(0.0)
    for index in range(len(transitions) - 1, -1, -1):
        next_value = torch.tensor(0.0) if dones[index] else values[index + 1]
        delta = rewards[index] + gamma * next_value - values[index]
        running = delta + gamma * gae_lambda * (0.0 if dones[index] else running)
        advantages[index] = running
    return advantages, advantages + values


def train_ppo_seed42(data: OfficialShallowWater, refiner: HybridRefiner, c: dict, device: torch.device, utility_checkpoint: Path, checkpoint: Path, result: Path) -> AdaptivePatchPolicy:
    utility_prior, utility_stats = load_utility_bundle(utility_checkpoint, device)
    set_seed(int(c["seed"]))
    policy = AdaptivePatchPolicy(int(utility_stats["feature_mean"].numel()), utility_prior).to(device)
    if checkpoint.exists():
        policy.load_state_dict(torch.load(checkpoint, map_location=device)["model"])
        return policy.eval()
    opt = torch.optim.AdamW([parameter for parameter in policy.parameters() if parameter.requires_grad], lr=float(c["ppo"]["lr"]))
    generator, history = np.random.default_rng(int(c["seed"])), []
    for update in tqdm(range(int(c["ppo"]["updates"])), desc="v3:ppo-seed42"):
        transitions, episode_errors = [], []
        for _ in range(int(c["ppo"]["episodes_per_update"])):
            case = int(generator.choice(data.case_ids("train")))
            budget = int(generator.choice(c["oracle"]["budgets"]))
            episode, metrics = collect_ppo_episode(data, refiner, policy, utility_stats, c, case, budget, sample=True)
            transitions.extend(episode); episode_errors.append(metrics[0])
        advantages, returns = ppo_advantages(transitions, float(c["ppo"]["gamma"]), float(c["ppo"]["gae_lambda"]))
        advantages = ((advantages - advantages.mean()) / advantages.std().clamp_min(1e-6)).to(device)
        returns = returns.to(device)
        fields = torch.cat([torch.as_tensor(row["field"]) for row in transitions]).to(device)
        features = torch.cat([torch.as_tensor(row["features"]) for row in transitions]).to(device)
        context = torch.cat([torch.as_tensor(row["context"]) for row in transitions]).to(device)
        invalid = torch.stack([torch.as_tensor(row["invalid"]) for row in transitions]).to(device)
        actions = torch.stack([torch.as_tensor(row["action"]) for row in transitions]).to(device)
        old_logprob = torch.cat([torch.as_tensor(row["logprob"]) for row in transitions]).to(device)
        for _ in range(int(c["ppo"]["epochs_per_update"])):
            for indices in torch.randperm(len(transitions), device=device).split(int(c["ppo"]["minibatch_size"])):
                logits, values = policy(fields[indices], features[indices], context[indices])
                distribution = torch.distributions.Categorical(logits=logits.masked_fill(invalid[indices], -1e9))
                ratio = (distribution.log_prob(actions[indices]) - old_logprob[indices]).exp()
                clipped = ratio.clamp(1.0 - float(c["ppo"]["clip_ratio"]), 1.0 + float(c["ppo"]["clip_ratio"])) * advantages[indices]
                actor = -torch.minimum(ratio * advantages[indices], clipped).mean()
                critic = F.mse_loss(values, returns[indices])
                loss = actor + 0.5 * critic - float(c["ppo"]["entropy_weight"]) * distribution.entropy().mean()
                opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0); opt.step()
        history.append({"Update": update + 1, "Transitions": len(transitions), "MeanTrainTrajectoryError": float(np.mean(episode_errors))})
    pd.DataFrame(history).to_csv(result / "ppo_seed42_training.csv", index=False)
    torch.save({"model": policy.state_dict()}, checkpoint)
    return policy.eval()


@torch.no_grad()
def baseline_rollout(data: OfficialShallowWater, refiner: HybridRefiner, c: dict, case: int, budget: int, method: str, utility_model: PatchUtilityNet | None, utility_stats: dict[str, torch.Tensor] | None, policy: AdaptivePatchPolicy | None) -> tuple[float, float, int]:
    if method == "PPO":
        assert policy is not None and utility_stats is not None
        _, metrics = collect_ppo_episode(data, refiner, policy, utility_stats, c, case, budget, sample=False)
        return metrics
    device = next(refiner.coarse_model.parameters()).device
    stride = int(c["data"]["stride"])
    generator = torch.Generator(device=device).manual_seed(case + budget)
    state, states, targets, remaining, used = data.frame(case, 0).unsqueeze(0).to(device), [], [], budget, 0
    states.append(state); targets.append(state)
    time_indices = list(range(0, 72 - stride, stride))
    for step, time_index in enumerate(time_indices):
        target = data.frame(case, time_index + stride).unsqueeze(0).to(device)
        _, provisional = refiner.coarse_next(state, time_index / 71.0)
        count = min(remaining, int(np.ceil(remaining / (len(time_indices) - step))), int(c["oracle"]["max_refinements_per_time"]))
        if count:
            inputs = refiner.patch_inputs(state, provisional, list(range(64)), time_index / 71.0)
            features = patch_features_from_inputs(inputs, state, time_index / 71.0)
            if method == "Random":
                scores = torch.rand(64, generator=generator, device=device)
            elif method == "GradientHeuristic":
                scores = features[:, 5] + features[:, 6] + features[:, 7]
            else:
                assert utility_model is not None and utility_stats is not None
                scores = utility_model(((features - utility_stats["feature_mean"]) / utility_stats["feature_std"]))
            selected = torch.topk(scores, count).indices.tolist()
            corrections = refiner.local_model(inputs[selected])
            state = refiner.blend_corrections(provisional, selected, corrections)
            remaining -= len(selected); used += len(selected)
        else:
            state = provisional
        states.append(state); targets.append(target)
    trajectory, final = trajectory_error(states, targets)
    return trajectory, final, used


def evaluate_seed42(data: OfficialShallowWater, refiner: HybridRefiner, c: dict, device: torch.device, utility_checkpoint: Path, policy: AdaptivePatchPolicy, result: Path) -> None:
    utility_model, utility_stats = load_utility_bundle(utility_checkpoint, device)
    rows = []
    for budget in c["oracle"]["budgets"]:
        for method in ("Random", "GradientHeuristic", "SupervisedMyopic", "PPO"):
            started = time.perf_counter()
            metrics = [baseline_rollout(data, refiner, c, case, int(budget), method, utility_model, utility_stats, policy) for case in tqdm(data.case_ids("val"), desc=f"v3:{method}:B{budget}")]
            rows.append({"Method": method, "Budget": int(budget), "Cases": len(metrics), "MeanTrajectoryError": float(np.mean([row[0] for row in metrics])), "FinalError": float(np.mean([row[1] for row in metrics])), "WallTime": time.perf_counter() - started, "LocalOperatorCalls": float(np.mean([row[2] for row in metrics]))})
    table = pd.DataFrame(rows)
    table.to_csv(result / "rl_seed42.csv", index=False)
    table[["Method", "Budget", "MeanTrajectoryError", "FinalError", "WallTime", "LocalOperatorCalls"]].to_csv(result / "accuracy_compute_pareto.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["coarse", "local", "utility", "oracle", "ppo", "all"], default="all")
    args = parser.parse_args()
    c, device = config(), get_device(str(config()["device"]))
    result, checkpoint = paths()
    data = OfficialShallowWater(ROOT / str(c["data"]["official_dir"]), int(c["data"]["train_cases"]), int(c["data"]["validation_cases"]), int(c["data"]["test_cases"]))
    stats = train_stats(data, c)
    model = train_coarse(data, c, stats, device, checkpoint / "coarse_transition.pt")
    write_coarse_results(data, model, stats, c, device, result)
    if args.stage in {"local", "utility", "oracle", "ppo", "all"}:
        local = train_local(data, model, c, stats, device, checkpoint / "local_corrector.pt")
        refiner = HybridRefiner(model, local, int(c["data"]["stride"]), int(c["model"]["patch_core"]), int(c["model"]["patch_halo"]), stats["mean"], stats["std"])
        if args.stage in {"local", "utility", "oracle", "ppo", "all"}:
            rows = []
            for case in tqdm(data.case_ids("val"), desc="v3:local-check"):
                current, target = data.frame(case, 0).unsqueeze(0).to(device), data.frame(case, int(c["data"]["stride"])).unsqueeze(0).to(device)
                _, provisional = refiner.coarse_next(current, 0.0)
                refined = refiner.apply_patches(current, provisional, list(range(64)), 0.0)
                rows.append({"Case": case, "ProvisionalRelativeL2": float(relative_l2(provisional, target)[0]), "AllPatchRelativeL2": float(relative_l2(refined, target)[0])})
            pd.DataFrame(rows).to_csv(result / "local_corrector_results.csv", index=False)
        if args.stage in {"utility", "oracle", "ppo", "all"}:
            train_patch_utility(data, refiner, c, device, checkpoint / "patch_utility.pt", result)
        if args.stage in {"oracle", "all"}:
            evaluate_budget_oracles(data, refiner, c, result)
        if args.stage in {"ppo", "all"}:
            policy = train_ppo_seed42(data, refiner, c, device, checkpoint / "patch_utility.pt", checkpoint / "ppo_seed42.pt", result)
            evaluate_seed42(data, refiner, c, device, checkpoint / "patch_utility.pt", policy, result)
    (result / "summary.json").write_text(json.dumps({"stage": args.stage, "metadata": data.metadata}, indent=2), encoding="utf-8")
    print(f"Solver V3 {args.stage} stage completed.")


if __name__ == "__main__":
    main()
