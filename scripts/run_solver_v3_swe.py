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
from src.solver_v3.models.patch_policy import PatchUtilityNet
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
    return corrections, features


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
    stride, steps = int(c["data"]["stride"]), 18
    state = data.frame(case, 0).unsqueeze(0).to(next(refiner.coarse_model.parameters()).device)
    states, targets, remaining, used = [state], [state], budget, 0
    macros = [int(value) for value in c["oracle"]["macro_actions"]]
    for step, time_index in enumerate(range(0, 72 - stride, stride)):
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["coarse", "local", "utility", "oracle", "all"], default="all")
    args = parser.parse_args()
    c, device = config(), get_device(str(config()["device"]))
    result, checkpoint = paths()
    data = OfficialShallowWater(ROOT / str(c["data"]["official_dir"]), int(c["data"]["train_cases"]), int(c["data"]["validation_cases"]), int(c["data"]["test_cases"]))
    stats = train_stats(data, c)
    model = train_coarse(data, c, stats, device, checkpoint / "coarse_transition.pt")
    write_coarse_results(data, model, stats, c, device, result)
    if args.stage in {"local", "utility", "oracle", "all"}:
        local = train_local(data, model, c, stats, device, checkpoint / "local_corrector.pt")
        refiner = HybridRefiner(model, local, int(c["data"]["stride"]), int(c["model"]["patch_core"]), int(c["model"]["patch_halo"]), stats["mean"], stats["std"])
        if args.stage in {"local", "utility", "oracle", "all"}:
            rows = []
            for case in tqdm(data.case_ids("val"), desc="v3:local-check"):
                current, target = data.frame(case, 0).unsqueeze(0).to(device), data.frame(case, int(c["data"]["stride"])).unsqueeze(0).to(device)
                _, provisional = refiner.coarse_next(current, 0.0)
                refined = refiner.apply_patches(current, provisional, list(range(64)), 0.0)
                rows.append({"Case": case, "ProvisionalRelativeL2": float(relative_l2(provisional, target)[0]), "AllPatchRelativeL2": float(relative_l2(refined, target)[0])})
            pd.DataFrame(rows).to_csv(result / "local_corrector_results.csv", index=False)
        if args.stage in {"utility", "oracle", "all"}:
            train_patch_utility(data, refiner, c, device, checkpoint / "patch_utility.pt", result)
        if args.stage in {"oracle", "all"}:
            evaluate_budget_oracles(data, refiner, c, result)
    (result / "summary.json").write_text(json.dumps({"stage": args.stage, "metadata": data.metadata}, indent=2), encoding="utf-8")
    print(f"Solver V3 {args.stage} stage completed.")


if __name__ == "__main__":
    main()
