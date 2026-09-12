from __future__ import annotations

import argparse
import json
import sys
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

import run_solver_v5_brusselator as v1
from src.utils.seed import get_device, set_seed


V1_VAL_ONE_STEP = 0.3411626282334328


def cfg() -> dict:
    return yaml.safe_load((ROOT / "configs/solver_v5_brusselator_v2.yaml").read_text(encoding="utf-8"))


def paths() -> tuple[Path, Path]:
    result = ROOT / "results/solver_v5/brusselator_v2"
    checkpoint = ROOT / "checkpoints/solver_v5/brusselator_v2"
    result.mkdir(parents=True, exist_ok=True)
    checkpoint.mkdir(parents=True, exist_ok=True)
    return result, checkpoint


def normalized_input(previous: torch.Tensor, current: torch.Tensor, force: torch.Tensor, time: float, stats: dict[str, torch.Tensor]) -> torch.Tensor:
    device = current.device
    mean, std = stats["mean"].to(device), stats["std"].to(device)
    force_value = ((force.to(device) - stats["force_mean"].to(device)) / stats["force_std"].to(device)).view(-1, 1, 1, 1)
    force_field = force_value.expand_as(current[:, :1])
    time_field = torch.full_like(current[:, :1], time)
    norm = lambda value: (value - mean) / std
    return torch.cat([force_field, norm(previous), norm(current), norm(current - previous), time_field], dim=1)


@torch.no_grad()
def rollout_states(data, coarse: torch.nn.Module, stats: dict[str, torch.Tensor], case: int) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    device = next(coarse.parameters()).device
    previous = current = data.frame(case, 0).unsqueeze(0).to(device)
    states, targets = [current], [current]
    mean, std = stats["mean"].to(device), stats["std"].to(device)
    for step in range(38):
        force = data.forcing(case, step).reshape(1).to(device)
        normalized_prediction = coarse(normalized_input(previous, current, force, step / 38, stats))
        prediction = normalized_prediction * std + mean
        # This is the existing environment's finite-value guard, not a new solver mechanism.
        prediction = torch.nan_to_num(prediction, nan=float(mean), posinf=float(mean + 8 * std), neginf=float(mean - 8 * std)).clamp(mean - 8 * std, mean + 8 * std)
        previous, current = current, prediction
        states.append(current)
        targets.append(data.frame(case, step + 1).unsqueeze(0).to(device))
    return states, targets


@torch.no_grad()
def rollout_metrics(data, coarse: torch.nn.Module, stats: dict[str, torch.Tensor], split: str, cases: np.ndarray | None = None) -> tuple[pd.DataFrame, dict[str, float]]:
    if cases is None:
        cases = np.asarray(data.case_ids(split))
    by_step = [[] for _ in range(38)]
    trajectory_relative, final_relative = [], []
    trajectory_sq_error = trajectory_values = final_sq_error = final_values = 0.0
    for case in tqdm(cases, desc=f"v5b-v2:coarse:{split}", leave=False):
        states, targets = rollout_states(data, coarse, stats, int(case))
        predicted, truth = torch.cat(states), torch.cat(targets)
        trajectory_relative.append(float(torch.linalg.vector_norm(predicted - truth) / torch.linalg.vector_norm(truth).clamp_min(1e-8)))
        final_relative.append(float(v1.relative_l2(predicted[-1:], truth[-1:])[0]))
        trajectory_sq_error += float((predicted - truth).square().sum())
        trajectory_values += predicted.numel()
        final_sq_error += float((predicted[-1:] - truth[-1:]).square().sum())
        final_values += predicted[-1:].numel()
        for step, (prediction, target) in enumerate(zip(states[1:], targets[1:]), start=1):
            error = prediction - target
            target_norm = torch.linalg.vector_norm(target).clamp_min(1e-8)
            by_step[step - 1].append({
                "relative": float(torch.linalg.vector_norm(error) / target_norm),
                "rmse": float(error.square().mean().sqrt()),
                "prediction_rms": float(prediction.square().mean().sqrt()),
                "target_rms": float(target.square().mean().sqrt()),
                "prediction_abs": float(prediction.abs().mean()),
                "target_abs": float(target.abs().mean()),
                "prediction_max": float(prediction.abs().amax()),
                "target_max": float(target.abs().amax()),
            })
    rows = []
    for step, values in enumerate(by_step, start=1):
        frame = pd.DataFrame(values)
        prediction_rms = frame.prediction_rms.mean()
        target_rms = frame.target_rms.mean()
        rows.append({
            "Step": step,
            "MeanRelativeL2": frame.relative.mean(),
            "MedianRelativeL2": frame.relative.median(),
            "MeanRMSE": frame.rmse.mean(),
            "MeanPredictionRMS": prediction_rms,
            "MeanTargetRMS": target_rms,
            "MeanAbsPrediction": frame.prediction_abs.mean(),
            "MeanAbsTarget": frame.target_abs.mean(),
            "MeanMaxAbsPrediction": frame.prediction_max.mean(),
            "MeanMaxAbsTarget": frame.target_max.mean(),
            "RMSRatio": prediction_rms / max(target_rms, 1e-12),
        })
    curve = pd.DataFrame(rows)
    final = curve.iloc[-1]
    summary = {
        "OneStepRelativeL2": float(curve.iloc[0].MeanRelativeL2),
        "TrajectoryRelativeL2": float(np.mean(trajectory_relative)),
        "TrajectoryRMSE": float(np.sqrt(trajectory_sq_error / max(trajectory_values, 1))),
        "FinalFrameRelativeL2": float(np.mean(final_relative)),
        "FinalFrameRMSE": float(np.sqrt(final_sq_error / max(final_values, 1))),
        "PredictionRMSRatio": float(final.RMSRatio),
        "MaxAmplitudeRatio": float(final.MeanMaxAbsPrediction / max(final.MeanMaxAbsTarget, 1e-12)),
        "Cases": int(len(cases)),
    }
    return curve, summary


@torch.no_grad()
def v1_diagnostic(data, config: dict, device: torch.device, out: Path) -> None:
    destination = out / "v1_rollout_stability_diagnostic.csv"
    if destination.exists():
        return
    v1_config = v1.cfg()
    coarse, _ = v1.models(v1_config, device)
    saved = torch.load(ROOT / "checkpoints/solver_v5/brusselator/conditional_coarse.pt", map_location=device)
    coarse.load_state_dict(saved["model"])
    stable_stats = {name: value.to(device) for name, value in saved["stats"].items()}
    curve, summary = rollout_metrics(data, coarse.eval(), stable_stats, "val")
    curve.to_csv(destination, index=False)
    (out / "v1_rollout_stability_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def validation_cases(data, config: dict) -> np.ndarray:
    rng = np.random.default_rng(int(config["data"]["split_seed"]))
    return np.sort(rng.choice(np.asarray(data.case_ids("val")), int(config["training"]["coarse_validation_cases"]), replace=False))


def rollout_loss(coarse: torch.nn.Module, data, batch: list[tuple[int, int]], config: dict, stats: dict[str, torch.Tensor], device: torch.device) -> tuple[torch.Tensor, dict[str, float]]:
    horizon = int(config["training"]["coarse_rollout_horizon"])
    previous = torch.stack([data.frame(case, max(start - 1, 0)) for case, start in batch]).to(device)
    current = torch.stack([data.frame(case, start) for case, start in batch]).to(device)
    starts = torch.tensor([start for _, start in batch], device=device)
    mean, std = stats["mean"].to(device), stats["std"].to(device)
    rollout = torch.zeros((), device=device)
    anchor = torch.zeros((), device=device)
    stability = torch.zeros((), device=device)
    terms = 0
    for offset in range(horizon):
        active = starts + offset < 38
        if not active.any():
            break
        step = starts + offset
        force = torch.stack([data.forcing(case, min(int(start + offset), 38)) for case, start in batch]).to(device)
        target = torch.stack([data.frame(case, min(int(start + offset + 1), 38)) for case, start in batch]).to(device)
        input_time = (step.float() / 38).tolist()
        # FNO receives a per-sample time channel, so form it without collapsing distinct t0 values.
        model_input = normalized_input(previous, current, force, 0.0, stats)
        model_input[:, -1].copy_((step.float() / 38).view(-1, 1, 1).expand_as(model_input[:, -1]))
        prediction_normalized = coarse(model_input)
        target_normalized = (target - mean) / std
        step_loss = F.mse_loss(prediction_normalized[active], target_normalized[active])
        rollout = rollout + step_loss
        if offset == 0:
            anchor = step_loss
        prediction = prediction_normalized * std + mean
        prediction_rms = prediction[active].square().mean(dim=(1, 2, 3)).sqrt()
        target_rms = target[active].square().mean(dim=(1, 2, 3)).sqrt().clamp_min(1e-8)
        stability = stability + F.relu(prediction_rms / target_rms - float(config["training"]["coarse_stability_alpha"])).square().mean()
        previous, current = current, prediction
        terms += 1
    total = rollout / terms + float(config["training"]["coarse_one_step_weight"]) * anchor + float(config["training"]["coarse_stability_weight"]) * stability / terms
    return total, {"rollout": float((rollout / terms).detach()), "anchor": float(anchor.detach()), "stability": float((stability / terms).detach())}


def train_coarse_v2(data, config: dict, stats: dict[str, torch.Tensor], device: torch.device, checkpoint: Path, out: Path) -> torch.nn.Module:
    path = checkpoint / "conditional_coarse.pt"
    coarse, _ = v1.models(config, device)
    if path.exists():
        coarse.load_state_dict(torch.load(path, map_location=device)["model"])
        return coarse.eval()
    set_seed(int(config["seed"]))
    optimizer = torch.optim.AdamW(coarse.parameters(), lr=float(config["training"]["coarse_lr"]), weight_decay=1e-5)
    rows = [(int(case), step) for case in data.case_ids("train") for step in range(38)]
    validation = validation_cases(data, config)
    best = float("inf")
    history = []
    for epoch in tqdm(range(int(config["training"]["coarse_epochs"])), desc="v5b-v2:coarse"):
        coarse.train()
        losses = []
        for indices in torch.randperm(len(rows)).split(int(config["training"]["coarse_batch_size"])):
            batch = [rows[index] for index in indices.tolist()]
            loss, components = rollout_loss(coarse, data, batch, config, stats, device)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite V2 coarse rollout loss; checkpoint was not written.")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(coarse.parameters(), 1.0)
            optimizer.step()
            losses.append((float(loss.detach()), components))
        record = {"Epoch": epoch + 1, "TrainLoss": float(np.mean([item[0] for item in losses])), "TrainRolloutLoss": float(np.mean([item[1]["rollout"] for item in losses])), "TrainOneStepLoss": float(np.mean([item[1]["anchor"] for item in losses])), "TrainStabilityPenalty": float(np.mean([item[1]["stability"] for item in losses]))}
        interval = int(config["training"]["coarse_validation_interval"])
        if (epoch + 1) % interval == 0 or epoch + 1 == int(config["training"]["coarse_epochs"]):
            curve, metrics = rollout_metrics(data, coarse.eval(), stats, "val", validation)
            record.update({f"Val{name}": value for name, value in metrics.items() if name != "Cases"})
            if metrics["TrajectoryRelativeL2"] < best:
                best = metrics["TrajectoryRelativeL2"]
                torch.save({"model": coarse.state_dict(), "stats": {name: value.cpu() for name, value in stats.items()}, "validation": metrics, "epoch": epoch + 1}, path)
        history.append(record)
    pd.DataFrame(history).to_csv(out / "coarse_training_history.csv", index=False)
    if not path.exists():
        raise RuntimeError("No V2 coarse validation checkpoint was written.")
    coarse.load_state_dict(torch.load(path, map_location=device)["model"])
    return coarse.eval()


def write_v2_coarse_results(data, coarse: torch.nn.Module, stats: dict[str, torch.Tensor], config: dict, out: Path) -> dict[str, dict[str, float]]:
    summaries = {}
    for split in ("val", "test"):
        curve, summary = rollout_metrics(data, coarse, stats, split)
        summaries[split] = summary
        if split == "val":
            curve.to_csv(out / "rollout_error_by_time.csv", index=False)
    pd.DataFrame([{"Split": split, **values} for split, values in summaries.items()]).to_csv(out / "conditional_coarse_results.csv", index=False)
    v1_summary = json.loads((out / "v1_rollout_stability_summary.json").read_text(encoding="utf-8"))
    comparison = []
    for key, label in (("OneStepRelativeL2", "OneStepRelL2"), ("TrajectoryRelativeL2", "TrajectoryRelL2"), ("FinalFrameRelativeL2", "FinalFrameRelL2"), ("TrajectoryRMSE", "TrajectoryRMSE"), ("FinalFrameRMSE", "FinalRMSE")):
        old, new = float(v1_summary[key]), float(summaries["val"][key])
        comparison.append({"Metric": label, "V1": old, "V2": new, "Improvement": (old - new) / max(abs(old), 1e-12)})
    pd.DataFrame(comparison).to_csv(out / "brusselator_solver_stability_comparison.csv", index=False)
    return summaries


def enforce_gate(summary: dict[str, float]) -> None:
    if summary["TrajectoryRelativeL2"] >= 4.0:
        raise RuntimeError(f"V2 coarse validation gate failed: trajectory relative L2={summary['TrajectoryRelativeL2']:.4f} (required < 4.0). Downstream training was not started.")
    if summary["OneStepRelativeL2"] > 1.15 * V1_VAL_ONE_STEP:
        raise RuntimeError(f"V2 one-step gate failed: relative L2={summary['OneStepRelativeL2']:.4f} exceeds 15% degradation from V1. Downstream training was not started.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("diagnostic", "coarse", "local", "selector", "teacher", "policy", "evaluate", "all"), default="all")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    config = cfg()
    config["seed"] = args.seed
    device = get_device(config["device"])
    set_seed(args.seed)
    out, checkpoint = paths()
    data = v1.data(config)
    statistics = v1.stats(data)
    v1_diagnostic(data, config, device, out)
    if args.stage == "diagnostic":
        return
    coarse = train_coarse_v2(data, config, statistics, device, checkpoint, out)
    summaries = write_v2_coarse_results(data, coarse, statistics, config, out)
    enforce_gate(summaries["val"])
    if args.stage == "coarse":
        return
    local = v1.train_local(data, coarse, config, statistics, device, checkpoint)
    environment = v1.refiner(coarse, local, config, statistics)
    if args.stage == "local":
        return
    selector_data = v1.build_selector_data(data, environment, config, checkpoint)
    selector, selector_stats = v1.train_selector(selector_data, config, device, checkpoint)
    if args.stage == "selector":
        return
    teacher = v1.beam_teacher(data, environment, selector, selector_stats, config, checkpoint)
    beam = v1.train_bc(teacher, config, device, checkpoint)
    if args.stage == "teacher":
        return
    immediate = v1.train_rvpi(data, environment, selector, selector_stats, beam, config, device, checkpoint, args.seed, immediate=True)
    rvpi = v1.train_rvpi(data, environment, selector, selector_stats, beam, config, device, checkpoint, args.seed, immediate=False)
    if args.stage == "policy":
        return
    methods = (("CoarseOnly", None), ("RandomMacro", beam), ("UniformMacro", beam), ("GradientMacro", beam), ("SetAwareMyopicMacro", beam), ("BeamBC", beam), ("ImmediateOnlyPI", immediate), ("RVPI", rvpi))
    frames = []
    for name, policy in methods:
        frame = v1.evaluate(data, environment, selector, selector_stats, policy, name, "test", args.seed)
        frame.insert(1, "Seed", args.seed)
        frames.append(frame)
    pd.concat(frames, ignore_index=True).to_csv(out / f"per_case_comparison_seed{args.seed}.csv", index=False)
    print(f"Brusselator V2 seed {args.seed} evaluation written to {out}")


if __name__ == "__main__":
    main()
