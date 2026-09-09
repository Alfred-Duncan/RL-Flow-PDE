from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch
import yaml
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_brusselator_oracle import (
    fno_predictions,
    relative_l2,
    train_fno,
)
from src.data.official_brusselator import OfficialBrusselatorDataset
from src.solver_v2.benchmarks.brusselator.pca_correction_basis import PCACorrectionBasis
from src.utils.seed import get_device, set_seed


PCA_CACHE_VERSION = "pca_corrections_v2"
PCA_DIMS = (16, 32, 64, 96)
CONTINUATIONS = ("FixedContraction", "GreedyLocal")


def load_config() -> dict:
    with open(ROOT / "configs" / "default.yaml", "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def output_dir() -> Path:
    path = ROOT / "results" / "solver_v2" / "brusselator"
    path.mkdir(parents=True, exist_ok=True)
    return path


def checkpoint_path() -> Path:
    path = ROOT / "checkpoints" / "solver_v2" / "brusselator"
    path.mkdir(parents=True, exist_ok=True)
    return path / f"{PCA_CACHE_VERSION}.pt"


def rms(values: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.mean(values.square(), dim=tuple(range(1, values.ndim))))


@torch.no_grad()
def fit_pca_from_train(config: dict, device: torch.device) -> tuple[PCACorrectionBasis, pd.DataFrame]:
    """Fit PCA exclusively from train FNO residuals and intermediate train trajectories."""
    cache = checkpoint_path()
    if cache.exists():
        payload = torch.load(cache, map_location=device)
        basis = PCACorrectionBasis(
            mean_delta=payload["mean_delta"].to(device),
            components=payload["components"].to(device),
            latent_mean=payload["latent_mean"].to(device),
            latent_std=payload["latent_std"].to(device),
            max_step_rms=float(payload["max_step_rms"]),
            lower=float(payload["lower"]),
            upper=float(payload["upper"]),
        )
        return basis, pd.DataFrame(payload["quality"])

    c = config["brusselator"]
    set_seed(int(c["fno_seed"]))
    model, stats = train_fno(config, device)
    train = OfficialBrusselatorDataset(ROOT / str(c["official_npz"]), "train", split_seed=int(c["split_seed"]))
    prediction, target, _ = fno_predictions(model, train, stats, device)
    correction = (target - prediction).to(device)
    # Intermediate states u_alpha = u_fno + alpha * (u_gt - u_fno), alpha in {0,.25,.5,.75}.
    samples = torch.cat([factor * correction for factor in (1.0, 0.75, 0.5, 0.25)], dim=0)
    matrix = samples.reshape(samples.shape[0], -1)
    mean = matrix.mean(dim=0)
    centered = matrix - mean
    max_dim = min(max(PCA_DIMS), centered.shape[0] - 1, centered.shape[1])
    _, singular, components = torch.pca_lowrank(centered, q=max_dim, center=False, niter=4)
    total_energy = centered.square().sum().clamp_min(1e-8)
    all_latents = centered @ components
    quality_rows = []
    for latent_dim in PCA_DIMS:
        if latent_dim > max_dim:
            continue
        reconstruction = mean + all_latents[:, :latent_dim] @ components[:, :latent_dim].T
        reconstruction_error = torch.linalg.vector_norm(matrix - reconstruction, dim=1) / torch.linalg.vector_norm(matrix, dim=1).clamp_min(1e-8)
        explained = singular[:latent_dim].square().sum() / total_energy
        quality_rows.append({
            "LatentDim": latent_dim,
            "ExplainedVarianceRatio": float(explained.cpu()),
            "ReconstructionRelativeL2Mean": float(reconstruction_error.mean().cpu()),
            "ReconstructionRelativeL2Median": float(reconstruction_error.median().cpu()),
            "TrainSamples": int(matrix.shape[0]),
        })
    latent_mean = all_latents.mean(dim=0)
    latent_std = all_latents.std(dim=0).clamp_min(1e-6)
    max_step_rms = torch.quantile(rms(correction), float(c["pca_step_rms_quantile"]))
    basis = PCACorrectionBasis(
        mean_delta=mean.reshape(39, 14, 14), components=components,
        latent_mean=latent_mean, latent_std=latent_std,
        max_step_rms=float(max_step_rms.cpu()),
        lower=float(stats["target_min"]), upper=float(stats["target_max"]),
    )
    quality = pd.DataFrame(quality_rows)
    torch.save({
        "mean_delta": basis.mean_delta.cpu(), "components": basis.components.cpu(),
        "latent_mean": basis.latent_mean.cpu(), "latent_std": basis.latent_std.cpu(),
        "max_step_rms": basis.max_step_rms, "lower": basis.lower, "upper": basis.upper,
        "quality": quality.to_dict(orient="records"),
    }, cache)
    return basis, quality


def candidate_actions(basis: PCACorrectionBasis, latent_dim: int, count: int, seed: int, device: torch.device) -> torch.Tensor:
    """Shared, GT-free local candidates around the zero/current reference action."""
    if count < 32:
        raise ValueError("PCA diagnostic requires at least 32 candidates.")
    generator = torch.Generator(device=device.type).manual_seed(int(seed))
    std = basis.latent_std[:latent_dim].to(device)
    mean = basis.latent_mean[:latent_dim].to(device)
    values = [torch.zeros(latent_dim, device=device)]  # zero and current/reference action coincide initially
    radii = (0.10, 0.25, 0.50, 0.75)
    while len(values) < count:
        direction = torch.randn(latent_dim, generator=generator, device=device)
        direction = direction / torch.sqrt(torch.mean(direction.square())).clamp_min(1e-8)
        radius = radii[(len(values) - 1) % len(radii)]
        probe = radius * std * direction
        values.extend([probe, -probe])
    actions = torch.stack(values[:count])
    return actions.clamp(mean - 3.0 * std, mean + 3.0 * std)


def reward(before: torch.Tensor, after: torch.Tensor, actions: torch.Tensor, target: torch.Tensor, action_cost: float) -> torch.Tensor:
    target_batch = target.expand(before.shape[0], -1, -1, -1)
    return torch.log((relative_l2(before, target_batch) + 1e-8) / (relative_l2(after, target_batch) + 1e-8)) - action_cost * torch.mean(actions.square(), dim=1)


@torch.no_grad()
def immediate_candidates(basis: PCACorrectionBasis, current: torch.Tensor, target: torch.Tensor, actions: torch.Tensor, latent_dim: int, action_cost: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    candidate_state, correction_norm, _ = basis.advance(current, actions, latent_dim)
    return candidate_state, reward(current.expand_as(candidate_state), candidate_state, actions, target, action_cost), correction_norm


@torch.no_grad()
def greedy_local_continuation(basis: PCACorrectionBasis, states: torch.Tensor, target: torch.Tensor, actions: torch.Tensor, latent_dim: int, action_cost: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact immediate greedy selection using L2 inner products, with a clipping fallback."""
    deltas, _ = basis.decode(actions, latent_dim)
    flat_states = states.reshape(states.shape[0], -1)
    flat_target = target.reshape(1, -1)
    flat_deltas = deltas.reshape(deltas.shape[0], -1)
    target_norm_sq = flat_target.square().sum().clamp_min(1e-8)
    errors = flat_states - flat_target
    error_sq = errors.square().sum(dim=1, keepdim=True)
    candidate_sq = error_sq + flat_deltas.square().sum(dim=1).unsqueeze(0) + 2.0 * (errors @ flat_deltas.T)
    candidate_sq = candidate_sq.clamp_min(1e-12)
    scores = 0.5 * torch.log(error_sq.clamp_min(1e-12) / candidate_sq) - action_cost * torch.mean(actions.square(), dim=1).unsqueeze(0)

    # The closed-form path is exact when no per-field range clipping can occur. For the few
    # boundary states, fall back to the original field evaluation to preserve the transition.
    state_min, state_max = states.amin(dim=(1, 2, 3)), states.amax(dim=(1, 2, 3))
    delta_min, delta_max = deltas.amin(dim=(1, 2, 3)), deltas.amax(dim=(1, 2, 3))
    safe = (state_min[:, None] + delta_min[None] >= basis.lower) & (state_max[:, None] + delta_max[None] <= basis.upper)
    for state_index in torch.where(~safe.all(dim=1))[0].tolist():
        _, exact_scores, _ = immediate_candidates(basis, states[state_index], target, actions, latent_dim, action_cost)
        scores[state_index] = exact_scores
    best = scores.argmax(dim=1)
    next_states, _, _ = basis.advance(states, actions[best], latent_dim)
    exact_reward = reward(states, next_states, actions[best], target, action_cost)
    return next_states, exact_reward


@torch.no_grad()
def rollout_candidates(basis: PCACorrectionBasis, current: torch.Tensor, baseline: torch.Tensor, target: torch.Tensor, actions: torch.Tensor, latent_dim: int, horizon: int, continuation: str, retention: float, action_cost: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    first, immediate, correction_norm = immediate_candidates(basis, current, target, actions, latent_dim, action_cost)
    total = immediate.clone()
    states = first
    for _ in range(1, horizon):
        if continuation == "FixedContraction":
            next_states = basis.fixed_contraction(states, baseline.unsqueeze(0).expand_as(states), retention)
            zero = torch.zeros_like(actions)
            step_reward = reward(states, next_states, zero, target, action_cost)
        elif continuation == "GreedyLocal":
            next_states, step_reward = greedy_local_continuation(basis, states, target, actions, latent_dim, action_cost)
        else:
            raise ValueError(f"Unknown continuation: {continuation}")
        total += step_reward
        states = next_states
    final_error = relative_l2(states, target.expand_as(states))
    return immediate, total, final_error, correction_norm


def controllability(basis: PCACorrectionBasis, prediction: torch.Tensor, target: torch.Tensor, case_ids: torch.Tensor, config: dict) -> pd.DataFrame:
    c = config["brusselator"]
    rows = []
    for latent_dim in PCA_DIMS:
        for index in tqdm(range(len(prediction)), desc=f"brusselator:pca-controllability:d{latent_dim}"):
            actions = candidate_actions(basis, latent_dim, int(c["candidates"]), 200_000 + latent_dim * 1_000 + index, prediction.device)
            candidates, _, correction_norm = immediate_candidates(basis, prediction[index], target[index], actions, latent_dim, float(c["action_cost"]))
            reference_error = float(relative_l2(prediction[index : index + 1], target[index : index + 1])[0])
            for candidate_index in range(1, len(actions)):
                rows.append({
                    "LatentDim": latent_dim, "Case": int(case_ids[index]), "SolverStep": 0,
                    "LatentDistance": float(torch.linalg.vector_norm(actions[candidate_index] - actions[0])),
                    "CorrectionDifference": float(rms((candidates[candidate_index] - candidates[0]).unsqueeze(0))[0]),
                    "NextErrorDifference": float(relative_l2(candidates[candidate_index : candidate_index + 1], target[index : index + 1])[0]) - reference_error,
                    "CorrectionNorm": float(correction_norm[candidate_index]),
                    "RelativeStateChange": float(rms((candidates[candidate_index] - prediction[index]).unsqueeze(0))[0] / rms(prediction[index : index + 1])[0].clamp_min(1e-8)),
                })
    return pd.DataFrame(rows)


def one_action_oracle(basis: PCACorrectionBasis, prediction: torch.Tensor, target: torch.Tensor, case_ids: torch.Tensor, config: dict) -> pd.DataFrame:
    c = config["brusselator"]
    rows = []
    for latent_dim in PCA_DIMS:
        for index in tqdm(range(len(prediction)), desc=f"brusselator:pca-one-action:d{latent_dim}"):
            actions = candidate_actions(basis, latent_dim, int(c["candidates"]), 300_000 + latent_dim * 1_000 + index, prediction.device)
            for horizon in c["horizons"]:
                immediate, long_return, final_error, _ = rollout_candidates(
                    basis, prediction[index], prediction[index], target[index], actions, latent_dim, int(horizon), "FixedContraction", float(c["correction_retention"]), float(c["action_cost"])
                )
                greedy_index, long_index = int(immediate.argmax()), int(long_return.argmax())
                greedy_error, long_error = float(final_error[greedy_index]), float(final_error[long_index])
                rows.append({
                    "LatentDim": latent_dim, "Case": int(case_ids[index]), "Horizon": int(horizon),
                    "GreedyFinalError": greedy_error, "LongFinalError": long_error,
                    "AbsoluteOracleGain": greedy_error - long_error,
                    "RelativeOracleGain": (greedy_error - long_error) / max(greedy_error, 1e-8),
                    "ActionsDifferent": bool(not torch.allclose(actions[greedy_index], actions[long_index])),
                })
    detail = pd.DataFrame(rows)
    return detail.groupby(["LatentDim", "Horizon"], as_index=False).agg(
        MeanRelativeOracleGain=("RelativeOracleGain", "mean"), MedianRelativeOracleGain=("RelativeOracleGain", "median"),
        PositiveGainRate=("AbsoluteOracleGain", lambda values: float((values > 0.0).mean())),
        ActionsDifferentRate=("ActionsDifferent", "mean"), Samples=("Case", "count"),
    )


def stage_name(step: int) -> str:
    return "Early" if step <= 5 else "Middle" if step <= 12 else "Late"


def sequential_oracle(basis: PCACorrectionBasis, prediction: torch.Tensor, target: torch.Tensor, case_ids: torch.Tensor, config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    c = config["brusselator"]
    rows = []
    for latent_dim in PCA_DIMS:
        for continuation in CONTINUATIONS:
            for horizon in (10, 20):
                for index in tqdm(range(len(prediction)), desc=f"brusselator:pca-sequential:d{latent_dim}:{continuation}:K{horizon}"):
                    greedy_state, long_state = prediction[index].clone(), prediction[index].clone()
                    case_rows = []
                    for step in range(horizon):
                        actions = candidate_actions(basis, latent_dim, int(c["candidates"]), 400_000 + latent_dim * 100_000 + horizon * 10_000 + index * 100 + step, prediction.device)
                        greedy_next, greedy_immediate, _, _ = rollout_candidates(
                            basis, greedy_state, prediction[index], target[index], actions, latent_dim, 1, continuation, float(c["correction_retention"]), float(c["action_cost"])
                        )
                        long_immediate, long_return, _, _ = rollout_candidates(
                            basis, long_state, prediction[index], target[index], actions, latent_dim, horizon - step, continuation, float(c["correction_retention"]), float(c["action_cost"])
                        )
                        greedy_index, long_index = int(greedy_immediate.argmax()), int(long_return.argmax())
                        # Recompute only the selected real action; no fixed-contraction shortcut is applied to actual trajectories.
                        greedy_state = basis.advance(greedy_state, actions[greedy_index : greedy_index + 1], latent_dim)[0].squeeze(0)
                        long_state = basis.advance(long_state, actions[long_index : long_index + 1], latent_dim)[0].squeeze(0)
                        case_rows.append({
                            "LatentDim": latent_dim, "Continuation": continuation, "Horizon": horizon,
                            "Case": int(case_ids[index]), "SolverStep": step, "Stage": stage_name(step),
                            "ActionsDifferent": bool(not torch.allclose(actions[int(long_immediate.argmax())], actions[long_index])),
                            "ImmediateGain": float(long_immediate[long_index] - long_immediate.max()),
                        })
                    greedy_final = float(relative_l2(greedy_state.unsqueeze(0), target[index : index + 1])[0])
                    long_final = float(relative_l2(long_state.unsqueeze(0), target[index : index + 1])[0])
                    for row in case_rows:
                        row["GreedyFinalError"] = greedy_final
                        row["LongFinalError"] = long_final
                        row["TrajectoryGain"] = greedy_final - long_final
                        rows.append(row)
    detail = pd.DataFrame(rows)
    cases = detail.groupby(["LatentDim", "Continuation", "Horizon", "Case"], as_index=False).agg(
        GreedyFinalError=("GreedyFinalError", "first"), LongFinalError=("LongFinalError", "first")
    )
    cases["RelativeGain"] = (cases["GreedyFinalError"] - cases["LongFinalError"]) / cases["GreedyFinalError"].clip(lower=1e-8)
    summary = cases.groupby(["LatentDim", "Continuation", "Horizon"], as_index=False).agg(
        Cases=("Case", "count"), GreedyFinalErrorMean=("GreedyFinalError", "mean"), LongFinalErrorMean=("LongFinalError", "mean"),
        MedianCaseRelativeGain=("RelativeGain", "median"), PositiveCaseRate=("RelativeGain", lambda values: float((values > 0.0).mean())),
    )
    action_rate = detail.groupby(["LatentDim", "Continuation", "Horizon"], as_index=False).agg(ActionsDifferentRate=("ActionsDifferent", "mean"))
    summary = summary.merge(action_rate, on=["LatentDim", "Continuation", "Horizon"])
    summary["AbsoluteSequentialGain"] = summary["GreedyFinalErrorMean"] - summary["LongFinalErrorMean"]
    summary["RelativeSequentialGain"] = summary["AbsoluteSequentialGain"] / summary["GreedyFinalErrorMean"].clip(lower=1e-8)
    stage = detail[detail["Horizon"].eq(20)].groupby(["LatentDim", "Continuation", "Stage"], as_index=False).agg(
        ActionsDifferentRate=("ActionsDifferent", "mean"), MeanImmediateGain=("ImmediateGain", "mean"), MeanTrajectoryGain=("TrajectoryGain", "mean"),
    )
    stage["Stage"] = pd.Categorical(stage["Stage"], categories=["Early", "Middle", "Late"], ordered=True)
    return summary, stage.sort_values(["LatentDim", "Continuation", "Stage"]).reset_index(drop=True)


def action_space_summary(quality: pd.DataFrame, sequential: pd.DataFrame) -> pd.DataFrame:
    old = pd.read_csv(ROOT / "results" / "solver_v2" / "brusselator" / "sequential_oracle_headroom.csv")
    old_k20 = old[old["Horizon"].eq(20)].iloc[0]
    rows = [{
        "ActionSpace": "LowFrequency16", "LatentDim": 16, "ExplainedVariance": float("nan"), "Continuation": "FixedContraction",
        "K20GreedyError": old_k20["GreedyFinalErrorMean"], "K20LongError": old_k20["LongFinalErrorMean"],
        "RelativeSequentialGain": old_k20["RelativeSequentialGain"], "PositiveCaseRate": old_k20["PositiveCaseRate"], "ActionsDifferentRate": old_k20["ActionsDifferentRate"],
    }]
    explained = quality.set_index("LatentDim")["ExplainedVarianceRatio"].to_dict()
    for _, row in sequential[sequential["Horizon"].eq(20)].iterrows():
        rows.append({
            "ActionSpace": f"PCA{int(row['LatentDim'])}", "LatentDim": int(row["LatentDim"]),
            "ExplainedVariance": explained[int(row["LatentDim"])], "Continuation": row["Continuation"],
            "K20GreedyError": row["GreedyFinalErrorMean"], "K20LongError": row["LongFinalErrorMean"],
            "RelativeSequentialGain": row["RelativeSequentialGain"], "PositiveCaseRate": row["PositiveCaseRate"], "ActionsDifferentRate": row["ActionsDifferentRate"],
        })
    return pd.DataFrame(rows)


def update_docs(quality: pd.DataFrame, oracle: pd.DataFrame, sequential: pd.DataFrame, stage: pd.DataFrame, summary: pd.DataFrame) -> None:
    old_path = ROOT / "docs" / "solver_v2_brusselator_results.md"
    existing = old_path.read_text(encoding="utf-8")
    marker = "\n## Action-Space Headroom Diagnostic\n"
    existing = existing.split(marker, 1)[0]
    best = summary[summary["ActionSpace"].str.startswith("PCA")].sort_values("RelativeSequentialGain", ascending=False).iloc[0]
    pca64 = summary[(summary["ActionSpace"].eq("PCA64"))]
    pca64_best = float(pca64["RelativeSequentialGain"].max())
    strong = float(best["RelativeSequentialGain"]) >= 0.05 and float(best["PositiveCaseRate"]) >= 0.70
    bottleneck = pca64_best >= 0.025
    conclusion = (
        "PCA action space exposes strong headroom; retain Brusselator for a later learned-decoder RL phase."
        if strong else ("PCA action space materially exceeds the low-frequency diagnostic, but does not meet the strong 5% criterion." if bottleneck else "PCA64 does not clear 2.5%; the low-frequency basis is not the main headroom bottleneck, so do not start RL training and move the next benchmark diagnostic to official LNO Shallow Water.")
    )
    section = [
        "## Action-Space Headroom Diagnostic",
        "",
        "This is a PCA correction oracle diagnostic, not a final RL solver. PCA uses only FNO residual corrections from the official training split, including residuals at alpha = 0, 0.25, 0.50, and 0.75 intermediate trajectories. Validation/test targets are not used to fit the PCA basis or candidate actions.",
        "",
        "### PCA Basis Quality",
        "",
        quality.to_markdown(index=False, floatfmt=".6f"),
        "",
        "### Action-Space K=20 Summary",
        "",
        summary.to_markdown(index=False, floatfmt=".6f"),
        "",
        "### One-Action Oracle",
        "",
        oracle.to_markdown(index=False, floatfmt=".6f"),
        "",
        "### Sequential Oracle",
        "",
        sequential.to_markdown(index=False, floatfmt=".6f"),
        "",
        "### K=20 Decision Stages",
        "",
        stage.to_markdown(index=False, floatfmt=".6f"),
        "",
        f"Best PCA K=20 result: {best['ActionSpace']} with {best['Continuation']}, relative sequential gain {float(best['RelativeSequentialGain']):.2%}, positive-case rate {float(best['PositiveCaseRate']):.2%}, and action-difference rate {float(best['ActionsDifferentRate']):.2%}.",
        f"LowFrequency16 remains the reference at 1.94% K=20 gain. PCA64 best gain is {pca64_best:.2%}. {conclusion}",
        "",
        "GreedyLocal uses only the current state, the shared GT-free candidate generator, and immediate oracle reward during hypothetical continuation; it does not use a future-return selector. Actual sequential trajectories always replan at each solver step and apply bounded incremental PCA corrections.",
    ]
    old_path.write_text(existing.rstrip() + "\n\n" + "\n".join(section) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="PCA action-space headroom diagnostic for official 3D_Brusselator.")
    parser.add_argument("--stage", choices=["all", "fit", "oracle"], default="all")
    args = parser.parse_args()
    config = load_config()
    c = config["brusselator"]
    device = get_device(str(config.get("device", "cuda")))
    out = output_dir()
    basis, quality = fit_pca_from_train(config, device)
    quality.to_csv(out / "pca_basis_quality.csv", index=False)
    if args.stage == "fit":
        print(quality.to_string(index=False))
        return
    model, stats = train_fno(config, device)
    val = OfficialBrusselatorDataset(ROOT / str(c["official_npz"]), "val", max_cases=int(c["validation_cases"]), split_seed=int(c["split_seed"]))
    prediction, target, case_ids = fno_predictions(model, val, stats, device)
    prediction, target, case_ids = prediction.to(device), target.to(device), case_ids.to(device)
    control = controllability(basis, prediction, target, case_ids, config)
    oracle = one_action_oracle(basis, prediction, target, case_ids, config)
    sequential, stage = sequential_oracle(basis, prediction, target, case_ids, config)
    summary = action_space_summary(quality, sequential)
    control.to_csv(out / "pca_action_controllability.csv", index=False)
    oracle.to_csv(out / "pca_oracle_headroom_by_horizon.csv", index=False)
    sequential.to_csv(out / "pca_sequential_oracle_headroom.csv", index=False)
    stage.to_csv(out / "pca_sequential_oracle_by_stage.csv", index=False)
    summary.to_csv(out / "action_space_headroom_summary.csv", index=False)
    update_docs(quality, oracle, sequential, stage, summary)
    (out / "pca_action_space_summary.json").write_text(json.dumps({"device": str(device), "pca_cache": str(checkpoint_path()), "max_step_rms": basis.max_step_rms}, indent=2), encoding="utf-8")
    print("Brusselator PCA action-space oracle diagnostic completed.")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
