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
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.official_brusselator import OfficialBrusselatorDataset, expected_data_path
from src.solver_v2.models.brusselator_fno import BrusselatorFNO
from src.utils.seed import get_device, set_seed


def load_config() -> dict:
    with open(ROOT / "configs" / "default.yaml", "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def relative_l2(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.linalg.vector_norm((prediction - target).flatten(1), dim=1) / torch.linalg.vector_norm(target.flatten(1), dim=1).clamp_min(1e-8)


def result_dir() -> Path:
    path = ROOT / "results" / "solver_v2" / "brusselator"
    path.mkdir(parents=True, exist_ok=True)
    (ROOT / "checkpoints" / "solver_v2" / "brusselator").mkdir(parents=True, exist_ok=True)
    return path


def normalization(train: OfficialBrusselatorDataset) -> dict[str, torch.Tensor]:
    return {
        "forcing_mean": train.forcing.mean(),
        "forcing_std": train.forcing.std().clamp_min(1e-6),
        "target_mean": train.gt.mean(),
        "target_std": train.gt.std().clamp_min(1e-6),
        "target_min": train.gt.amin(),
        "target_max": train.gt.amax(),
    }


def normalized_forcing(forcing: torch.Tensor, stats: dict[str, torch.Tensor], device: torch.device) -> torch.Tensor:
    return (forcing.to(device) - stats["forcing_mean"].to(device)) / stats["forcing_std"].to(device)


def denormalize_target(output: torch.Tensor, stats: dict[str, torch.Tensor]) -> torch.Tensor:
    return output * stats["target_std"].to(output.device) + stats["target_mean"].to(output.device)


def make_model(config: dict, device: torch.device) -> BrusselatorFNO:
    c = config["brusselator"]
    return BrusselatorFNO(
        width=int(c["fno_width"]),
        modes_t=int(c["fno_modes_t"]),
        modes_x=int(c["fno_modes_x"]),
        modes_y=int(c["fno_modes_y"]),
        depth=int(c["fno_depth"]),
    ).to(device)


def train_fno(config: dict, device: torch.device) -> tuple[BrusselatorFNO, dict[str, torch.Tensor]]:
    c = config["brusselator"]
    seed = int(c["fno_seed"])
    checkpoint = ROOT / "checkpoints" / "solver_v2" / "brusselator" / f"fno_baseline_v2_seed{seed}.pt"
    data_path = ROOT / str(c["official_npz"])
    train = OfficialBrusselatorDataset(data_path, "train", split_seed=int(c["split_seed"]))
    stats = normalization(train)
    set_seed(seed)
    model = make_model(config, device)
    if checkpoint.exists():
        payload = torch.load(checkpoint, map_location=device)
        model.load_state_dict(payload["model"])
        return model.eval(), {name: value.float() for name, value in payload["stats"].items()}

    loader = DataLoader(train, batch_size=int(c["fno_batch_size"]), shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(c["fno_lr"]), weight_decay=1e-5)
    for _ in tqdm(range(int(c["fno_epochs"])), desc="brusselator:fno-baseline"):
        model.train()
        for batch in loader:
            force = normalized_forcing(batch["forcing"], stats, device)
            target = (batch["gt"].to(device) - stats["target_mean"].to(device)) / stats["target_std"].to(device)
            prediction = model(force)
            loss = F.mse_loss(prediction, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
    torch.save({"model": model.state_dict(), "stats": {name: value.cpu() for name, value in stats.items()}}, checkpoint)
    return model.eval(), stats


@torch.no_grad()
def fno_predictions(model: BrusselatorFNO, dataset: OfficialBrusselatorDataset, stats: dict[str, torch.Tensor], device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    loader = DataLoader(dataset, batch_size=16, shuffle=False)
    predictions, targets, identifiers = [], [], []
    for batch in loader:
        output = denormalize_target(model(normalized_forcing(batch["forcing"], stats, device)), stats)
        predictions.append(output.cpu())
        targets.append(batch["gt"].float())
        identifiers.append(batch["case_id"].long())
    return torch.cat(predictions), torch.cat(targets), torch.cat(identifiers)


class LowFrequencyCorrectionBasis:
    """Fixed bounded low-frequency space-time basis; it has no learned encoder or decoder."""

    def __init__(self, latent_dim: int, scale: float, retention: float, lower: float, upper: float, device: torch.device):
        if latent_dim != 16:
            raise ValueError("The fixed Brusselator basis is defined for latent_dim=16.")
        t = torch.arange(39, device=device, dtype=torch.float32) / 38.0
        x = torch.arange(14, device=device, dtype=torch.float32) / 13.0
        y = torch.arange(14, device=device, dtype=torch.float32) / 13.0
        tt, xx, yy = torch.meshgrid(t, x, y, indexing="ij")
        fields = []
        for ft in range(3):
            for fx in range(3):
                for fy in range(3):
                    if ft == fx == fy == 0:
                        continue
                    field = torch.cos(torch.pi * (ft * tt + fx * xx + fy * yy))
                    fields.append(field / torch.sqrt(torch.mean(field.square())).clamp_min(1e-6))
                    if len(fields) == latent_dim:
                        break
                if len(fields) == latent_dim:
                    break
            if len(fields) == latent_dim:
                break
        if len(fields) != latent_dim:
            raise RuntimeError("Insufficient fixed spectral modes for requested latent dimension.")
        self.fields = torch.stack(fields)
        self.scale = float(scale)
        self.retention = float(retention)
        self.lower, self.upper = float(lower), float(upper)

    def correction(self, actions: torch.Tensor) -> torch.Tensor:
        return self.scale * torch.einsum("bd,dTXY->bTXY", actions, self.fields)

    def step(self, current: torch.Tensor, baseline: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        if current.ndim == 3:
            current = current.unsqueeze(0)
        if baseline.ndim == 3:
            baseline = baseline.unsqueeze(0)
        next_u = baseline + self.retention * (current - baseline) + self.correction(actions)
        return next_u.clamp(self.lower, self.upper)


def candidate_actions(latent_dim: int, count: int, seed: int, device: torch.device) -> torch.Tensor:
    if count < 16:
        raise ValueError("At least 16 shared candidate actions are required.")
    generator = torch.Generator(device=device.type).manual_seed(seed)
    actions = [torch.zeros(latent_dim, device=device)]
    radii = (0.25, 0.5, 0.75, 1.0)
    while len(actions) < count:
        direction = torch.randn(latent_dim, generator=generator, device=device)
        direction = direction / torch.sqrt(torch.mean(direction.square())).clamp_min(1e-6)
        radius = radii[(len(actions) - 1) % len(radii)]
        actions.extend([radius * direction, -radius * direction])
    return torch.stack(actions[:count])


def reward(before: torch.Tensor, after: torch.Tensor, actions: torch.Tensor, target: torch.Tensor, action_cost: float) -> torch.Tensor:
    e0 = relative_l2(before, target.expand(before.shape[0], -1, -1, -1))
    e1 = relative_l2(after, target.expand(after.shape[0], -1, -1, -1))
    return torch.log((e0 + 1e-8) / (e1 + 1e-8)) - action_cost * torch.mean(actions.square(), dim=1)


@torch.no_grad()
def rollout_candidates(basis: LowFrequencyCorrectionBasis, current: torch.Tensor, baseline: torch.Tensor, target: torch.Tensor, actions: torch.Tensor, horizon: int, action_cost: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    repeated_current = current.unsqueeze(0).expand(actions.shape[0], -1, -1, -1)
    repeated_baseline = baseline.unsqueeze(0).expand_as(repeated_current)
    first = basis.step(repeated_current, repeated_baseline, actions)
    immediate = reward(repeated_current, first, actions, target, action_cost)
    total = immediate.clone()
    zero = torch.zeros_like(actions)
    iterate = first
    for _ in range(1, horizon):
        next_u = basis.step(iterate, repeated_baseline, zero)
        total += reward(iterate, next_u, zero, target, action_cost)
        iterate = next_u
    return immediate, total, relative_l2(iterate, target.expand_as(iterate))


def controllability_table(basis: LowFrequencyCorrectionBasis, baseline: torch.Tensor, target: torch.Tensor, case_ids: torch.Tensor, config: dict) -> pd.DataFrame:
    c = config["brusselator"]
    rows: list[dict[str, float | int]] = []
    for index in tqdm(range(len(baseline)), desc="brusselator:action-controllability"):
        action = candidate_actions(int(c["latent_dim"]), int(c["candidates"]), 9_000 + index, baseline.device)
        candidate = basis.step(baseline[index], baseline[index], action)
        ref_error = float(relative_l2(baseline[index : index + 1], target[index : index + 1])[0])
        ref = candidate[0]
        for candidate_index in range(1, len(action)):
            rows.append({
                "Case": int(case_ids[index]), "SolverStep": 0,
                "LatentDistance": float(torch.linalg.vector_norm(action[candidate_index] - action[0])),
                "CorrectionDifference": float(torch.sqrt(torch.mean((candidate[candidate_index] - ref).square()))),
                "NextErrorDifference": float(relative_l2(candidate[candidate_index : candidate_index + 1], target[index : index + 1])[0]) - ref_error,
            })
    return pd.DataFrame(rows)


def one_action_headroom(basis: LowFrequencyCorrectionBasis, baseline: torch.Tensor, target: torch.Tensor, case_ids: torch.Tensor, config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    c = config["brusselator"]
    detail: list[dict[str, float | int | bool]] = []
    for index in tqdm(range(len(baseline)), desc="brusselator:one-action-oracle"):
        actions = candidate_actions(int(c["latent_dim"]), int(c["candidates"]), 12_000 + index, baseline.device)
        for horizon in c["horizons"]:
            immediate, long_return, final_error = rollout_candidates(basis, baseline[index], baseline[index], target[index], actions, int(horizon), float(c["action_cost"]))
            greedy_index, long_index = int(immediate.argmax()), int(long_return.argmax())
            greedy_error, long_error = float(final_error[greedy_index]), float(final_error[long_index])
            detail.append({
                "Case": int(case_ids[index]), "Horizon": int(horizon), "GreedyFinalError": greedy_error,
                "LongFinalError": long_error, "AbsoluteOracleGain": greedy_error - long_error,
                "RelativeOracleGain": (greedy_error - long_error) / max(greedy_error, 1e-8),
                "ActionsDifferent": bool(not torch.allclose(actions[greedy_index], actions[long_index])),
            })
    detail_frame = pd.DataFrame(detail)
    summary = detail_frame.groupby("Horizon", as_index=False).agg(
        MeanRelativeOracleGain=("RelativeOracleGain", "mean"),
        MedianRelativeOracleGain=("RelativeOracleGain", "median"),
        PositiveGainRate=("AbsoluteOracleGain", lambda value: float((value > 0.0).mean())),
        ActionsDifferentRate=("ActionsDifferent", "mean"), Samples=("Case", "count"),
    )
    return detail_frame, summary


def stage_name(step: int) -> str:
    return "Early" if step <= 5 else "Middle" if step <= 12 else "Late"


def choose_action(basis: LowFrequencyCorrectionBasis, current: torch.Tensor, baseline: torch.Tensor, target: torch.Tensor, horizon: int, step: int, actions: torch.Tensor, action_cost: float) -> dict[str, torch.Tensor | float | bool]:
    immediate, long_return, _ = rollout_candidates(basis, current, baseline, target, actions, horizon - step, action_cost)
    greedy_index, long_index = int(immediate.argmax()), int(long_return.argmax())
    return {
        "greedy_action": actions[greedy_index], "long_action": actions[long_index],
        "long_immediate_gain": float(immediate[long_index] - immediate[greedy_index]),
        "different": bool(not torch.allclose(actions[greedy_index], actions[long_index])),
    }


def sequential_oracle(basis: LowFrequencyCorrectionBasis, baseline: torch.Tensor, target: torch.Tensor, case_ids: torch.Tensor, config: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    c = config["brusselator"]
    all_rows: list[dict[str, float | int | bool | str]] = []
    for horizon in c["horizons"]:
        for index in tqdm(range(len(baseline)), desc=f"brusselator:sequential-oracle:K{horizon}"):
            greedy_u, long_u = baseline[index].clone(), baseline[index].clone()
            case_rows = []
            for step in range(int(horizon)):
                actions = candidate_actions(int(c["latent_dim"]), int(c["candidates"]), 100_000 + int(horizon) * 10_000 + index * 100 + step, baseline.device)
                greedy_choice = choose_action(basis, greedy_u, baseline[index], target[index], int(horizon), step, actions, float(c["action_cost"]))
                long_choice = choose_action(basis, long_u, baseline[index], target[index], int(horizon), step, actions, float(c["action_cost"]))
                greedy_u = basis.step(greedy_u, baseline[index], greedy_choice["greedy_action"].unsqueeze(0)).squeeze(0)
                long_u = basis.step(long_u, baseline[index], long_choice["long_action"].unsqueeze(0)).squeeze(0)
                case_rows.append({
                    "Horizon": int(horizon), "Case": int(case_ids[index]), "SolverStep": step, "Stage": stage_name(step),
                    "ActionsDifferent": bool(long_choice["different"]), "ImmediateGain": float(long_choice["long_immediate_gain"]),
                })
            greedy_final = float(relative_l2(greedy_u.unsqueeze(0), target[index : index + 1])[0])
            long_final = float(relative_l2(long_u.unsqueeze(0), target[index : index + 1])[0])
            for row in case_rows:
                row["GreedyFinalError"] = greedy_final
                row["LongFinalError"] = long_final
                row["TrajectoryGain"] = greedy_final - long_final
                all_rows.append(row)
    detail = pd.DataFrame(all_rows)
    per_case = detail.groupby(["Horizon", "Case"], as_index=False).agg(GreedyFinalError=("GreedyFinalError", "first"), LongFinalError=("LongFinalError", "first"))
    per_case["RelativeGain"] = (per_case["GreedyFinalError"] - per_case["LongFinalError"]) / per_case["GreedyFinalError"].clip(lower=1e-8)
    summary = per_case.groupby("Horizon", as_index=False).agg(
        Cases=("Case", "count"), GreedyFinalErrorMean=("GreedyFinalError", "mean"), LongFinalErrorMean=("LongFinalError", "mean"),
        MedianCaseRelativeGain=("RelativeGain", "median"), PositiveCaseRate=("RelativeGain", lambda value: float((value > 0.0).mean())),
    )
    action_rate = detail.groupby("Horizon", as_index=False).agg(ActionsDifferentRate=("ActionsDifferent", "mean"))
    summary = summary.merge(action_rate, on="Horizon")
    summary["AbsoluteSequentialGain"] = summary["GreedyFinalErrorMean"] - summary["LongFinalErrorMean"]
    summary["RelativeSequentialGain"] = summary["AbsoluteSequentialGain"] / summary["GreedyFinalErrorMean"].clip(lower=1e-8)
    stage = detail[detail["Horizon"].eq(20)].groupby("Stage", as_index=False).agg(
        ActionsDifferentRate=("ActionsDifferent", "mean"), MeanImmediateGain=("ImmediateGain", "mean"), MeanTrajectoryGain=("TrajectoryGain", "mean"),
    )
    stage["Stage"] = pd.Categorical(stage["Stage"], categories=["Early", "Middle", "Late"], ordered=True)
    stage = stage.sort_values("Stage").reset_index(drop=True)
    return detail, summary, stage


def write_docs(baseline: pd.DataFrame, controllability: pd.DataFrame, headroom: pd.DataFrame, sequential: pd.DataFrame, stage: pd.DataFrame) -> None:
    b = baseline.set_index("Split")
    k20 = sequential[sequential["Horizon"].eq(20)].iloc[0]
    strong = float(k20["RelativeSequentialGain"]) >= 0.05 and float(k20["PositiveCaseRate"]) >= 0.65
    moderate = float(k20["RelativeSequentialGain"]) >= 0.02
    conclusion = "strong headroom; a later RL phase is justified" if strong else ("moderate headroom; any later RL phase requires separate validation" if moderate else "weak headroom; do not start full RL training")
    lines = [
        "# Solver V2: Official 3D Brusselator Diagnostic",
        "",
        "This diagnostic preserves the Reaction-Diffusion results and adds the official LNO 3D_Brusselator NPZ only. The baseline is a newly trained FNO baseline, not LNO; the official LNO implementation was not adapted because its script is globally coupled and CUDA-hardwired.",
        "",
        f"- Official preprocessing: `(case, time, x, y) = (*, 39, 28, 28)`, then official `r=2` subsampling to `(*, 39, 14, 14)`.",
        f"- FNO baseline validation Relative L2: {b.loc['validation', 'Mean Relative L2']:.6f}; held-out test Relative L2: {b.loc['test', 'Mean Relative L2']:.6f}.",
        f"- Action controllability: median correction difference {controllability['CorrectionDifference'].median():.6g}; median next-error difference {controllability['NextErrorDifference'].median():.6g}.",
        "- Rewards use ground-truth Relative L2 reduction plus a small action cost. This is a validation oracle diagnostic, not deployment-time reward.",
        "- The solver state is the full trajectory field; solver iterations are separate from the 39 physical time indices.",
        "- The fixed low-frequency correction basis uses no autoencoder. Its reference continuation is a fixed contraction to the FNO initial trajectory, not a learned policy or a reconstructed Brusselator residual.",
        "",
        "## One-Action Headroom",
        "",
        headroom.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Sequential Headroom",
        "",
        sequential.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## K=20 Decision Stages",
        "",
        stage.to_markdown(index=False, floatfmt=".6f"),
        "",
        f"Conclusion: {conclusion}. The specified strong criterion is relative sequential gain >= 5% and positive-case rate >= 65%; observed K=20 values are {float(k20['RelativeSequentialGain']):.2%} and {float(k20['PositiveCaseRate']):.2%}.",
    ]
    (ROOT / "docs" / "solver_v2_brusselator_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Official 3D_Brusselator FNO and oracle headroom diagnostics.")
    parser.add_argument("--stage", choices=["all", "baseline", "oracle"], default="all")
    args = parser.parse_args()
    config = load_config()
    c = config["brusselator"]
    device = get_device(str(config.get("device", "cuda")))
    out = result_dir()
    data_path = ROOT / str(c["official_npz"])
    if not data_path.exists():
        data_path = expected_data_path(ROOT)
    model, stats = train_fno(config, device)
    val = OfficialBrusselatorDataset(data_path, "val", max_cases=int(c["validation_cases"]), split_seed=int(c["split_seed"]))
    test = OfficialBrusselatorDataset(data_path, "test", max_cases=int(c["test_cases"]), split_seed=int(c["split_seed"]))
    val_prediction, val_target, val_ids = fno_predictions(model, val, stats, device)
    test_prediction, test_target, _ = fno_predictions(model, test, stats, device)
    baseline = pd.DataFrame([
        {"Method": "FNO baseline (not LNO)", "Split": "validation", "Mean Relative L2": float(relative_l2(val_prediction, val_target).mean()), "Cases": len(val)},
        {"Method": "FNO baseline (not LNO)", "Split": "test", "Mean Relative L2": float(relative_l2(test_prediction, test_target).mean()), "Cases": len(test)},
    ])
    baseline.to_csv(out / "fno_baseline_results.csv", index=False)
    if args.stage == "baseline":
        print(baseline.to_string(index=False))
        return

    scale = float(c["correction_scale_fraction"]) * float(stats["target_std"])
    basis = LowFrequencyCorrectionBasis(int(c["latent_dim"]), scale, float(c["correction_retention"]), float(stats["target_min"]), float(stats["target_max"]), device)
    val_prediction, val_target, val_ids = val_prediction.to(device), val_target.to(device), val_ids.to(device)
    controllability = controllability_table(basis, val_prediction, val_target, val_ids, config)
    headroom_detail, headroom = one_action_headroom(basis, val_prediction, val_target, val_ids, config)
    sequential_detail, sequential, stage = sequential_oracle(basis, val_prediction, val_target, val_ids, config)
    controllability.to_csv(out / "action_controllability.csv", index=False)
    headroom_detail.to_csv(out / "oracle_headroom_detail.csv", index=False)
    headroom.to_csv(out / "oracle_headroom_by_horizon.csv", index=False)
    sequential_detail.to_csv(out / "sequential_oracle_detail.csv", index=False)
    sequential.to_csv(out / "sequential_oracle_headroom.csv", index=False)
    stage.to_csv(out / "sequential_oracle_by_stage.csv", index=False)
    write_docs(baseline, controllability, headroom, sequential, stage)
    (out / "summary.json").write_text(json.dumps({"device": str(device), "baseline": baseline.to_dict(orient="records"), "correction_scale": scale}, indent=2), encoding="utf-8")
    print("Official 3D_Brusselator oracle diagnostics completed.")
    print(headroom.to_string(index=False))
    print(sequential.to_string(index=False))


if __name__ == "__main__":
    main()
