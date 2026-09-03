from __future__ import annotations

from pathlib import Path

import torch

from src.data.official_reaction_diffusion import ReactionDiffusionDataset
from src.solver_v2.pde.reaction_diffusion import ReactionDiffusionPDE


def _stack_train(npz_path: Path, limit: int) -> tuple[torch.Tensor, torch.Tensor]:
    ds = ReactionDiffusionDataset(npz_path, "train", limit)
    sources, targets = [], []
    for i in range(len(ds)):
        item = ds[i]
        sources.append(item["source"])
        targets.append(item["gt"])
    return torch.stack(sources), torch.stack(targets)


def fit_train_stats(cfg: dict, npz_path: Path, pde: ReactionDiffusionPDE, device: torch.device, out_dir: Path) -> dict:
    sources, targets = _stack_train(npz_path, int(cfg["solver_v2"]["train_cases"]))
    sources = sources.to(device)
    targets = targets.to(device)
    residuals = []
    for i in range(targets.shape[0]):
        ds = ReactionDiffusionDataset(npz_path, "train", int(cfg["solver_v2"]["train_cases"]))
        case = pde.make_case(ds[i], device)
        residuals.append(pde.residual(targets[i], case).detach())
    residual = torch.stack(residuals)
    temporal_delta = torch.diff(targets, dim=2).abs().reshape(-1)
    q = float(cfg["solver_v2"].get("step_bound_quantile", 0.99))
    multiplier = float(cfg["solver_v2"].get("step_bound_multiplier", 2.0))
    lo = float(cfg["solver_v2"].get("step_bound_min", 0.01))
    hi = float(cfg["solver_v2"].get("step_bound_max", 0.05))
    step_bound = (multiplier * torch.quantile(temporal_delta, q)).clamp(lo, hi)
    stats = {
        "u_mean": targets.mean().detach().cpu(),
        "u_std": targets.std().clamp_min(1e-6).detach().cpu(),
        "source_mean": sources.mean().detach().cpu(),
        "source_std": sources.std().clamp_min(1e-6).detach().cpu(),
        "residual_rms": torch.sqrt(torch.mean(residual ** 2)).clamp_min(1e-6).detach().cpu(),
        "step_bound": step_bound.detach().cpu(),
        "reward_scale": torch.tensor(1.0),
        "x": ReactionDiffusionDataset(npz_path, "train", 1)[0]["x"].detach().cpu(),
        "t": ReactionDiffusionDataset(npz_path, "train", 1)[0]["t"].detach().cpu(),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"u_mean": stats["u_mean"], "u_std": stats["u_std"], "source_mean": stats["source_mean"], "source_std": stats["source_std"]}, out_dir / "field_stats.pt")
    torch.save({"residual_rms": stats["residual_rms"], "step_bound": stats["step_bound"]}, out_dir / "residual_stats.pt")
    torch.save({"reward_scale": stats["reward_scale"]}, out_dir / "reward_stats.pt")
    return stats


def normalize_u(u: torch.Tensor, stats: dict) -> torch.Tensor:
    return (u - stats["u_mean"].to(u.device)) / stats["u_std"].to(u.device).clamp_min(1e-6)


def normalize_source(f: torch.Tensor, stats: dict) -> torch.Tensor:
    return (f - stats["source_mean"].to(f.device)) / stats["source_std"].to(f.device).clamp_min(1e-6)


def normalize_residual(r: torch.Tensor, stats: dict) -> torch.Tensor:
    return r / stats["residual_rms"].to(r.device).clamp_min(1e-6)


def bounded_delta(raw_delta: torch.Tensor, stats: dict, temperature: float = 1.0) -> torch.Tensor:
    return stats["step_bound"].to(raw_delta.device) * torch.tanh(raw_delta / max(float(temperature), 1e-6))
