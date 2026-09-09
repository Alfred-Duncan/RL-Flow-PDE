from __future__ import annotations

import argparse
import json
import sys
import time
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
V2_CACHE_VERSION = "latent_operator_v3"
INITIALIZER_CACHE_VERSION = "latent_operator_v2"
CRITIC_CACHE_VERSION = "within_state_rank_v2"
VERIFIED_PI_CACHE_VERSION = "verified_pi_v1"

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from src.data.official_reaction_diffusion import ReactionDiffusionDataset, prepare_reaction_diffusion
from src.solver_v2.data.normalization import fit_train_stats, normalize_residual, normalize_source, normalize_u
from src.solver_v2.data.replay_buffer import RawCorrectionTransition, SolverTransition
from src.solver_v2.models.correction_autoencoder import CorrectionAutoencoder, CorrectionEncoder, CorrectionOperatorDecoder, latent_stats
from src.solver_v2.models.fno_initializer import FNOInitializer, initializer_loss
from src.solver_v2.models.operator_actor import DeterministicNeuralOperatorActor
from src.solver_v2.models.operator_critic import TwinOperatorCritic
from src.solver_v2.pde.reaction_diffusion import ReactionDiffusionCase, ReactionDiffusionPDE
from src.solver_v2.rl.td3_bc import TD3BCTrainer
from src.solver_v2.rl.rollout_verified_pi import TrustedPolicy, VerifiedTarget, normalized_rms_shift, project_to_trust_region, train_verified_policy
from src.solver_v2.training.pretrain_actor import pretrain_actor
from src.utils.seed import get_device, set_seed


class RawTransitionDataset(Dataset):
    def __init__(self, rows: list[RawCorrectionTransition]):
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        tr = self.rows[idx]
        return {"state_fields": tr.state_fields, "state_scalars": tr.state_scalars, "delta_u": tr.delta_u}


def load_config() -> dict:
    with open(ROOT / "configs/default.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dirs() -> None:
    for path in ["checkpoints/solver_v2", "results/solver_v2/tables", "results/solver_v2/figures", "docs"]:
        (ROOT / path).mkdir(parents=True, exist_ok=True)


def state_from_u(pde: ReactionDiffusionPDE, case: ReactionDiffusionCase, u: torch.Tensor, step_frac: float, stats: dict) -> tuple[torch.Tensor, torch.Tensor]:
    res = pde.residual(u, case)
    ic_field = case.ic[:, None].expand_as(u)
    bc_field = torch.zeros_like(u)
    bc_field[0, :] = case.bc_left
    bc_field[-1, :] = case.bc_right
    fields = torch.stack(
        [
            normalize_u(u, stats),
            normalize_residual(res, stats),
            normalize_source(case.source, stats),
            normalize_u(ic_field, stats),
            normalize_u(bc_field, stats),
        ],
        dim=0,
    )
    scalars = torch.tensor([pde.diffusion, pde.reaction, float(step_frac), 1.0], dtype=torch.float32, device=u.device)
    return fields, scalars


def raw_case_from_transition(tr: RawCorrectionTransition, stats: dict, pde: ReactionDiffusionPDE, device: torch.device) -> tuple[ReactionDiffusionCase, torch.Tensor]:
    fields = tr.state_fields.to(device)
    gt = tr.gt.to(device)
    u = fields[0] * stats["u_std"].to(device) + stats["u_mean"].to(device)
    source = fields[2] * stats["source_std"].to(device) + stats["source_mean"].to(device)
    case = ReactionDiffusionCase(source, gt, gt[:, 0], gt[0, :], gt[-1, :], stats["x"].to(device).reshape(-1), stats["t"].to(device).reshape(-1), tr.case_id)
    return case, u


def train_initializer(cfg: dict, npz_path: Path, seed: int, pde: ReactionDiffusionPDE, device: torch.device) -> FNOInitializer:
    s = cfg["solver_v2"]
    ckpt = ROOT / "checkpoints" / "solver_v2" / f"fno_initializer_{INITIALIZER_CACHE_VERSION}_seed{seed}.pt"
    model = FNOInitializer(width=int(s["width"]), modes=int(s["modes"]), depth=3).to(device)
    if ckpt.exists():
        model.load_state_dict(torch.load(ckpt, map_location=device)["model"])
        return model.eval()
    ds = ReactionDiffusionDataset(npz_path, "train", int(s["train_cases"]))
    loader = DataLoader(ds, batch_size=int(s["batch_size"]), shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=float(s["lr"]), weight_decay=1e-4)
    for _ in tqdm(range(int(s["initializer_epochs"])), desc=f"v2:init:{seed}"):
        model.train()
        for batch in loader:
            source = batch["source"].to(device)
            gt = batch["gt"].to(device)
            pred = model(source)
            loss = initializer_loss(pred, gt)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
    torch.save({"model": model.state_dict()}, ckpt)
    return model.eval()


def initial_solution(initializer: FNOInitializer, pde: ReactionDiffusionPDE, case: ReactionDiffusionCase) -> torch.Tensor:
    with torch.no_grad():
        pred = initializer(case.source.unsqueeze(0)).squeeze(0)
    return pde.project_hard_constraints(pred, case)


def bounded_target_delta(u: torch.Tensor, target: torch.Tensor, stats: dict, remaining: int) -> torch.Tensor:
    raw = (target - u) / float(max(1, remaining))
    bound = stats["step_bound"].to(u.device)
    return raw.clamp(-bound, bound)


def make_raw_transition(
    pde: ReactionDiffusionPDE,
    case: ReactionDiffusionCase,
    u: torch.Tensor,
    delta: torch.Tensor,
    stats: dict,
    split: str,
    episode_id: int,
    step_idx: int,
    horizon: int,
    source_policy: str,
    cfg: dict,
) -> tuple[RawCorrectionTransition, torch.Tensor]:
    fields, scalars = state_from_u(pde, case, u, step_idx / max(1, horizon), stats)
    next_u = pde.step(u, delta, case)
    next_fields, next_scalars = state_from_u(pde, case, next_u, (step_idx + 1) / max(1, horizon), stats)
    r = pde.reward(u, next_u, delta, case, float(cfg["solver_v2"]["lambda_action"]))
    row = RawCorrectionTransition(
        case_id=case.case_id,
        episode_id=episode_id,
        step_idx=step_idx,
        split=split,
        gt=case.gt.detach().cpu(),
        state_fields=fields.detach().cpu(),
        state_scalars=scalars.detach().cpu(),
        delta_u=delta.detach().cpu(),
        reward=r["reward"],
        next_fields=next_fields.detach().cpu(),
        next_scalars=next_scalars.detach().cpu(),
        done=float(step_idx == horizon - 1),
        error_before=r["error_before"],
        error_after=r["error_after"],
        residual_before=r["residual_before"],
        residual_after=r["residual_after"],
        physics_before=r["physics_before"],
        physics_after=r["physics_after"],
        action_norm=r["action_norm"],
        source_policy=source_policy,
    )
    return row, next_u.detach()


def gradient_delta(pde: ReactionDiffusionPDE, case: ReactionDiffusionCase, u: torch.Tensor, stats: dict, lr: float = 0.008) -> torch.Tensor:
    var = u.detach().clone().requires_grad_(True)
    loss = pde.physics_metrics(var, case)["energy"]
    grad = torch.autograd.grad(loss, var)[0]
    delta = -lr * grad / grad.abs().mean().clamp_min(1e-4)
    return delta.clamp(-stats["step_bound"].to(u.device), stats["step_bound"].to(u.device)).detach()


def pinn_delta(pde: ReactionDiffusionPDE, case: ReactionDiffusionCase, u: torch.Tensor, stats: dict, steps: int = 3) -> torch.Tensor:
    bound = stats["step_bound"].to(u.device)
    raw = torch.zeros_like(u, requires_grad=True)
    opt = torch.optim.Adam([raw], lr=0.002)
    for _ in range(steps):
        delta = 0.5 * bound * torch.tanh(raw)
        cand = pde.step(u, delta, case)
        pm = pde.physics_metrics(cand, case)
        loss = pm["energy"] / stats["residual_rms"].to(u.device).clamp_min(1e-6) + 0.5 * delta.pow(2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return (0.5 * bound * torch.tanh(raw.detach())).detach()


def generate_positive_trajectories(cfg: dict, npz_path: Path, initializer: FNOInitializer, pde: ReactionDiffusionPDE, stats: dict, seed: int, split: str, device: torch.device) -> list[RawCorrectionTransition]:
    cache = ROOT / "checkpoints" / "solver_v2" / f"positive_traj_{V2_CACHE_VERSION}_{split}_seed{seed}.pt"
    if cache.exists():
        return torch.load(cache, weights_only=False)
    rows: list[RawCorrectionTransition] = []
    horizon = int(cfg["solver_v2"]["horizon"])
    limit = int(cfg["solver_v2"][f"{split}_cases"])
    ds = ReactionDiffusionDataset(npz_path, split, limit)
    episode = 100000 if split == "val" else 0
    for i in tqdm(range(len(ds)), desc=f"v2:positive_traj:{split}:{seed}"):
        case = pde.make_case(ds[i], device)
        case.case_id = i if split == "train" else 10000 + i
        for policy_name in ["gt_directed", "gradient"]:
            u = initial_solution(initializer, pde, case)
            for k in range(horizon):
                delta = bounded_target_delta(u, case.gt, stats, horizon - k) if policy_name == "gt_directed" else gradient_delta(pde, case, u, stats)
                row, next_u = make_raw_transition(pde, case, u, delta, stats, split, episode, k, horizon, policy_name, cfg)
                if row.error_after < row.error_before:
                    rows.append(row)
                    u = next_u
                elif policy_name != "gt_directed":
                    break
            episode += 1
    torch.save(rows, cache)
    return rows


def train_correction_autoencoder(cfg: dict, rows: list[RawCorrectionTransition], stats: dict, seed: int, device: torch.device) -> tuple[CorrectionEncoder, CorrectionOperatorDecoder, dict[str, torch.Tensor]]:
    s = cfg["solver_v2"]
    latent_dim = int(s["latent_dim"])
    encoder = CorrectionEncoder(latent_dim=latent_dim, width=int(s["width"]), modes=int(s["modes"]), depth=2).to(device)
    decoder = CorrectionOperatorDecoder(latent_dim=latent_dim, width=int(s["width"]), modes=int(s["modes"]), depth=int(s["actor_depth"])).to(device)
    ckpt = ROOT / "checkpoints" / "solver_v2" / f"correction_autoencoder_{V2_CACHE_VERSION}_seed{seed}.pt"
    if ckpt.exists():
        payload = torch.load(ckpt, map_location=device)
        encoder.load_state_dict(payload["encoder"])
        decoder.load_state_dict(payload["decoder"])
        zmeta = payload.get("latent_stats", {})
        stats.update(zmeta)
        for p in decoder.parameters():
            p.requires_grad_(False)
        return encoder.eval(), decoder.eval(), zmeta
    ds = RawTransitionDataset([r for r in rows if r.split == "train"])
    loader = DataLoader(ds, batch_size=int(s["batch_size"]), shuffle=True)
    ae = CorrectionAutoencoder(encoder, decoder).to(device)
    opt = torch.optim.AdamW(ae.parameters(), lr=float(s["lr"]), weight_decay=1e-4)
    for _ in tqdm(range(int(s.get("autoencoder_epochs", 70))), desc=f"v2:correction_ae:{seed}"):
        ae.train()
        for batch in loader:
            fields = batch["state_fields"].to(device)
            scalars = batch["state_scalars"].to(device)
            delta = batch["delta_u"].to(device)
            recon, z = ae(fields, scalars, delta, stats, float(s["temperature"]))
            perm = torch.randperm(z.shape[0], device=device)
            z_neg = z[perm]
            z_pert = z + torch.randn_like(z) * z.detach().std(dim=0, keepdim=True).clamp_min(0.05)
            recon_neg = decoder(fields, scalars, z_neg, stats, float(s["temperature"]))
            recon_pert = decoder(fields, scalars, z_pert, stats, float(s["temperature"]))
            pos_err = torch.mean((recon - delta) ** 2, dim=(1, 2))
            wrong_err = torch.mean((recon_neg - delta) ** 2, dim=(1, 2))
            sens = torch.mean((recon_pert - recon.detach()) ** 2, dim=(1, 2))
            margin = float(s.get("decoder_wrong_margin", 0.05)) * float(stats["step_bound"]) ** 2
            sens_floor = float(s.get("decoder_sensitivity_floor", 0.03)) * float(stats["step_bound"]) ** 2
            wrong_loss = torch.relu(pos_err.detach() + margin - wrong_err).mean()
            sens_loss = torch.relu(sens_floor - sens).mean()
            loss = pos_err.mean() + float(s.get("lambda_decoder_wrong", 0.5)) * wrong_loss + float(s.get("lambda_decoder_sens", 0.2)) * sens_loss + 1e-4 * z.pow(2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ae.parameters(), 1.0)
            opt.step()
    zs = []
    encoder.eval()
    with torch.no_grad():
        for batch in DataLoader(ds, batch_size=int(s["batch_size"])):
            zs.append(encoder(batch["state_fields"].to(device), batch["state_scalars"].to(device), batch["delta_u"].to(device)).cpu())
    zmeta = latent_stats(torch.cat(zs, dim=0))
    stats.update(zmeta)
    for p in decoder.parameters():
        p.requires_grad_(False)
    torch.save({"encoder": encoder.state_dict(), "decoder": decoder.state_dict(), "latent_stats": zmeta}, ckpt)
    return encoder.eval(), decoder.eval(), zmeta


def compute_mc_returns(rows: list[SolverTransition]) -> list[SolverTransition]:
    by_ep: dict[int, list[SolverTransition]] = {}
    for tr in rows:
        by_ep.setdefault(tr.episode_id, []).append(tr)
    for episode_rows in by_ep.values():
        running = 0.0
        for tr in sorted(episode_rows, key=lambda x: x.step_idx, reverse=True):
            running = float(tr.reward) + running
            tr.mc_return = running
    return rows


def encode_transitions(raw_rows: list[RawCorrectionTransition], encoder: CorrectionEncoder, device: torch.device) -> list[SolverTransition]:
    rows: list[SolverTransition] = []
    encoder.eval()
    with torch.no_grad():
        for tr in raw_rows:
            z = encoder(tr.state_fields.unsqueeze(0).to(device), tr.state_scalars.unsqueeze(0).to(device), tr.delta_u.unsqueeze(0).to(device)).squeeze(0).cpu()
            rows.append(SolverTransition(**tr.__dict__, action=z, mc_return=0.0))
    return compute_mc_returns(rows)


def rollout_actor_to_replay(cfg: dict, npz_path: Path, initializer: FNOInitializer, actor: DeterministicNeuralOperatorActor, decoder: CorrectionOperatorDecoder, pde: ReactionDiffusionPDE, stats: dict, seed: int, cycle: int, device: torch.device) -> list[SolverTransition]:
    rows: list[SolverTransition] = []
    horizon = int(cfg["solver_v2"]["horizon"])
    ds = ReactionDiffusionDataset(npz_path, "train", int(cfg["solver_v2"]["online_rollout_cases"]))
    latent_std = stats.get("latent_std", torch.ones(int(cfg["solver_v2"]["latent_dim"]))).to(device).view(1, -1)
    latent_mean = stats.get("latent_mean", torch.zeros(int(cfg["solver_v2"]["latent_dim"]))).to(device).view(1, -1)
    actor.eval()
    decoder.eval()
    for i in tqdm(range(len(ds)), desc=f"v2:online_rollout:{seed}:{cycle}"):
        case = pde.make_case(ds[i], device)
        case.case_id = 50000 + cycle * 1000 + i
        u = initial_solution(initializer, pde, case)
        episode = 50000 + cycle * 1000 + i
        for k in range(horizon):
            fields, scalars = state_from_u(pde, case, u, k / max(1, horizon), stats)
            with torch.no_grad():
                z = actor(fields.unsqueeze(0), scalars.unsqueeze(0), stats, float(cfg["solver_v2"]["temperature"])).squeeze(0)
                z = z + 0.08 * latent_std * torch.randn_like(z)
                delta = decoder(fields.unsqueeze(0), scalars.unsqueeze(0), z.unsqueeze(0), stats, float(cfg["solver_v2"]["temperature"])).squeeze(0)
            raw, next_u = make_raw_transition(pde, case, u, delta, stats, "train", episode, k, horizon, f"online_actor_cycle{cycle}", cfg)
            rows.append(SolverTransition(**raw.__dict__, action=z.detach().cpu(), mc_return=0.0))
            u = next_u
    return compute_mc_returns(rows)


def _corr_metrics(qs: list[float], rs: list[float]) -> tuple[float, float]:
    if len(qs) < 3 or np.std(qs) < 1e-10 or np.std(rs) < 1e-10:
        return 0.0, 0.0
    q = pd.Series(qs, dtype="float64")
    r = pd.Series(rs, dtype="float64")
    return float(q.corr(r, method="pearson")), float(q.corr(r, method="spearman"))


def structured_latent_candidates(z_sup: torch.Tensor, z_actor: torch.Tensor, latent_mean: torch.Tensor, latent_std: torch.Tensor) -> list[tuple[str, torch.Tensor]]:
    z_clip_lo = latent_mean - 3.0 * latent_std
    z_clip_hi = latent_mean + 3.0 * latent_std
    candidates = [
        ("supervised", z_sup),
        ("actor", z_actor),
        ("zero", torch.zeros_like(z_sup)),
        ("random", latent_mean + latent_std * torch.randn_like(z_sup)),
    ]
    candidates.append(("shuffled", z_sup[torch.randperm(z_sup.shape[0], device=z_sup.device)] if z_sup.shape[0] > 1 else -z_sup))
    for scale in [0.25, 0.5, 1.0, 2.0]:
        candidates.append((f"perturb_{scale:g}sigma", z_sup + scale * latent_std * torch.randn_like(z_sup)))
    return [(name, z.clamp(z_clip_lo, z_clip_hi)) for name, z in candidates]


def rollout_k_step_return(
    cfg: dict,
    case: ReactionDiffusionCase,
    u: torch.Tensor,
    first_z: torch.Tensor,
    continuation_actor: DeterministicNeuralOperatorActor,
    decoder: CorrectionOperatorDecoder,
    pde: ReactionDiffusionPDE,
    stats: dict,
    device: torch.device,
    start_step: int,
) -> tuple[float, torch.Tensor, torch.Tensor, dict[str, float]]:
    horizon = int(cfg["solver_v2"]["horizon"])
    total = 0.0
    cur = u
    first_delta = torch.zeros_like(u)
    first_reward: dict[str, float] | None = None
    for k in range(start_step, horizon):
        fields, scalars = state_from_u(pde, case, cur, k / max(1, horizon), stats)
        with torch.no_grad():
            if k == start_step:
                z = first_z
            else:
                z = continuation_actor(fields.unsqueeze(0).to(device), scalars.unsqueeze(0).to(device), stats, float(cfg["solver_v2"]["temperature"]))
            delta = decoder(fields.unsqueeze(0).to(device), scalars.unsqueeze(0).to(device), z, stats, float(cfg["solver_v2"]["temperature"])).squeeze(0)
        nxt = pde.step(cur, delta, case)
        reward = pde.reward(cur, nxt, delta, case, float(cfg["solver_v2"]["lambda_action"]))
        total += reward["reward"]
        if k == start_step:
            first_delta = delta.detach()
            first_reward = reward
        cur = nxt.detach()
    assert first_reward is not None
    return total, cur, first_delta, first_reward


def batched_state_from_u(pde: ReactionDiffusionPDE, case: ReactionDiffusionCase, u: torch.Tensor, step_frac: float, stats: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """Vectorized state construction for oracle candidate rollouts, with no model changes."""
    dx = (case.x[1] - case.x[0]).abs().clamp_min(1e-6)
    dt = (case.t[1] - case.t[0]).abs().clamp_min(1e-6)
    ut = torch.zeros_like(u)
    ut[:, :, 1:-1] = (u[:, :, 2:] - u[:, :, :-2]) / (2.0 * dt)
    ut[:, :, 0] = (u[:, :, 1] - u[:, :, 0]) / dt
    ut[:, :, -1] = (u[:, :, -1] - u[:, :, -2]) / dt
    uxx = torch.zeros_like(u)
    uxx[:, 1:-1, :] = (u[:, 2:, :] - 2.0 * u[:, 1:-1, :] + u[:, :-2, :]) / (dx * dx)
    residual = ut - pde.diffusion * uxx - pde.reaction * (u ** 2) - case.source.unsqueeze(0)
    u_mean, u_std = stats["u_mean"].to(u.device), stats["u_std"].to(u.device).clamp_min(1e-6)
    residual_rms = stats["residual_rms"].to(u.device).clamp_min(1e-6)
    s_mean, s_std = stats["source_mean"].to(u.device), stats["source_std"].to(u.device).clamp_min(1e-6)
    ic = case.ic.view(1, -1, 1).expand_as(u)
    bc = torch.zeros_like(u)
    bc[:, 0, :] = case.bc_left
    bc[:, -1, :] = case.bc_right
    source = case.source.unsqueeze(0).expand_as(u)
    fields = torch.stack([(u - u_mean) / u_std, residual / residual_rms, (source - s_mean) / s_std, (ic - u_mean) / u_std, (bc - u_mean) / u_std], dim=1)
    scalars = torch.tensor([pde.diffusion, pde.reaction, float(step_frac), 1.0], dtype=torch.float32, device=u.device).view(1, -1).expand(u.shape[0], -1)
    return fields, scalars


def batched_pde_step(u: torch.Tensor, delta: torch.Tensor, case: ReactionDiffusionCase) -> torch.Tensor:
    out = (u + delta).clamp(-8.0, 8.0)
    out[:, :, 0] = case.ic
    out[:, 0, :] = case.bc_left
    out[:, -1, :] = case.bc_right
    return out


def oracle_local_candidates(z_current: torch.Tensor, z_supervised: torch.Tensor, latent_mean: torch.Tensor, latent_std: torch.Tensor, cfg: dict) -> torch.Tensor:
    """Shared bounded local set for both immediate and long-horizon oracle choices."""
    n = int(cfg["solver_v2"]["oracle_headroom_candidates"])
    if n < 8:
        raise ValueError("oracle_headroom_candidates must be at least 8.")
    values = [z_current.squeeze(0), z_supervised.squeeze(0)]
    scale_values = [0.10, 0.25, 0.50]
    for scale in scale_values:
        direction = torch.randn_like(z_supervised.squeeze(0))
        direction = direction / torch.sqrt(torch.mean(direction ** 2)).clamp_min(1e-6)
        values.extend([z_supervised.squeeze(0) + scale * latent_std.squeeze(0) * direction, z_supervised.squeeze(0) - scale * latent_std.squeeze(0) * direction])
    for idx in range(n - len(values)):
        direction = torch.randn_like(z_supervised.squeeze(0))
        direction = direction / torch.sqrt(torch.mean(direction ** 2)).clamp_min(1e-6)
        values.append(z_supervised.squeeze(0) + scale_values[idx % len(scale_values)] * latent_std.squeeze(0) * direction)
    lo = latent_mean.squeeze(0) - float(cfg["solver_v2"]["verified_candidate_clip_std"]) * latent_std.squeeze(0)
    hi = latent_mean.squeeze(0) + float(cfg["solver_v2"]["verified_candidate_clip_std"]) * latent_std.squeeze(0)
    return torch.stack(values[:n], dim=0).clamp(lo, hi)


@torch.no_grad()
def oracle_candidate_rollouts(
    cfg: dict,
    case: ReactionDiffusionCase,
    u: torch.Tensor,
    candidates: torch.Tensor,
    continuation_actor: DeterministicNeuralOperatorActor,
    decoder: CorrectionOperatorDecoder,
    pde: ReactionDiffusionPDE,
    stats: dict,
    horizon: int,
    start_step: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return immediate rewards, true K-step returns, and final errors for every candidate."""
    count = candidates.shape[0]
    cur = u.unsqueeze(0).expand(count, -1, -1).clone()
    fields, scalars = state_from_u(pde, case, u, start_step / max(1, horizon), stats)
    fields_b, scalars_b = fields.unsqueeze(0).expand(count, -1, -1, -1), scalars.unsqueeze(0).expand(count, -1)
    delta = decoder(fields_b.to(device), scalars_b.to(device), candidates, stats, float(cfg["solver_v2"]["temperature"]))
    nxt = batched_pde_step(cur, delta, case)
    e0 = pde.relative_l2(u, case.gt)
    e1 = torch.sqrt(torch.sum((nxt - case.gt.unsqueeze(0)) ** 2, dim=(1, 2))) / torch.norm(case.gt).clamp_min(1e-8)
    immediate = torch.log((e0 + 1e-8) / (e1 + 1e-8)) - float(cfg["solver_v2"]["lambda_action"]) * torch.mean(delta ** 2, dim=(1, 2))
    returns = immediate.clone()
    cur = nxt
    for k in range(start_step + 1, horizon):
        fields_b, scalars_b = batched_state_from_u(pde, case, cur, k / max(1, horizon), stats)
        z = continuation_actor(fields_b, scalars_b, stats, float(cfg["solver_v2"]["temperature"]))
        delta = decoder(fields_b, scalars_b, z, stats, float(cfg["solver_v2"]["temperature"]))
        nxt = batched_pde_step(cur, delta, case)
        e_prev = torch.sqrt(torch.sum((cur - case.gt.unsqueeze(0)) ** 2, dim=(1, 2))) / torch.norm(case.gt).clamp_min(1e-8)
        e_next = torch.sqrt(torch.sum((nxt - case.gt.unsqueeze(0)) ** 2, dim=(1, 2))) / torch.norm(case.gt).clamp_min(1e-8)
        returns += torch.log((e_prev + 1e-8) / (e_next + 1e-8)) - float(cfg["solver_v2"]["lambda_action"]) * torch.mean(delta ** 2, dim=(1, 2))
        cur = nxt
    final_error = torch.sqrt(torch.sum((cur - case.gt.unsqueeze(0)) ** 2, dim=(1, 2))) / torch.norm(case.gt).clamp_min(1e-8)
    return immediate, returns, final_error


def load_oracle_components(cfg: dict, npz_path: Path, seed: int, device: torch.device) -> tuple[FNOInitializer, DeterministicNeuralOperatorActor, CorrectionOperatorDecoder, ReactionDiffusionPDE, dict]:
    set_seed(seed)
    pde = ReactionDiffusionPDE(float(cfg["benchmark"]["diffusion"]), float(cfg["benchmark"]["reaction"]))
    stats = fit_train_stats(cfg, npz_path, pde, device, ROOT / "checkpoints" / "solver_v2")
    initializer = train_initializer(cfg, npz_path, seed, pde, device)
    train_raw = generate_positive_trajectories(cfg, npz_path, initializer, pde, stats, seed, "train", device)
    encoder, decoder, zmeta = train_correction_autoencoder(cfg, train_raw, stats, seed, device)
    del encoder
    stats.update(zmeta)
    actor = DeterministicNeuralOperatorActor(width=int(cfg["solver_v2"]["width"]), modes=int(cfg["solver_v2"]["modes"]), depth=int(cfg["solver_v2"]["actor_depth"]), latent_dim=int(cfg["solver_v2"]["latent_dim"]), state_dim=int(cfg["solver_v2"]["state_dim"])).to(device)
    payload = torch.load(ROOT / "checkpoints" / "solver_v2" / f"actor_pretrain_{V2_CACHE_VERSION}_seed{seed}.pt", map_location=device)
    actor.load_state_dict(payload["actor"])
    actor.eval()
    decoder.eval()
    return initializer, actor, decoder, pde, stats


def oracle_headroom_diagnostic(cfg: dict, npz_path: Path, seed: int, horizon: int, device: torch.device) -> pd.DataFrame:
    initializer, actor, decoder, pde, stats = load_oracle_components(cfg, npz_path, seed, device)
    latent_mean = stats["latent_mean"].to(device).view(1, -1)
    latent_std = stats["latent_std"].to(device).view(1, -1)
    ds = ReactionDiffusionDataset(npz_path, "val", int(cfg["solver_v2"]["val_cases"]))
    rows = []
    for case_idx in tqdm(range(len(ds)), desc=f"solver_v2:oracle_headroom:{seed}:K{horizon}"):
        case = pde.make_case(ds[case_idx], device)
        u = initial_solution(initializer, pde, case)
        for step in range(horizon):
            fields, scalars = state_from_u(pde, case, u, step / max(1, horizon), stats)
            z_sup = actor(fields.unsqueeze(0), scalars.unsqueeze(0), stats, float(cfg["solver_v2"]["temperature"]))
            candidates = oracle_local_candidates(z_sup, z_sup, latent_mean, latent_std, cfg)
            immediate, returns, final_errors = oracle_candidate_rollouts(cfg, case, u, candidates, actor, decoder, pde, stats, horizon, step, device)
            greedy_idx = int(torch.argmax(immediate).item())
            long_idx = int(torch.argmax(returns).item())
            greedy_error = float(final_errors[greedy_idx].cpu())
            long_error = float(final_errors[long_idx].cpu())
            gain = greedy_error - long_error
            rows.append({
                "Seed": seed, "Case": case_idx, "Step": step, "Horizon": horizon,
                "GreedyImmediateReward": float(immediate[greedy_idx].cpu()), "GreedyFinalError": greedy_error,
                "LongHorizonReturn": float(returns[long_idx].cpu()), "LongHorizonFinalError": long_error,
                "AbsoluteOracleGain": gain, "RelativeOracleGain": gain / max(greedy_error, 1e-8),
                "GreedyActionShift": float(normalized_rms_shift(candidates[greedy_idx : greedy_idx + 1], z_sup, latent_std).item()),
                "LongActionShift": float(normalized_rms_shift(candidates[long_idx : long_idx + 1], z_sup, latent_std).item()),
                "ActionsDifferent": bool(not torch.allclose(candidates[greedy_idx], candidates[long_idx], atol=1e-6, rtol=1e-5)),
            })
            with torch.no_grad():
                z_continue = actor(fields.unsqueeze(0), scalars.unsqueeze(0), stats, float(cfg["solver_v2"]["temperature"]))
                delta_continue = decoder(fields.unsqueeze(0), scalars.unsqueeze(0), z_continue, stats, float(cfg["solver_v2"]["temperature"])).squeeze(0)
            u = pde.step(u, delta_continue, case).detach()
    return pd.DataFrame(rows)


def run_oracle_headroom_stage(cfg: dict, npz_path: Path, seeds: list[int], horizon: int, device: torch.device) -> pd.DataFrame:
    diagnostics = [oracle_headroom_diagnostic(cfg, npz_path, seed, horizon, device) for seed in seeds]
    detail = pd.concat(diagnostics, ignore_index=True)
    summary = pd.DataFrame([{
        "Horizon": horizon,
        "MeanRelativeOracleGain": float(detail["RelativeOracleGain"].mean()),
        "MedianRelativeOracleGain": float(detail["RelativeOracleGain"].median()),
        "PositiveGainRate": float((detail["AbsoluteOracleGain"] > 0.0).mean()),
        "ActionsDifferentRate": float(detail["ActionsDifferent"].mean()),
        "Samples": len(detail),
    }])
    table_dir = ROOT / "results" / "solver_v2" / "tables"
    detail_path = table_dir / "oracle_headroom.csv"
    summary_path = table_dir / "oracle_headroom_by_horizon.csv"
    if detail_path.exists():
        old = pd.read_csv(detail_path)
        detail = pd.concat([old[~old["Horizon"].eq(horizon)], detail], ignore_index=True)
    if summary_path.exists():
        old_summary = pd.read_csv(summary_path)
        summary = pd.concat([old_summary[~old_summary["Horizon"].eq(horizon)], summary], ignore_index=True).sort_values("Horizon")
    detail.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)
    print("Oracle long-horizon headroom diagnostic completed.")
    print(summary.to_string(index=False))
    return summary


def critic_ranking_validation(cfg: dict, val_raw: list[RawCorrectionTransition], actor: DeterministicNeuralOperatorActor, supervised_actor: DeterministicNeuralOperatorActor, critic: TwinOperatorCritic, decoder: CorrectionOperatorDecoder, pde: ReactionDiffusionPDE, stats: dict, device: torch.device, label: str) -> pd.DataFrame:
    rows = []
    group_metrics: list[dict[str, float]] = []
    latent_std = stats.get("latent_std", torch.ones(int(cfg["solver_v2"]["latent_dim"]))).to(device).view(1, -1)
    latent_mean = stats.get("latent_mean", torch.zeros(int(cfg["solver_v2"]["latent_dim"]))).to(device).view(1, -1)
    actor.eval()
    supervised_actor.eval()
    critic.eval()
    decoder.eval()
    for idx, tr in enumerate(val_raw[: min(len(val_raw), 80)]):
        fields = tr.state_fields.unsqueeze(0).to(device)
        scalars = tr.state_scalars.unsqueeze(0).to(device)
        case, u = raw_case_from_transition(tr, stats, pde, device)
        with torch.no_grad():
            z_sup = supervised_actor(fields, scalars, stats, float(cfg["solver_v2"]["temperature"]))
            z_actor = actor(fields, scalars, stats, float(cfg["solver_v2"]["temperature"]))
            candidates = structured_latent_candidates(z_sup, z_actor, latent_mean, latent_std)
            local_q, local_r, local_names = [], [], []
            for name, z in candidates:
                rv, _, _, _ = rollout_k_step_return(cfg, case, u, z, supervised_actor, decoder, pde, stats, device, tr.step_idx)
                qv = float(critic.q_min(fields, scalars, z).item())
                local_q.append(qv)
                local_r.append(float(rv))
                local_names.append(name)
                rows.append({"variant": label, "state": idx, "candidate": name, "q": qv, "return": float(rv), "horizon": int(cfg["solver_v2"]["horizon"])})
            pair_ok = pair_total = 0
            for a in range(len(local_q)):
                for b in range(a + 1, len(local_q)):
                    dr = local_r[a] - local_r[b]
                    dq = local_q[a] - local_q[b]
                    if abs(dr) >= float(cfg["solver_v2"]["critic_return_margin"]):
                        pair_total += 1
                        pair_ok += int(np.sign(dr) == np.sign(dq))
            _, local_spearman = _corr_metrics(local_q, local_r)
            best_return = int(np.argmax(local_r))
            top_q = np.argsort(local_q)[-2:]
            group_metrics.append({
                "spearman": local_spearman,
                "pairwise": float(pair_ok / max(1, pair_total)),
                "top1": float(int(np.argmax(local_q)) == best_return),
                "top2": float(best_return in top_q),
            })
    metrics = pd.DataFrame(group_metrics)
    summary = {
        "variant": label,
        "state": -1,
        "candidate": "summary",
        "q": float("nan"),
        "return": float("nan"),
        "horizon": int(cfg["solver_v2"]["horizon"]),
        "pearson": float("nan"),
        "spearman": float(metrics["spearman"].mean()) if not metrics.empty else 0.0,
        "pairwise": float(metrics["pairwise"].mean()) if not metrics.empty else 0.0,
        "top1": float(metrics["top1"].mean()) if not metrics.empty else 0.0,
        "top2": float(metrics["top2"].mean()) if not metrics.empty else 0.0,
    }
    out = pd.DataFrame(rows + [summary])
    for key, value in summary.items():
        if key not in out:
            out[key] = value
    out.loc[out["candidate"].eq("summary"), list(summary)] = list(summary.values())
    return out


def critic_candidate_transitions(
    cfg: dict,
    base_rows: list[SolverTransition],
    actor: DeterministicNeuralOperatorActor,
    supervised_actor: DeterministicNeuralOperatorActor,
    decoder: CorrectionOperatorDecoder,
    pde: ReactionDiffusionPDE,
    stats: dict,
    device: torch.device,
    label: str,
) -> list[SolverTransition]:
    rows: list[SolverTransition] = []
    latent_std = stats.get("latent_std", torch.ones(int(cfg["solver_v2"]["latent_dim"]))).to(device).view(1, -1)
    latent_mean = stats.get("latent_mean", torch.zeros(int(cfg["solver_v2"]["latent_dim"]))).to(device).view(1, -1)
    actor.eval()
    supervised_actor.eval()
    decoder.eval()
    for idx, tr in enumerate(base_rows[: min(len(base_rows), int(cfg["solver_v2"]["critic_candidate_groups"]))]):
        fields = tr.state_fields.unsqueeze(0).to(device)
        scalars = tr.state_scalars.unsqueeze(0).to(device)
        case, u = raw_case_from_transition(tr, stats, pde, device)
        with torch.no_grad():
            z_sup = supervised_actor(fields, scalars, stats, float(cfg["solver_v2"]["temperature"]))
            z_actor = actor(fields, scalars, stats, float(cfg["solver_v2"]["temperature"]))
            candidates = structured_latent_candidates(z_sup, z_actor, latent_mean, latent_std)
            for name, z in candidates:
                k_return, _, delta, r = rollout_k_step_return(cfg, case, u, z, supervised_actor, decoder, pde, stats, device, tr.step_idx)
                next_u = pde.step(u, delta, case)
                next_fields, next_scalars = state_from_u(pde, case, next_u, 1.0, stats)
                rows.append(
                    SolverTransition(
                        case_id=tr.case_id,
                        episode_id=900000 + idx,
                        step_idx=0,
                        split="train",
                        gt=tr.gt,
                        state_fields=tr.state_fields,
                        state_scalars=tr.state_scalars,
                        delta_u=delta.detach().cpu(),
                        reward=r["reward"],
                        next_fields=next_fields.detach().cpu(),
                        next_scalars=next_scalars.detach().cpu(),
                        done=1.0,
                        error_before=r["error_before"],
                        error_after=r["error_after"],
                        residual_before=r["residual_before"],
                        residual_after=r["residual_after"],
                        physics_before=r["physics_before"],
                        physics_after=r["physics_after"],
                        action_norm=r["action_norm"],
                        source_policy=f"critic_candidate_{label}_{name}",
                        action=z.squeeze(0).detach().cpu(),
                        mc_return=float(k_return),
                        group_id=f"{label}_state_{idx}",
                        candidate_name=name,
                    )
                )
    return rows


def latent_controllability_diagnostics(
    cfg: dict,
    val_raw: list[RawCorrectionTransition],
    actor: DeterministicNeuralOperatorActor,
    decoder: CorrectionOperatorDecoder,
    pde: ReactionDiffusionPDE,
    stats: dict,
    device: torch.device,
    seed: int,
) -> pd.DataFrame:
    rows = []
    latent_std = stats.get("latent_std", torch.ones(int(cfg["solver_v2"]["latent_dim"]))).to(device).view(1, -1)
    latent_mean = stats.get("latent_mean", torch.zeros(int(cfg["solver_v2"]["latent_dim"]))).to(device).view(1, -1)
    actor.eval()
    decoder.eval()
    for idx, tr in enumerate(val_raw[: min(len(val_raw), 80)]):
        fields = tr.state_fields.unsqueeze(0).to(device)
        scalars = tr.state_scalars.unsqueeze(0).to(device)
        case, u = raw_case_from_transition(tr, stats, pde, device)
        with torch.no_grad():
            z_sup = actor(fields, scalars, stats, float(cfg["solver_v2"]["temperature"]))
            delta_sup = decoder(fields, scalars, z_sup, stats, float(cfg["solver_v2"]["temperature"])).squeeze(0)
            u_sup = pde.step(u, delta_sup, case)
            reward_sup = pde.reward(u, u_sup, delta_sup, case, float(cfg["solver_v2"]["lambda_action"]))["reward"]
            candidates = structured_latent_candidates(z_sup, z_sup, latent_mean, latent_std)
            for name, z in candidates:
                delta = decoder(fields, scalars, z, stats, float(cfg["solver_v2"]["temperature"])).squeeze(0)
                u_next = pde.step(u, delta, case)
                reward = pde.reward(u, u_next, delta, case, float(cfg["solver_v2"]["lambda_action"]))["reward"]
                rows.append({
                    "Seed": seed,
                    "state": idx,
                    "candidate": name,
                    "latent_perturbation_magnitude": float(torch.norm(z - z_sup).detach().cpu()),
                    "correction_change_norm": float(torch.norm(delta - delta_sup).detach().cpu()),
                    "solution_change_norm": float(torch.norm(u_next - u_sup).detach().cpu()),
                    "reward": float(reward),
                    "reward_delta": float(reward - reward_sup),
                })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    corr = float(df["latent_perturbation_magnitude"].corr(df["correction_change_norm"])) if df["latent_perturbation_magnitude"].std() > 0 else 0.0
    summary = {
        "Seed": seed,
        "state": -1,
        "candidate": "summary",
        "latent_perturbation_magnitude": float(df["latent_perturbation_magnitude"].mean()),
        "correction_change_norm": float(df[df["candidate"].ne("supervised")]["correction_change_norm"].mean()),
        "solution_change_norm": float(df[df["candidate"].ne("supervised")]["solution_change_norm"].mean()),
        "reward": float(df["reward"].mean()),
        "reward_delta": float(df.groupby("state")["reward"].var().mean()),
        "reward_variance": float(df.groupby("state")["reward"].var().mean()),
        "action_effect_correlation": corr,
    }
    return pd.concat([df, pd.DataFrame([summary])], ignore_index=True)


def policy_shift_diagnostics(
    cfg: dict,
    val_raw: list[RawCorrectionTransition],
    actor_rl: DeterministicNeuralOperatorActor,
    actor_sup: DeterministicNeuralOperatorActor,
    decoder: CorrectionOperatorDecoder,
    stats: dict,
    device: torch.device,
    seed: int,
    policy_label: str,
) -> pd.DataFrame:
    rows = []
    actor_rl.eval()
    actor_sup.eval()
    decoder.eval()
    for idx, tr in enumerate(val_raw[: min(len(val_raw), 80)]):
        fields = tr.state_fields.unsqueeze(0).to(device)
        scalars = tr.state_scalars.unsqueeze(0).to(device)
        with torch.no_grad():
            z_sup = actor_sup(fields, scalars, stats, float(cfg["solver_v2"]["temperature"]))
            z_rl = actor_rl(fields, scalars, stats, float(cfg["solver_v2"]["temperature"]))
            d_sup = decoder(fields, scalars, z_sup, stats, float(cfg["solver_v2"]["temperature"]))
            d_rl = decoder(fields, scalars, z_rl, stats, float(cfg["solver_v2"]["temperature"]))
        rows.append({
            "Seed": seed,
            "Policy": policy_label,
            "state": idx,
            "latent_policy_shift": float(torch.norm(z_rl - z_sup).detach().cpu()),
            "correction_policy_shift": float(torch.norm(d_rl - d_sup).detach().cpu()),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return pd.concat([df, pd.DataFrame([{
        "Seed": seed,
        "Policy": policy_label,
        "state": -1,
        "latent_policy_shift": float(df["latent_policy_shift"].mean()),
        "correction_policy_shift": float(df["correction_policy_shift"].mean()),
    }])], ignore_index=True)


def validation_solver_error(cfg: dict, npz_path: Path, initializer: FNOInitializer, actor: DeterministicNeuralOperatorActor, decoder: CorrectionOperatorDecoder, pde: ReactionDiffusionPDE, stats: dict, device: torch.device) -> float:
    ds = ReactionDiffusionDataset(npz_path, "val", int(cfg["solver_v2"]["val_cases"]))
    steps = int(cfg["solver_v2"]["horizon"])
    vals = []
    actor.eval()
    decoder.eval()
    for i in range(len(ds)):
        case = pde.make_case(ds[i], device)
        u = initial_solution(initializer, pde, case)
        for k in range(steps):
            fields, scalars = state_from_u(pde, case, u, k / max(1, steps), stats)
            with torch.no_grad():
                z = actor(fields.unsqueeze(0), scalars.unsqueeze(0), stats, float(cfg["solver_v2"]["temperature"]))
                delta = decoder(fields.unsqueeze(0), scalars.unsqueeze(0), z, stats, float(cfg["solver_v2"]["temperature"])).squeeze(0)
            u = pde.step(u, delta, case)
        vals.append(float(pde.relative_l2(u, case.gt).detach().cpu()))
    return float(np.mean(vals))


def train_td3_variant(cfg: dict, transitions: list[SolverTransition], val_raw: list[RawCorrectionTransition], actor: DeterministicNeuralOperatorActor, supervised_actor: DeterministicNeuralOperatorActor, decoder: CorrectionOperatorDecoder, seed: int, variant: str, stats: dict, pde: ReactionDiffusionPDE, device: torch.device, epochs_override: int | None = None, controllability_ok: bool = True) -> tuple[DeterministicNeuralOperatorActor, TwinOperatorCritic, list[dict[str, float]], pd.DataFrame]:
    local_cfg = deepcopy(cfg)
    if variant == "scratch":
        local_cfg["solver_v2"]["lambda_bc_start"] = 0.0
    if variant == "td3_bc":
        local_cfg["solver_v2"]["lambda_phys"] = 0.0
    if epochs_override is not None:
        local_cfg["solver_v2"]["td3_epochs"] = int(epochs_override)
    critic = TwinOperatorCritic(width=int(cfg["solver_v2"]["width"]), modes=int(cfg["solver_v2"]["modes"]), depth=int(cfg["solver_v2"]["critic_depth"]), latent_dim=int(cfg["solver_v2"]["latent_dim"]), state_dim=int(cfg["solver_v2"]["state_dim"])).to(device)
    trainer = TD3BCTrainer(actor, critic, decoder, local_cfg, stats, device, supervised_actor=supervised_actor, allow_actor_update=False, variant=variant)
    cache_version = f"{CRITIC_CACHE_VERSION}_td3bc_sup_anchor_v1" if variant == "td3_bc" else CRITIC_CACHE_VERSION
    ckpt = ROOT / "checkpoints" / "solver_v2" / f"td3_{cache_version}_{variant}_seed{seed}.pt"
    if ckpt.exists():
        actor, critic, hist = trainer.fit(transitions, [], ckpt)
        rank = critic_ranking_validation(cfg, val_raw, actor, supervised_actor, critic, decoder, pde, stats, device, variant)
        return actor, critic, hist, rank
    candidate_cache = ROOT / "checkpoints" / "solver_v2" / f"critic_candidates_{CRITIC_CACHE_VERSION}_seed{seed}.pt"
    if candidate_cache.exists():
        critic_rows = torch.load(candidate_cache, map_location="cpu", weights_only=False)
    else:
        critic_rows = critic_candidate_transitions(cfg, [tr for tr in transitions if tr.split == "train"], supervised_actor, supervised_actor, decoder, pde, stats, device, "shared")
        torch.save(critic_rows, candidate_cache)
    trainer.pretrain_critic_mc(critic_rows)
    rank = critic_ranking_validation(cfg, val_raw, actor, supervised_actor, critic, decoder, pde, stats, device, variant)
    summary = rank[rank["candidate"].eq("summary")].iloc[0]
    trainer.allow_actor_update = bool(
        controllability_ok
        and float(summary["spearman"]) > float(cfg["solver_v2"]["critic_gate_spearman"])
        and float(summary["pairwise"]) > float(cfg["solver_v2"]["critic_gate_pairwise"])
        and float(summary["top1"]) > float(cfg["solver_v2"]["critic_gate_top1"])
    )
    if not trainer.allow_actor_update:
        torch.save({"actor": trainer.actor.state_dict(), "critic": trainer.critic.state_dict(), "history": trainer.history, "actor_updates": trainer.actor_updates}, ckpt)
        return trainer.actor.eval(), trainer.critic.eval(), trainer.history, rank
    actor, critic, hist = trainer.fit(transitions, critic_rows, ckpt)
    final_rank = critic_ranking_validation(cfg, val_raw, actor, supervised_actor, critic, decoder, pde, stats, device, variant)
    return actor, critic, hist, final_rank


def rollout_method(method: str, cfg: dict, initializer: FNOInitializer, actor: DeterministicNeuralOperatorActor | None, critic: TwinOperatorCritic | None, decoder: CorrectionOperatorDecoder | None, pde: ReactionDiffusionPDE, stats: dict, case: ReactionDiffusionCase, steps: int, device: torch.device) -> tuple[torch.Tensor, list[dict[str, float]], float]:
    u = initial_solution(initializer, pde, case)
    trace = []
    start = time.perf_counter()
    for k in range(steps):
        if method == "Base FNO Initializer":
            break
        if method == "Gradient baseline":
            delta = gradient_delta(pde, case, u, stats)
            q1 = q2 = np.nan
        elif method == "PINN-style baseline":
            delta = pinn_delta(pde, case, u, stats, steps=3)
            q1 = q2 = np.nan
        elif actor is not None and decoder is not None:
            fields, scalars = state_from_u(pde, case, u, k / max(1, steps), stats)
            with torch.no_grad():
                z = actor(fields.unsqueeze(0).to(device), scalars.unsqueeze(0).to(device), stats, float(cfg["solver_v2"]["temperature"]))
                delta = decoder(fields.unsqueeze(0).to(device), scalars.unsqueeze(0).to(device), z, stats, float(cfg["solver_v2"]["temperature"])).squeeze(0)
                if critic is not None:
                    q1_t, q2_t = critic(fields.unsqueeze(0).to(device), scalars.unsqueeze(0).to(device), z)
                    q1, q2 = float(q1_t.item()), float(q2_t.item())
                else:
                    q1 = q2 = np.nan
        else:
            raise ValueError(method)
        next_u = pde.step(u, delta, case)
        r = pde.reward(u, next_u, delta, case, float(cfg["solver_v2"]["lambda_action"]))
        trace.append({"step": k + 1, "Relative L2": r["error_after"], "PDE residual norm": r["residual_after"], "Physics energy": r["physics_after"], "action norm": r["action_norm"], "Q1": q1, "Q2": q2, "reward": r["reward"]})
        u = next_u
    return u, trace, time.perf_counter() - start


def evaluate_methods(cfg: dict, npz_path: Path, seed: int, initializer: FNOInitializer, actors: dict[str, DeterministicNeuralOperatorActor], critics: dict[str, TwinOperatorCritic | None], decoder: CorrectionOperatorDecoder, pde: ReactionDiffusionPDE, stats: dict, device: torch.device) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ds = ReactionDiffusionDataset(npz_path, "test", int(cfg["solver_v2"]["eval_cases"]))
    rows, per_case, conv = [], [], []
    methods = ["Base FNO Initializer", "Supervised Latent Neural Operator Corrector", "TD3+BC", "Rollout-Verified RL Neural Operator Solver", "Gradient baseline", "PINN-style baseline"]
    for steps in tqdm(cfg["solver_v2"]["eval_steps"], desc=f"v2:evaluate:{seed}"):
        method_rows = {m: [] for m in methods}
        for i in range(len(ds)):
            case = pde.make_case(ds[i], device)
            base_u = initial_solution(initializer, pde, case)
            base_err = float(pde.relative_l2(base_u, case.gt).detach().cpu())
            for method in methods:
                actor = actors.get(method)
                critic = critics.get(method)
                u, trace, wall = rollout_method(method, cfg, initializer, actor, critic, decoder if actor is not None else None, pde, stats, case, int(steps), device)
                pm = pde.physics_metrics(u, case)
                err = float(pde.relative_l2(u, case.gt).detach().cpu())
                row = {
                    "Seed": seed,
                    "Method": method,
                    "Steps": int(steps),
                    "Relative L2": err,
                    "PDE residual norm": float(pm["residual_norm"].detach().cpu()),
                    "BC error": float(pm["bc_error"].detach().cpu()),
                    "IC error": float(pm["ic_error"].detach().cpu()),
                    "wall time": wall,
                    "solver steps": len(trace),
                    "action norm": float(np.mean([t["action norm"] for t in trace])) if trace else 0.0,
                    "return": float(np.sum([t["reward"] for t in trace])) if trace else 0.0,
                    "paired improvement": base_err - err,
                    "Case": i,
                }
                method_rows[method].append(row)
                per_case.append(row)
                for tr in trace:
                    conv.append({"Seed": seed, "Method": method, "Case": i, "StepCap": int(steps), **tr})
        for vals in method_rows.values():
            rows.extend(vals)
    return pd.DataFrame(rows), pd.DataFrame(per_case), pd.DataFrame(conv)


def summarize_main(per_case: pd.DataFrame) -> pd.DataFrame:
    metrics = ["Relative L2", "PDE residual norm", "BC error", "IC error", "wall time", "solver steps", "action norm", "return", "paired improvement"]
    rows = []
    for (method, steps), g in per_case.groupby(["Method", "Steps"]):
        row = {"Method": method, "Steps": steps}
        for metric in metrics:
            vals = g[metric].astype(float)
            row[f"{metric} mean"] = vals.mean()
            row[f"{metric} std"] = vals.std(ddof=0)
            row[f"{metric} median"] = vals.median()
            seed_vals = g.groupby("Seed")[metric].mean().astype(float)
            row[f"{metric} seed mean"] = seed_vals.mean()
            row[f"{metric} seed std"] = seed_vals.std(ddof=0)
        row["Samples"] = len(g)
        rows.append(row)
    return pd.DataFrame(rows)


def seed_summary(per_case: pd.DataFrame) -> pd.DataFrame:
    sub = per_case[per_case["Steps"].eq(10)]
    return sub.groupby(["Seed", "Method"], as_index=False).agg(relative_l2=("Relative L2", "mean"), residual=("PDE residual norm", "mean"), wall_time=("wall time", "mean"), return_mean=("return", "mean"), paired_improvement=("paired improvement", "mean"))


def residual_accuracy_quadrants(transitions: list[SolverTransition]) -> pd.DataFrame:
    df = pd.DataFrame([{"source_policy": tr.source_policy, "residual_delta": tr.residual_after - tr.residual_before, "error_delta": tr.error_after - tr.error_before} for tr in transitions])
    rows = []
    for source, g in df.groupby("source_policy"):
        for label, mask in [
            ("Residual better / Accuracy better", (g.residual_delta < 0) & (g.error_delta < 0)),
            ("Residual better / Accuracy worse", (g.residual_delta < 0) & (g.error_delta >= 0)),
            ("Residual worse / Accuracy better", (g.residual_delta >= 0) & (g.error_delta < 0)),
            ("Both worse", (g.residual_delta >= 0) & (g.error_delta >= 0)),
        ]:
            rows.append({"source_policy": source, "Quadrant": label, "Count": int(mask.sum()), "Fraction": float(mask.mean())})
    return pd.DataFrame(rows)


def build_training_summary(cfg: dict, seed: int, histories: dict[str, list[dict[str, float]]], rankings: list[pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for variant, hist in histories.items():
        for h in hist:
            row = dict(h)
            row["Seed"] = seed
            row["variant"] = variant
            rows.append(row)
    for rank in rankings:
        summary = rank[rank["candidate"].eq("summary")].iloc[0]
        rows.append({
            "Seed": seed,
            "variant": str(summary["variant"]) + "_ranking",
            "epoch": -999.0,
            "critic_loss": np.nan,
            "actor_loss": np.nan,
            "actor_updates_enabled": float(
                float(summary["spearman"]) > float(cfg["solver_v2"]["critic_gate_spearman"])
                and float(summary["pairwise"]) > float(cfg["solver_v2"]["critic_gate_pairwise"])
                and float(summary["top1"]) > float(cfg["solver_v2"]["critic_gate_top1"])
            ),
            "ranking_spearman": float(summary["spearman"]),
            "ranking_pairwise": float(summary["pairwise"]),
            "ranking_top1": float(summary.get("top1", np.nan)),
            "ranking_top2": float(summary.get("top2", np.nan)),
        })
    return pd.DataFrame(rows)


def verified_decision_states(
    cfg: dict,
    raw_rows: list[RawCorrectionTransition],
    initializer: FNOInitializer,
    policy: DeterministicNeuralOperatorActor,
    decoder: CorrectionOperatorDecoder,
    pde: ReactionDiffusionPDE,
    stats: dict,
    device: torch.device,
) -> list[tuple[ReactionDiffusionCase, torch.Tensor, torch.Tensor, torch.Tensor, int]]:
    """One on-policy state per physical training case, stratified across solver steps."""
    count = min(len(raw_rows), int(cfg["solver_v2"]["verified_states_per_cycle"]))
    if count <= 0:
        return []
    indices = np.linspace(0, len(raw_rows) - 1, count, dtype=int)
    rows = []
    policy.eval()
    for idx in tqdm(indices, desc="solver_v2:verified_states"):
        tr = raw_rows[int(idx)]
        case, _ = raw_case_from_transition(tr, stats, pde, device)
        u = initial_solution(initializer, pde, case)
        for k in range(tr.step_idx):
            fields, scalars = state_from_u(pde, case, u, k / max(1, int(cfg["solver_v2"]["horizon"])), stats)
            with torch.no_grad():
                z = policy(fields.unsqueeze(0), scalars.unsqueeze(0), stats, float(cfg["solver_v2"]["temperature"]))
                delta = decoder(fields.unsqueeze(0).to(device), scalars.unsqueeze(0).to(device), z, stats, float(cfg["solver_v2"]["temperature"])).squeeze(0)
            u = pde.step(u, delta, case).detach()
        fields, scalars = state_from_u(pde, case, u, tr.step_idx / max(1, int(cfg["solver_v2"]["horizon"])), stats)
        rows.append((case, u.detach(), fields.detach().cpu(), scalars.detach().cpu(), int(tr.step_idx)))
    return rows


def verified_candidates(z0: torch.Tensor, z_sup: torch.Tensor, latent_mean: torch.Tensor, latent_std: torch.Tensor, cfg: dict) -> torch.Tensor:
    """Exactly 48 bounded local candidates: current, supervised, and symmetric local probes."""
    n = int(cfg["solver_v2"]["verified_candidates"])
    if n < 2:
        raise ValueError("verified_candidates must include current and supervised policies.")
    scale_cycle = [0.10, 0.25, 0.50]
    values = [z0.squeeze(0), z_sup.squeeze(0)]
    for i in range(n - 2):
        center = z0.squeeze(0) if i % 2 == 0 else z_sup.squeeze(0)
        scale = scale_cycle[i % len(scale_cycle)]
        values.append(center + scale * latent_std.squeeze(0) * torch.randn_like(center))
    lo = latent_mean.squeeze(0) - float(cfg["solver_v2"]["verified_candidate_clip_std"]) * latent_std.squeeze(0)
    hi = latent_mean.squeeze(0) + float(cfg["solver_v2"]["verified_candidate_clip_std"]) * latent_std.squeeze(0)
    return torch.stack(values, dim=0).clamp(lo, hi)


def run_verified_policy_improvement(
    cfg: dict,
    npz_path: Path,
    seed: int,
    initializer: FNOInitializer,
    train_raw: list[RawCorrectionTransition],
    actor_sup: DeterministicNeuralOperatorActor,
    critic: TwinOperatorCritic,
    decoder: CorrectionOperatorDecoder,
    pde: ReactionDiffusionPDE,
    stats: dict,
    device: torch.device,
) -> tuple[TrustedPolicy, pd.DataFrame, pd.DataFrame, list[dict[str, float]], int]:
    ckpt = ROOT / "checkpoints" / "solver_v2" / f"verified_policy_{VERIFIED_PI_CACHE_VERSION}_seed{seed}.pt"
    latent_std = stats["latent_std"].to(device).view(1, -1)
    latent_mean = stats["latent_mean"].to(device).view(1, -1)
    raw_actor = deepcopy(actor_sup).to(device)
    if ckpt.exists():
        saved = torch.load(ckpt, map_location=device, weights_only=False)
        raw_actor.load_state_dict(saved["actor"])
        policy = TrustedPolicy(raw_actor, actor_sup, latent_std, float(saved["trust_radius"])).to(device).eval()
        return policy, pd.DataFrame(saved["cycles"]), pd.DataFrame(saved["screening"]), saved.get("history", []), int(saved["accepted_cycles"])

    critic.eval()
    actor_sup.eval()
    current: TrustedPolicy = TrustedPolicy(raw_actor, actor_sup, latent_std, float(cfg["solver_v2"]["verified_trust_radius_initial"])).to(device).eval()
    trust_radius = float(cfg["solver_v2"]["verified_trust_radius_initial"])
    accepted = 0
    cycle_rows: list[dict[str, float]] = []
    screening_rows: list[dict[str, float]] = []
    history: list[dict[str, float]] = []
    for cycle in range(int(cfg["solver_v2"]["online_cycles"])):
        val_before = validation_solver_error(cfg, npz_path, initializer, current, decoder, pde, stats, device)
        continuation = actor_sup if cycle == 0 else current
        targets: list[VerifiedTarget] = []
        states = verified_decision_states(cfg, train_raw, initializer, current, decoder, pde, stats, device)
        positive_advantages: list[float] = []
        for state_idx, (case, u, fields_cpu, scalars_cpu, start_step) in enumerate(tqdm(states, desc=f"solver_v2:verified_search:{seed}:{cycle}")):
            fields = fields_cpu.unsqueeze(0).to(device)
            scalars = scalars_cpu.unsqueeze(0).to(device)
            with torch.no_grad():
                z0 = current(fields, scalars, stats, float(cfg["solver_v2"]["temperature"]))
                z_sup = actor_sup(fields, scalars, stats, float(cfg["solver_v2"]["temperature"]))
                candidates = verified_candidates(z0, z_sup, latent_mean, latent_std, cfg)
                q = critic.q_min(fields.expand(len(candidates), -1, -1, -1), scalars.expand(len(candidates), -1), candidates)
            top_m = torch.topk(q, k=min(int(cfg["solver_v2"]["verified_top_m"]), len(candidates))).indices.tolist()
            selected = set(top_m)
            selected.add(0)  # The true current-policy baseline is always measured.
            remaining = [i for i in range(len(candidates)) if i not in selected]
            random_count = min(int(cfg["solver_v2"]["verified_random_candidates"]), len(remaining))
            # Convert random offsets back to candidate indices.
            if random_count:
                random_offsets = torch.randperm(len(remaining), device=device)[:random_count].cpu().tolist()
                selected.update(remaining[i] for i in random_offsets)
            selected_idx = sorted(selected)
            returns: dict[int, float] = {}
            for candidate_idx in selected_idx:
                value, _, _, _ = rollout_k_step_return(cfg, case, u, candidates[candidate_idx : candidate_idx + 1], continuation, decoder, pde, stats, device, start_step)
                returns[candidate_idx] = float(value)
            base_return = returns[0]
            values = np.asarray(list(returns.values()), dtype=float)
            margin = max(float(cfg["solver_v2"]["verified_advantage_margin_floor"]), float(cfg["solver_v2"]["verified_advantage_std_fraction"]) * float(values.std()))
            best_idx = max(returns, key=returns.get)
            advantage = returns[best_idx] - base_return
            positives_top = [returns[i] - base_return > 0.0 for i in top_m if i in returns]
            top1 = int(torch.argmax(q).item())
            top1_positive = float(returns[top1] - base_return > 0.0) if top1 in returns else float("nan")
            top_m_best = max((returns[i] for i in top_m if i in returns), default=base_return)
            screening_rows.append({
                "Seed": seed, "Cycle": cycle, "PrecisionAtM": float(np.mean(positives_top)) if positives_top else 0.0,
                "Top1TruePositiveRate": top1_positive, "BestFoundRegret": float(max(returns.values()) - top_m_best),
                "CandidateReturnSpread": float(values.std()), "State": state_idx,
            })
            if advantage > margin:
                target = project_to_trust_region(candidates[best_idx : best_idx + 1], z_sup, latent_std, trust_radius).squeeze(0).detach().cpu()
                targets.append(VerifiedTarget(fields_cpu, scalars_cpu, target, float(advantage)))
                positive_advantages.append(float(advantage))
        if targets:
            proposed_raw = deepcopy(current.actor).to(device)
            proposed, fit_history = train_verified_policy(proposed_raw, actor_sup, targets, stats, latent_std, trust_radius, cfg, device)
            history.extend([{**row, "cycle": float(cycle)} for row in fit_history])
            val_after = validation_solver_error(cfg, npz_path, initializer, proposed, decoder, pde, stats, device)
        else:
            proposed = current
            val_after = val_before
        accepted_cycle = bool(targets and val_after < val_before)
        if accepted_cycle:
            current = proposed.eval()
            accepted += 1
            trust_radius = float(cfg["solver_v2"]["verified_trust_radius_expanded"])
        shifts = []
        for target in targets:
            with torch.no_grad():
                f = target.fields.unsqueeze(0).to(device)
                sc = target.scalars.unsqueeze(0).to(device)
                z_sup = actor_sup(f, sc, stats, float(cfg["solver_v2"]["temperature"]))
                shifts.append(float(normalized_rms_shift(target.target.unsqueeze(0).to(device), z_sup, latent_std).item()))
        cycle_rows.append({
            "Seed": seed, "Cycle": cycle, "StatesEvaluated": len(states), "StatesWithPositiveCandidate": len(targets),
            "PositiveCandidateRate": float(len(targets) / max(1, len(states))),
            "MeanVerifiedAdvantage": float(np.mean(positive_advantages)) if positive_advantages else 0.0,
            "MedianVerifiedAdvantage": float(np.median(positive_advantages)) if positive_advantages else 0.0,
            "MeanTargetLatentShift": float(np.mean(shifts)) if shifts else 0.0,
            "ValErrorBefore": val_before, "ValErrorAfter": val_after, "CycleAccepted": accepted_cycle,
        })
    torch.save({"actor": current.actor.state_dict(), "trust_radius": current.radius, "cycles": cycle_rows, "screening": screening_rows, "history": history, "accepted_cycles": accepted}, ckpt)
    return current.eval(), pd.DataFrame(cycle_rows), pd.DataFrame(screening_rows), history, accepted


def run_seed(cfg: dict, npz_path: Path, seed: int, device: torch.device) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[SolverTransition], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    set_seed(seed)
    s = cfg["solver_v2"]
    pde = ReactionDiffusionPDE(float(cfg["benchmark"]["diffusion"]), float(cfg["benchmark"]["reaction"]))
    stats = fit_train_stats(cfg, npz_path, pde, device, ROOT / "checkpoints" / "solver_v2")
    initializer = train_initializer(cfg, npz_path, seed, pde, device)
    train_raw = generate_positive_trajectories(cfg, npz_path, initializer, pde, stats, seed, "train", device)
    val_raw = generate_positive_trajectories(cfg, npz_path, initializer, pde, stats, seed, "val", device)
    encoder, decoder, zmeta = train_correction_autoencoder(cfg, train_raw, stats, seed, device)
    stats.update(zmeta)
    supervised = encode_transitions(train_raw, encoder, device)
    val_latent = encode_transitions(val_raw, encoder, device)
    replay = supervised + val_latent
    actor_sup = DeterministicNeuralOperatorActor(width=int(s["width"]), modes=int(s["modes"]), depth=int(s["actor_depth"]), latent_dim=int(s["latent_dim"]), state_dim=int(s["state_dim"])).to(device)
    actor_sup = pretrain_actor(actor_sup, supervised, cfg, stats, device, ROOT / "checkpoints" / "solver_v2" / f"actor_pretrain_{V2_CACHE_VERSION}_seed{seed}.pt")
    sup_val = validation_solver_error(cfg, npz_path, initializer, actor_sup, decoder, pde, stats, device)
    controllability_df = latent_controllability_diagnostics(cfg, val_raw, actor_sup, decoder, pde, stats, device, seed)
    cont_summary = controllability_df[controllability_df["candidate"].eq("summary")]
    controllability_ok = bool(
        not cont_summary.empty
        and float(cont_summary["correction_change_norm"].iloc[0]) > 1e-4
        and float(cont_summary["reward_variance"].iloc[0]) > 1e-8
    )
    histories: dict[str, list[dict[str, float]]] = {"supervised": [{"epoch": 0.0, "critic_loss": np.nan, "actor_loss": np.nan, "val_error": sup_val}]}
    rankings: list[pd.DataFrame] = []
    _, critic_verified, critic_hist, rank_verified = train_td3_variant(
        cfg, replay, val_raw, deepcopy(actor_sup).to(device), actor_sup, decoder, seed,
        "verified_screening", stats, pde, device, controllability_ok=False,
    )
    histories["verified_critic"] = critic_hist
    rankings.append(rank_verified)
    verified_actor, verified_df, screening_df, verified_history, accepted_cycles = run_verified_policy_improvement(
        cfg, npz_path, seed, initializer, train_raw, actor_sup, critic_verified, decoder, pde, stats, device,
    )
    histories["rollout_verified"] = verified_history
    actor_td3bc, critic_td3bc, hist_td3bc, rank_td3bc = train_td3_variant(cfg, replay, val_raw, deepcopy(actor_sup).to(device), actor_sup, decoder, seed, "td3_bc", stats, pde, device, controllability_ok=controllability_ok)
    histories["td3_bc"] = hist_td3bc
    rankings.append(rank_td3bc)
    actors = {
        "Supervised Latent Neural Operator Corrector": actor_sup,
        "TD3+BC": actor_td3bc,
        "Rollout-Verified RL Neural Operator Solver": verified_actor,
    }
    critics = {"TD3+BC": critic_td3bc, "Rollout-Verified RL Neural Operator Solver": critic_verified}
    eval_df, per_case, conv = evaluate_methods(cfg, npz_path, seed, initializer, actors, critics, decoder, pde, stats, device)
    training_summary = build_training_summary(cfg, seed, histories, rankings)
    ranking_df = pd.concat(rankings, ignore_index=True)
    ranking_df["Seed"] = seed
    policy_shift_df = pd.concat([
        policy_shift_diagnostics(cfg, val_raw, verified_actor, actor_sup, decoder, stats, device, seed, "Rollout-Verified RL Neural Operator Solver"),
        policy_shift_diagnostics(cfg, val_raw, actor_td3bc, actor_sup, decoder, stats, device, seed, "TD3+BC"),
    ], ignore_index=True)
    selection_df = pd.DataFrame([{
        "Seed": seed,
        "SupervisedValidationError": sup_val,
        "VerifiedPolicyValidationError": float(verified_df.iloc[-1]["ValErrorAfter"]) if not verified_df.empty else sup_val,
        "AcceptedCycles": accepted_cycles,
        "SelectedProductionSource": "rollout_verified" if accepted_cycles else "supervised_anchor",
    }])
    seed_results = seed_summary(per_case)
    def err(method: str) -> float:
        values = seed_results[(seed_results["Seed"].eq(seed)) & (seed_results["Method"].eq(method))]["relative_l2"]
        return float(values.iloc[0]) if len(values) else float("nan")
    verified_shift = policy_shift_df[(policy_shift_df["Policy"].eq("Rollout-Verified RL Neural Operator Solver")) & (policy_shift_df["state"].eq(-1))].iloc[0]
    final_df = pd.DataFrame([{
        "Seed": seed, "BaseError": err("Base FNO Initializer"), "SupervisedError": err("Supervised Latent Neural Operator Corrector"),
        "VerifiedRLError": err("Rollout-Verified RL Neural Operator Solver"),
        "RL_vs_Supervised": err("Supervised Latent Neural Operator Corrector") - err("Rollout-Verified RL Neural Operator Solver"),
        "AcceptedCycles": accepted_cycles,
        "MeanVerifiedAdvantage": float(verified_df["MeanVerifiedAdvantage"].mean()) if not verified_df.empty else 0.0,
        "LatentPolicyShift": float(verified_shift["latent_policy_shift"]), "CorrectionPolicyShift": float(verified_shift["correction_policy_shift"]),
    }])
    return eval_df, per_case, conv, replay, training_summary, ranking_df, controllability_df, policy_shift_df, selection_df, verified_df, screening_df, final_df


def generate_figures(tables: dict[str, pd.DataFrame]) -> None:
    fig_dir = ROOT / "results" / "solver_v2" / "figures"
    main = tables["per_case_results"]
    conv = tables["convergence"]
    train = tables["training_summary"]
    trans = tables["transition_debug"]
    rank = tables["critic_long_horizon_ranking"]
    plt.figure(figsize=(8, 2.4))
    blocks = ["PDE", "FNO init", "state encoder", "latent actor z", "frozen FNO decoder", "projected update", "Twin-Q"]
    for i, b in enumerate(blocks):
        plt.text(i, 0.5, b, ha="center", va="center", bbox=dict(boxstyle="round,pad=0.25", fc="#edf7f6", ec="#1f2937"))
        if i < len(blocks) - 1:
            plt.arrow(i + 0.35, 0.5, 0.25, 0, head_width=0.04, color="#1f2937")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(fig_dir / "figure1_architecture.png", dpi=180)
    plt.close()
    for metric, fname in [("Relative L2", "figure3_l2_vs_step.png"), ("PDE residual norm", "figure4_residual_vs_step.png")]:
        plt.figure(figsize=(6, 4))
        for method, g in conv[conv["StepCap"].eq(10)].groupby("Method"):
            if method in ["Gradient baseline", "Supervised Latent Neural Operator Corrector", "Rollout-Verified RL Neural Operator Solver"]:
                gg = g.groupby("step")[metric].mean()
                plt.plot(gg.index, gg.values, marker="o", label=method)
        plt.xlabel("solver step")
        plt.ylabel(metric)
        plt.legend(fontsize=7)
        plt.tight_layout()
        plt.savefig(fig_dir / fname, dpi=180)
        plt.close()
    plt.figure(figsize=(6, 4))
    if not train.empty and "critic_loss" in train:
        for variant, g in train.groupby("variant"):
            if "ranking" not in str(variant):
                plt.plot(g.index, g["critic_loss"], label=str(variant))
    plt.xlabel("logged update")
    plt.ylabel("critic loss")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(fig_dir / "figure5_training_curve.png", dpi=180)
    plt.close()
    paired = main[(main["Steps"].eq(10)) & (main["Method"].isin(["Supervised Latent Neural Operator Corrector", "Rollout-Verified RL Neural Operator Solver"]))]
    pivot = paired.pivot_table(index=["Seed", "Case"], columns="Method", values="paired improvement")
    plt.figure(figsize=(5, 4))
    if {"Supervised Latent Neural Operator Corrector", "Rollout-Verified RL Neural Operator Solver"}.issubset(pivot.columns):
        plt.scatter(pivot["Supervised Latent Neural Operator Corrector"], pivot["Rollout-Verified RL Neural Operator Solver"], s=18, alpha=0.7)
        lo = float(np.nanmin(pivot.values))
        hi = float(np.nanmax(pivot.values))
        plt.plot([lo, hi], [lo, hi], color="black", linewidth=0.8)
    plt.xlabel("Supervised paired improvement")
    plt.ylabel("Rollout-verified RL paired improvement")
    plt.tight_layout()
    plt.savefig(fig_dir / "figure6_supervised_vs_rl_paired.png", dpi=180)
    plt.close()
    plt.figure(figsize=(5.5, 4))
    if not trans.empty:
        plt.axhline(0, color="black", linewidth=0.8)
        plt.axvline(0, color="black", linewidth=0.8)
        plt.scatter(trans["residual_after"] - trans["residual_before"], trans["error_after"] - trans["error_before"], s=8, alpha=0.35)
    plt.xlabel("Delta residual")
    plt.ylabel("Delta accuracy error")
    plt.tight_layout()
    plt.savefig(fig_dir / "figure7_residual_accuracy_change.png", dpi=180)
    plt.close()
    plt.figure(figsize=(6, 4))
    for method, g in main[main["Steps"].eq(10)].groupby("Method"):
        plt.scatter(g["wall time"], g["Relative L2"], s=14, alpha=0.55, label=method)
    plt.xlabel("wall time")
    plt.ylabel("Relative L2")
    plt.legend(fontsize=6)
    plt.tight_layout()
    plt.savefig(fig_dir / "figure8_error_vs_walltime.png", dpi=180)
    plt.close()
    plt.figure(figsize=(5.5, 4))
    detail = rank[~rank["candidate"].eq("summary")]
    if not detail.empty:
        plt.scatter(detail["q"], detail["return"], s=8, alpha=0.35)
    plt.xlabel("Critic Q")
    plt.ylabel("Actual K-step return")
    plt.tight_layout()
    plt.savefig(fig_dir / "figure9_critic_ranking.png", dpi=180)
    plt.close()


def generate_representative_figure(cfg: dict, npz_path: Path, seed: int, device: torch.device) -> None:
    fig_dir = ROOT / "results" / "solver_v2" / "figures"
    pde = ReactionDiffusionPDE(float(cfg["benchmark"]["diffusion"]), float(cfg["benchmark"]["reaction"]))
    stats = fit_train_stats(cfg, npz_path, pde, device, ROOT / "checkpoints" / "solver_v2")
    initializer = FNOInitializer(width=int(cfg["solver_v2"]["width"]), modes=int(cfg["solver_v2"]["modes"]), depth=3).to(device)
    initializer.load_state_dict(torch.load(ROOT / "checkpoints" / "solver_v2" / f"fno_initializer_{INITIALIZER_CACHE_VERSION}_seed{seed}.pt", map_location=device)["model"])
    ae = torch.load(ROOT / "checkpoints" / "solver_v2" / f"correction_autoencoder_{V2_CACHE_VERSION}_seed{seed}.pt", map_location=device)
    stats.update(ae.get("latent_stats", {}))
    decoder = CorrectionOperatorDecoder(latent_dim=int(cfg["solver_v2"]["latent_dim"]), width=int(cfg["solver_v2"]["width"]), modes=int(cfg["solver_v2"]["modes"]), depth=int(cfg["solver_v2"]["actor_depth"])).to(device)
    decoder.load_state_dict(ae["decoder"])
    actor_sup = DeterministicNeuralOperatorActor(width=int(cfg["solver_v2"]["width"]), modes=int(cfg["solver_v2"]["modes"]), depth=int(cfg["solver_v2"]["actor_depth"]), latent_dim=int(cfg["solver_v2"]["latent_dim"]), state_dim=int(cfg["solver_v2"]["state_dim"])).to(device)
    actor = DeterministicNeuralOperatorActor(width=int(cfg["solver_v2"]["width"]), modes=int(cfg["solver_v2"]["modes"]), depth=int(cfg["solver_v2"]["actor_depth"]), latent_dim=int(cfg["solver_v2"]["latent_dim"]), state_dim=int(cfg["solver_v2"]["state_dim"])).to(device)
    selected = ROOT / "checkpoints" / "solver_v2" / f"verified_policy_{VERIFIED_PI_CACHE_VERSION}_seed{seed}.pt"
    fallback = ROOT / "checkpoints" / "solver_v2" / f"actor_pretrain_{V2_CACHE_VERSION}_seed{seed}.pt"
    actor_sup.load_state_dict(torch.load(fallback, map_location=device)["actor"])
    if selected.exists():
        payload = torch.load(selected, map_location=device)
        actor.load_state_dict(payload["actor"])
        actor = TrustedPolicy(actor, actor_sup, stats["latent_std"].to(device), float(payload["trust_radius"])).to(device).eval()
    else:
        actor = actor_sup.eval()
    ds = ReactionDiffusionDataset(npz_path, "test", 1)
    case = pde.make_case(ds[0], device)
    u0 = initial_solution(initializer, pde, case)
    u = u0.detach()
    for k in range(10):
        fields, scalars = state_from_u(pde, case, u, k / 10.0, stats)
        with torch.no_grad():
            z = actor(fields.unsqueeze(0), scalars.unsqueeze(0), stats, float(cfg["solver_v2"]["temperature"]))
            delta = decoder(fields.unsqueeze(0), scalars.unsqueeze(0), z, stats, float(cfg["solver_v2"]["temperature"])).squeeze(0)
        u = pde.step(u, delta, case)
    panels = [("GT", case.gt), ("Initial FNO", u0), ("RL refined", u), ("Absolute error", (u - case.gt).abs()), ("Initial residual", pde.residual(u0, case)), ("Refined residual", pde.residual(u, case))]
    plt.figure(figsize=(10, 4))
    for idx, (title, arr) in enumerate(panels, 1):
        plt.subplot(2, 3, idx)
        plt.imshow(arr.detach().cpu().numpy(), aspect="auto", cmap="viridis")
        plt.title(title)
        plt.colorbar(fraction=0.046)
    plt.tight_layout()
    plt.savefig(fig_dir / "figure2_representative_solution.png", dpi=180)
    plt.close()


def write_docs(tables: dict[str, pd.DataFrame]) -> None:
    main = tables["main_results"]
    contribution = tables["final_rl_contribution"]
    mean = contribution.mean(numeric_only=True)
    rl_gain = float(mean.get("RL_vs_Supervised", float("nan")))
    accepted = float(mean.get("AcceptedCycles", 0.0))
    positive_test_seeds = int((contribution["RL_vs_Supervised"] > 0.0).sum())
    headroom_path = ROOT / "results" / "solver_v2" / "tables" / "oracle_headroom_by_horizon.csv"
    headroom = pd.read_csv(headroom_path) if headroom_path.exists() else pd.DataFrame()
    headroom_sufficient = bool(
        not headroom.empty
        and ((headroom["MeanRelativeOracleGain"] >= 0.02) & (headroom["PositiveGainRate"] >= 0.60) & (headroom["ActionsDifferentRate"] > 0.0)).any()
    )
    if not headroom.empty and not headroom_sufficient:
        claim = "Current PDE formulation provides insufficient long-horizon policy headroom for meaningful RL improvement."
    elif np.isfinite(rl_gain) and rl_gain > 0.0 and accepted > 0.0:
        claim = (
            f"The three-seed mean favors rollout-verified conservative policy improvement over the supervised latent corrector "
            f"by {rl_gain:.6f} Relative L2, but the result is small and seed-level outcomes are mixed "
            f"({positive_test_seeds}/3 positive test differences)."
        )
    elif accepted > 0.0:
        claim = "Rollout-verified updates were accepted on validation, but the final three-seed test result does not establish an improvement over supervised correction."
    else:
        claim = "No rollout-verified policy cycle met the validation acceptance criterion; the reported RL policy remains at the supervised trust-region anchor."
    oracle_section = ""
    if not headroom.empty:
        oracle_section = "## Oracle Long-Horizon Headroom\n\n" + headroom.to_markdown(index=False) + "\n\n"
        if not headroom_sufficient:
            oracle_section += "No further Greedy-vs-RL policy training was run after this validation result.\n\n"
    text = (
        "# Solver V2 Results\n\n"
        + claim + "\n\n"
        + oracle_section
        + "The main RL algorithm uses critic screening only: candidate actions are ranked by Twin-Q, then 10-12 candidates per state receive true K-step PDE rollout returns. Actor regression uses only verified positive-advantage targets and a supervised-policy trust penalty; no Q-gradient enters the actor.\n\n"
        + "## Final Contribution\n\n" + contribution.to_markdown(index=False) + "\n\n"
        + "## Verified Cycles\n\n" + tables["verified_policy_improvement"].to_markdown(index=False) + "\n\n"
        + "## Main Table\n\n" + main.to_markdown(index=False) + "\n"
    )
    (ROOT / "docs" / "solver_v2_results.md").write_text(text, encoding="utf-8")
    (ROOT / "docs" / "solver_v2_claims.md").write_text("# Solver V2 Claims\n\n" + claim + "\n", encoding="utf-8")


def run_pipeline(mode: str, seeds: list[int] | None = None) -> None:
    cfg = load_config()
    ensure_dirs()
    npz_path = prepare_reaction_diffusion(cfg, ROOT)
    device = get_device(str(cfg.get("device", "cuda")))
    active_seeds = seeds if seeds is not None else [int(seed) for seed in cfg["solver_v2"]["seeds"]]
    all_eval, all_per_case, all_conv, all_transitions, all_training, all_ranking, all_controllability, all_shift, all_selection, all_verified, all_screening, all_final = [], [], [], [], [], [], [], [], [], [], [], []
    for seed in active_seeds:
        eval_df, per_case, conv, transitions, training_summary, ranking_df, controllability_df, policy_shift_df, selection_df, verified_df, screening_df, final_df = run_seed(cfg, npz_path, int(seed), device)
        all_eval.append(eval_df)
        all_per_case.append(per_case)
        all_conv.append(conv)
        all_transitions.extend(transitions)
        all_training.append(training_summary)
        all_ranking.append(ranking_df)
        all_controllability.append(controllability_df)
        all_shift.append(policy_shift_df)
        all_selection.append(selection_df)
        all_verified.append(verified_df)
        all_screening.append(screening_df)
        all_final.append(final_df)
    per_case_df = pd.concat(all_per_case, ignore_index=True)
    conv_df = pd.concat(all_conv, ignore_index=True)
    training_df = pd.concat(all_training, ignore_index=True)
    ranking_df = pd.concat(all_ranking, ignore_index=True)
    controllability_df = pd.concat(all_controllability, ignore_index=True)
    policy_shift_df = pd.concat(all_shift, ignore_index=True)
    selection_df = pd.concat(all_selection, ignore_index=True)
    verified_df = pd.concat(all_verified, ignore_index=True)
    screening_df = pd.concat(all_screening, ignore_index=True)
    final_df = pd.concat(all_final, ignore_index=True)
    transition_df = pd.DataFrame([tr.__dict__ for tr in all_transitions])
    tables = {
        "main_results": summarize_main(per_case_df),
        "seed_results": seed_summary(per_case_df),
        "convergence": conv_df,
        "critic_ranking": ranking_df,
        "critic_long_horizon_ranking": ranking_df,
        "latent_controllability": controllability_df,
        "policy_shift": policy_shift_df,
        "policy_selection": selection_df,
        "verified_policy_improvement": verified_df,
        "critic_screening": screening_df.groupby(["Seed", "Cycle"], as_index=False).agg(
            PrecisionAtM=("PrecisionAtM", "mean"), Top1TruePositiveRate=("Top1TruePositiveRate", "mean"),
            BestFoundRegret=("BestFoundRegret", "mean"), CandidateReturnSpread=("CandidateReturnSpread", "mean"),
        ),
        "final_rl_contribution": final_df,
        "training_summary": training_df,
        "residual_accuracy_quadrants": residual_accuracy_quadrants(all_transitions),
        "physics_metrics": per_case_df[["Seed", "Method", "Steps", "Case", "PDE residual norm", "BC error", "IC error"]],
        "per_case_results": per_case_df,
        "transition_debug": transition_df,
    }
    tables["rl_contribution"] = tables["final_rl_contribution"]
    tables["critic_within_state_ranking"] = tables["critic_long_horizon_ranking"]
    tables["actual_rl_contribution"] = tables["rl_contribution"]
    table_dir = ROOT / "results" / "solver_v2" / "tables"
    for name in ["main_results", "seed_results", "convergence", "critic_ranking", "critic_long_horizon_ranking", "critic_within_state_ranking", "latent_controllability", "policy_shift", "policy_selection", "verified_policy_improvement", "critic_screening", "final_rl_contribution", "rl_contribution", "actual_rl_contribution", "training_summary", "residual_accuracy_quadrants", "physics_metrics", "per_case_results"]:
        tables[name].to_csv(table_dir / f"{name}.csv", index=False)
    generate_figures(tables)
    generate_representative_figure(cfg, npz_path, int(cfg["solver_v2"]["seeds"][0]), device)
    write_docs(tables)
    summary = {"mode": mode, "device": str(device), "seeds": active_seeds, "tables": sorted(p.name for p in table_dir.glob("*.csv"))}
    (table_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    step10 = tables["main_results"][tables["main_results"]["Steps"].eq(10)]
    print("Solver V2 latent paper pipeline completed.")
    print(step10[["Method", "Relative L2 mean", "Relative L2 std", "paired improvement mean"]].to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="paper", choices=["paper"])
    parser.add_argument("--stage", default="paper", choices=["paper", "oracle-headroom"])
    parser.add_argument("--oracle-horizon", type=int, default=None)
    parser.add_argument("--seeds", default=None, help="Optional comma-separated seed subset for a smoke run.")
    args = parser.parse_args()
    seeds = [int(value) for value in args.seeds.split(",")] if args.seeds else None
    if args.stage == "oracle-headroom":
        cfg = load_config()
        ensure_dirs()
        npz_path = prepare_reaction_diffusion(cfg, ROOT)
        active_seeds = seeds if seeds is not None else [int(seed) for seed in cfg["solver_v2"]["seeds"]]
        run_oracle_headroom_stage(cfg, npz_path, active_seeds, int(args.oracle_horizon or cfg["solver_v2"]["oracle_headroom_horizon"]), get_device(str(cfg.get("device", "cuda"))))
    else:
        run_pipeline(args.mode, seeds)


if __name__ == "__main__":
    main()
