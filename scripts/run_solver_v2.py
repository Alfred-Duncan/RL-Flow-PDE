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
V2_CACHE_VERSION = "latent_operator_v2"

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
    ckpt = ROOT / "checkpoints" / "solver_v2" / f"fno_initializer_{V2_CACHE_VERSION}_seed{seed}.pt"
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
            loss = F.mse_loss(recon, delta) + 1e-4 * z.pow(2).mean()
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
    latent_std = stats.get("latent_std", torch.ones(int(cfg["solver_v2"]["latent_dim"]))).to(device)
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


def critic_ranking_validation(cfg: dict, val_raw: list[RawCorrectionTransition], actor: DeterministicNeuralOperatorActor, supervised_actor: DeterministicNeuralOperatorActor, critic: TwinOperatorCritic, decoder: CorrectionOperatorDecoder, pde: ReactionDiffusionPDE, stats: dict, device: torch.device, label: str) -> pd.DataFrame:
    rows = []
    qs: list[float] = []
    rs: list[float] = []
    pair_ok = pair_total = 0
    latent_std = stats.get("latent_std", torch.ones(int(cfg["solver_v2"]["latent_dim"]))).to(device)
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
            candidates = [("supervised", z_sup), ("actor", z_actor), ("zero", torch.zeros_like(z_sup))]
            for j in range(4):
                candidates.append((f"perturb{j}", z_sup + (0.06 * latent_std * torch.randn_like(z_sup))))
            local_q, local_r = [], []
            for name, z in candidates:
                delta = decoder(fields, scalars, z, stats, float(cfg["solver_v2"]["temperature"])).squeeze(0)
                next_u = pde.step(u, delta, case)
                rew = pde.reward(u, next_u, delta, case, float(cfg["solver_v2"]["lambda_action"]))
                qv = float(critic.q_min(fields, scalars, z).item())
                rv = float(rew["reward"])
                qs.append(qv)
                rs.append(rv)
                local_q.append(qv)
                local_r.append(rv)
                rows.append({"variant": label, "state": idx, "candidate": name, "q": qv, "return": rv})
            for a in range(len(local_q)):
                for b in range(a + 1, len(local_q)):
                    dr = local_r[a] - local_r[b]
                    dq = local_q[a] - local_q[b]
                    if abs(dr) > 1e-9:
                        pair_total += 1
                        pair_ok += int(np.sign(dr) == np.sign(dq))
    pearson, spearman = _corr_metrics(qs, rs)
    pairwise = float(pair_ok / max(1, pair_total))
    out = pd.DataFrame(rows + [{"variant": label, "state": -1, "candidate": "summary", "q": pearson, "return": spearman}])
    out["pearson"] = pearson
    out["spearman"] = spearman
    out["pairwise"] = pairwise
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
    latent_std = stats.get("latent_std", torch.ones(int(cfg["solver_v2"]["latent_dim"]))).to(device)
    actor.eval()
    supervised_actor.eval()
    decoder.eval()
    for idx, tr in enumerate(base_rows[: min(len(base_rows), 160)]):
        fields = tr.state_fields.unsqueeze(0).to(device)
        scalars = tr.state_scalars.unsqueeze(0).to(device)
        case, u = raw_case_from_transition(tr, stats, pde, device)
        with torch.no_grad():
            z_sup = supervised_actor(fields, scalars, stats, float(cfg["solver_v2"]["temperature"]))
            z_actor = actor(fields, scalars, stats, float(cfg["solver_v2"]["temperature"]))
            candidates = [("supervised", z_sup), ("actor", z_actor), ("zero", torch.zeros_like(z_sup))]
            for j in range(4):
                candidates.append((f"perturb{j}", z_sup + 0.08 * latent_std * torch.randn_like(z_sup)))
            for name, z in candidates:
                delta = decoder(fields, scalars, z, stats, float(cfg["solver_v2"]["temperature"])).squeeze(0)
                next_u = pde.step(u, delta, case)
                next_fields, next_scalars = state_from_u(pde, case, next_u, 1.0, stats)
                r = pde.reward(u, next_u, delta, case, float(cfg["solver_v2"]["lambda_action"]))
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
                        mc_return=r["reward"],
                    )
                )
    return rows


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


def train_td3_variant(cfg: dict, transitions: list[SolverTransition], val_raw: list[RawCorrectionTransition], actor: DeterministicNeuralOperatorActor, supervised_actor: DeterministicNeuralOperatorActor, decoder: CorrectionOperatorDecoder, seed: int, variant: str, stats: dict, pde: ReactionDiffusionPDE, device: torch.device, epochs_override: int | None = None) -> tuple[DeterministicNeuralOperatorActor, TwinOperatorCritic, list[dict[str, float]], pd.DataFrame]:
    local_cfg = deepcopy(cfg)
    if variant == "scratch":
        local_cfg["solver_v2"]["lambda_bc_start"] = 0.0
    if variant == "td3_bc":
        local_cfg["solver_v2"]["lambda_phys"] = 0.0
    if epochs_override is not None:
        local_cfg["solver_v2"]["td3_epochs"] = int(epochs_override)
    critic = TwinOperatorCritic(width=int(cfg["solver_v2"]["width"]), modes=int(cfg["solver_v2"]["modes"]), depth=int(cfg["solver_v2"]["critic_depth"]), latent_dim=int(cfg["solver_v2"]["latent_dim"]), state_dim=int(cfg["solver_v2"]["state_dim"])).to(device)
    trainer = TD3BCTrainer(actor, critic, decoder, local_cfg, stats, device, allow_actor_update=False, variant=variant)
    ckpt = ROOT / "checkpoints" / "solver_v2" / f"td3_{V2_CACHE_VERSION}_{variant}_seed{seed}.pt"
    if ckpt.exists():
        actor, critic, hist = trainer.fit(transitions, ckpt)
        rank = critic_ranking_validation(cfg, val_raw, actor, supervised_actor, critic, decoder, pde, stats, device, variant)
        return actor, critic, hist, rank
    critic_rows = transitions + critic_candidate_transitions(cfg, [tr for tr in transitions if tr.split == "train"], actor, supervised_actor, decoder, pde, stats, device, variant)
    trainer.pretrain_critic_mc(critic_rows)
    rank = critic_ranking_validation(cfg, val_raw, actor, supervised_actor, critic, decoder, pde, stats, device, variant)
    spearman = float(rank["spearman"].iloc[0])
    pairwise = float(rank["pairwise"].iloc[0])
    trainer.allow_actor_update = bool(spearman > 0.5 and pairwise > 0.65)
    actor, critic, hist = trainer.fit(transitions, ckpt)
    return actor, critic, hist, rank


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
    methods = ["Base FNO Initializer", "Supervised Latent Neural Operator Corrector", "TD3 from scratch", "TD3+BC", "Full RL Neural Operator Solver", "Gradient baseline", "PINN-style baseline"]
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


def build_training_summary(seed: int, histories: dict[str, list[dict[str, float]]], rankings: list[pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for variant, hist in histories.items():
        for h in hist:
            row = dict(h)
            row["Seed"] = seed
            row["variant"] = variant
            rows.append(row)
    for rank in rankings:
        summary = rank[rank["candidate"].eq("summary")].iloc[0]
        rows.append({"Seed": seed, "variant": str(summary["variant"]) + "_ranking", "epoch": -999.0, "critic_loss": np.nan, "actor_loss": np.nan, "actor_updates_enabled": float(summary["spearman"] > 0.5 and summary["pairwise"] > 0.65), "ranking_spearman": float(summary["spearman"]), "ranking_pairwise": float(summary["pairwise"])})
    return pd.DataFrame(rows)


def run_seed(cfg: dict, npz_path: Path, seed: int, device: torch.device) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[SolverTransition], pd.DataFrame, pd.DataFrame]:
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
    actor_full = deepcopy(actor_sup).to(device)
    best_actor = deepcopy(actor_sup).to(device)
    best_val = sup_val
    histories: dict[str, list[dict[str, float]]] = {"supervised": [{"epoch": 0.0, "critic_loss": np.nan, "actor_loss": np.nan, "val_error": sup_val}]}
    rankings: list[pd.DataFrame] = []
    critic_full: TwinOperatorCritic | None = None
    full_replay = list(replay)
    for cycle in range(int(s["online_cycles"])):
        full_replay.extend(rollout_actor_to_replay(cfg, npz_path, initializer, actor_full, decoder, pde, stats, seed, cycle, device))
        actor_full, critic_full, hist, rank = train_td3_variant(cfg, full_replay, val_raw, actor_full, actor_sup, decoder, seed, f"full_cycle{cycle}", stats, pde, device)
        histories[f"full_cycle{cycle}"] = hist
        rankings.append(rank)
        val_error = validation_solver_error(cfg, npz_path, initializer, actor_full, decoder, pde, stats, device)
        if val_error <= best_val:
            best_val = val_error
            best_actor = deepcopy(actor_full).to(device)
        else:
            actor_full = deepcopy(best_actor).to(device)
    torch.save({"actor": best_actor.state_dict(), "validation_error": best_val}, ROOT / "checkpoints" / "solver_v2" / f"actor_selected_{V2_CACHE_VERSION}_seed{seed}.pt")
    actor_td3bc, critic_td3bc, hist_td3bc, rank_td3bc = train_td3_variant(cfg, replay, val_raw, deepcopy(actor_sup).to(device), actor_sup, decoder, seed, "td3_bc", stats, pde, device)
    actor_scratch = DeterministicNeuralOperatorActor(width=int(s["width"]), modes=int(s["modes"]), depth=int(s["actor_depth"]), latent_dim=int(s["latent_dim"]), state_dim=int(s["state_dim"])).to(device)
    actor_scratch, critic_scratch, hist_scratch, rank_scratch = train_td3_variant(cfg, replay, val_raw, actor_scratch, actor_sup, decoder, seed, "scratch", stats, pde, device, int(s["td3_scratch_epochs"]))
    histories["td3_bc"] = hist_td3bc
    histories["scratch"] = hist_scratch
    rankings.extend([rank_td3bc, rank_scratch])
    actors = {"Supervised Latent Neural Operator Corrector": actor_sup, "TD3 from scratch": actor_scratch, "TD3+BC": actor_td3bc, "Full RL Neural Operator Solver": best_actor}
    critics = {"TD3 from scratch": critic_scratch, "TD3+BC": critic_td3bc, "Full RL Neural Operator Solver": critic_full}
    eval_df, per_case, conv = evaluate_methods(cfg, npz_path, seed, initializer, actors, critics, decoder, pde, stats, device)
    training_summary = build_training_summary(seed, histories, rankings)
    ranking_df = pd.concat(rankings, ignore_index=True)
    ranking_df["Seed"] = seed
    return eval_df, per_case, conv, full_replay, training_summary, ranking_df


def generate_figures(tables: dict[str, pd.DataFrame]) -> None:
    fig_dir = ROOT / "results" / "solver_v2" / "figures"
    main = tables["per_case_results"]
    conv = tables["convergence"]
    train = tables["training_summary"]
    trans = tables["transition_debug"]
    rank = tables["critic_ranking"]
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
            if method in ["Gradient baseline", "Supervised Latent Neural Operator Corrector", "Full RL Neural Operator Solver"]:
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
    paired = main[(main["Steps"].eq(10)) & (main["Method"].isin(["Supervised Latent Neural Operator Corrector", "Full RL Neural Operator Solver"]))]
    pivot = paired.pivot_table(index=["Seed", "Case"], columns="Method", values="paired improvement")
    plt.figure(figsize=(5, 4))
    if {"Supervised Latent Neural Operator Corrector", "Full RL Neural Operator Solver"}.issubset(pivot.columns):
        plt.scatter(pivot["Supervised Latent Neural Operator Corrector"], pivot["Full RL Neural Operator Solver"], s=18, alpha=0.7)
        lo = float(np.nanmin(pivot.values))
        hi = float(np.nanmax(pivot.values))
        plt.plot([lo, hi], [lo, hi], color="black", linewidth=0.8)
    plt.xlabel("Supervised paired improvement")
    plt.ylabel("Full RL paired improvement")
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
    plt.ylabel("Actual one-step return")
    plt.tight_layout()
    plt.savefig(fig_dir / "figure9_critic_ranking.png", dpi=180)
    plt.close()


def generate_representative_figure(cfg: dict, npz_path: Path, seed: int, device: torch.device) -> None:
    fig_dir = ROOT / "results" / "solver_v2" / "figures"
    pde = ReactionDiffusionPDE(float(cfg["benchmark"]["diffusion"]), float(cfg["benchmark"]["reaction"]))
    stats = fit_train_stats(cfg, npz_path, pde, device, ROOT / "checkpoints" / "solver_v2")
    initializer = FNOInitializer(width=int(cfg["solver_v2"]["width"]), modes=int(cfg["solver_v2"]["modes"]), depth=3).to(device)
    initializer.load_state_dict(torch.load(ROOT / "checkpoints" / "solver_v2" / f"fno_initializer_{V2_CACHE_VERSION}_seed{seed}.pt", map_location=device)["model"])
    ae = torch.load(ROOT / "checkpoints" / "solver_v2" / f"correction_autoencoder_{V2_CACHE_VERSION}_seed{seed}.pt", map_location=device)
    stats.update(ae.get("latent_stats", {}))
    decoder = CorrectionOperatorDecoder(latent_dim=int(cfg["solver_v2"]["latent_dim"]), width=int(cfg["solver_v2"]["width"]), modes=int(cfg["solver_v2"]["modes"]), depth=int(cfg["solver_v2"]["actor_depth"])).to(device)
    decoder.load_state_dict(ae["decoder"])
    actor = DeterministicNeuralOperatorActor(width=int(cfg["solver_v2"]["width"]), modes=int(cfg["solver_v2"]["modes"]), depth=int(cfg["solver_v2"]["actor_depth"]), latent_dim=int(cfg["solver_v2"]["latent_dim"]), state_dim=int(cfg["solver_v2"]["state_dim"])).to(device)
    selected = ROOT / "checkpoints" / "solver_v2" / f"actor_selected_{V2_CACHE_VERSION}_seed{seed}.pt"
    fallback = ROOT / "checkpoints" / "solver_v2" / f"actor_pretrain_{V2_CACHE_VERSION}_seed{seed}.pt"
    actor.load_state_dict(torch.load(selected if selected.exists() else fallback, map_location=device)["actor"])
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
    sub = main[main["Steps"].eq(10)]

    def metric(method: str) -> float:
        return float(sub[sub["Method"].eq(method)]["Relative L2 mean"].iloc[0])

    base = metric("Base FNO Initializer")
    sup = metric("Supervised Latent Neural Operator Corrector")
    rl = metric("Full RL Neural Operator Solver")
    rank_summary = tables["critic_ranking"][tables["critic_ranking"]["candidate"].eq("summary")]
    spearman = float(rank_summary["spearman"].mean()) if not rank_summary.empty else float("nan")
    pairwise = float(rank_summary["pairwise"].mean()) if not rank_summary.empty else float("nan")
    if rl < base and rl < sup and spearman > 0.5 and pairwise > 0.65:
        claim = "The latent-action RL neural-operator solver improves over both the corrected FNO initializer and the supervised latent corrector in this run."
    elif rl < base:
        claim = "The latent-action solver improves over the corrected FNO initializer, but the RL stage does not clearly beat the supervised latent corrector."
    else:
        claim = "The current latent-action RL stage is not yet a reliable improvement over the corrected FNO initializer."
    text = (
        "# Solver V2 Results\n\n"
        f"{claim}\n\n"
        f"- 10-step Base FNO Relative L2 mean: {base:.6f}\n"
        f"- 10-step Supervised Latent Corrector Relative L2 mean: {sup:.6f}\n"
        f"- 10-step Full RL Solver Relative L2 mean: {rl:.6f}\n"
        f"- Mean critic Spearman on validation candidate ranking: {spearman:.3f}\n"
        f"- Mean critic pairwise ranking accuracy: {pairwise:.3f}\n\n"
        "The architecture used here is: corrected FNO initializer, residual-conditioned neural-operator state encoder, 32D latent action actor, frozen neural-operator correction decoder, hard IC/BC projection, Twin-Q critic over `(state,z)`, MC return critic pretraining, validation ranking gate, and conservative TD3+BC.\n\n"
        "## Main Table\n\n"
        + main.to_markdown(index=False)
        + "\n"
    )
    (ROOT / "docs" / "solver_v2_results.md").write_text(text, encoding="utf-8")
    (ROOT / "docs" / "solver_v2_claims.md").write_text("# Solver V2 Claims\n\n" + claim + "\n", encoding="utf-8")


def run_pipeline(mode: str) -> None:
    cfg = load_config()
    ensure_dirs()
    npz_path = prepare_reaction_diffusion(cfg, ROOT)
    device = get_device(str(cfg.get("device", "cuda")))
    all_eval, all_per_case, all_conv, all_transitions, all_training, all_ranking = [], [], [], [], [], []
    for seed in cfg["solver_v2"]["seeds"]:
        eval_df, per_case, conv, transitions, training_summary, ranking_df = run_seed(cfg, npz_path, int(seed), device)
        all_eval.append(eval_df)
        all_per_case.append(per_case)
        all_conv.append(conv)
        all_transitions.extend(transitions)
        all_training.append(training_summary)
        all_ranking.append(ranking_df)
    per_case_df = pd.concat(all_per_case, ignore_index=True)
    conv_df = pd.concat(all_conv, ignore_index=True)
    training_df = pd.concat(all_training, ignore_index=True)
    ranking_df = pd.concat(all_ranking, ignore_index=True)
    transition_df = pd.DataFrame([tr.__dict__ for tr in all_transitions])
    tables = {
        "main_results": summarize_main(per_case_df),
        "seed_results": seed_summary(per_case_df),
        "convergence": conv_df,
        "critic_ranking": ranking_df,
        "training_summary": training_df,
        "residual_accuracy_quadrants": residual_accuracy_quadrants(all_transitions),
        "physics_metrics": per_case_df[["Seed", "Method", "Steps", "Case", "PDE residual norm", "BC error", "IC error"]],
        "per_case_results": per_case_df,
        "transition_debug": transition_df,
    }
    table_dir = ROOT / "results" / "solver_v2" / "tables"
    for name in ["main_results", "seed_results", "convergence", "critic_ranking", "training_summary", "residual_accuracy_quadrants", "physics_metrics", "per_case_results"]:
        tables[name].to_csv(table_dir / f"{name}.csv", index=False)
    generate_figures(tables)
    generate_representative_figure(cfg, npz_path, int(cfg["solver_v2"]["seeds"][0]), device)
    write_docs(tables)
    summary = {"mode": mode, "device": str(device), "seeds": cfg["solver_v2"]["seeds"], "tables": sorted(p.name for p in table_dir.glob("*.csv"))}
    (table_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    step10 = tables["main_results"][tables["main_results"]["Steps"].eq(10)]
    print("Solver V2 latent paper pipeline completed.")
    print(step10[["Method", "Relative L2 mean", "Relative L2 std", "paired improvement mean"]].to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="paper", choices=["paper"])
    args = parser.parse_args()
    run_pipeline(args.mode)


if __name__ == "__main__":
    main()
