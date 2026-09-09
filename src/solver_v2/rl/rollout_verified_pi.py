from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from src.solver_v2.models.operator_actor import DeterministicNeuralOperatorActor


def normalized_rms_shift(z: torch.Tensor, z_sup: torch.Tensor, latent_std: torch.Tensor) -> torch.Tensor:
    scale = latent_std.to(z.device).reshape(1, -1).clamp_min(1e-6)
    return torch.sqrt(torch.mean(((z - z_sup) / scale) ** 2, dim=-1, keepdim=True).clamp_min(1e-12))


def project_to_trust_region(z: torch.Tensor, z_sup: torch.Tensor, latent_std: torch.Tensor, radius: float) -> torch.Tensor:
    shift = normalized_rms_shift(z, z_sup, latent_std)
    factor = torch.clamp(float(radius) / shift, max=1.0)
    return z_sup + factor * (z - z_sup)


class TrustedPolicy(nn.Module):
    """Policy output constrained to a normalized latent trust region around pi_sup."""

    def __init__(self, actor: DeterministicNeuralOperatorActor, supervised_actor: DeterministicNeuralOperatorActor, latent_std: torch.Tensor, radius: float):
        super().__init__()
        self.actor = actor
        self.supervised_actor = supervised_actor
        self.register_buffer("latent_std", latent_std.detach().reshape(-1).float())
        self.radius = float(radius)
        self.supervised_actor.eval()
        for p in self.supervised_actor.parameters():
            p.requires_grad_(False)

    def forward(self, fields: torch.Tensor, scalars: torch.Tensor, stats: dict | None = None, temperature: float = 1.0) -> torch.Tensor:
        with torch.no_grad():
            z_sup = self.supervised_actor(fields, scalars, stats, temperature)
        z = self.actor(fields, scalars, stats, temperature)
        return project_to_trust_region(z, z_sup, self.latent_std, self.radius)


@dataclass
class VerifiedTarget:
    fields: torch.Tensor
    scalars: torch.Tensor
    target: torch.Tensor
    advantage: float


class _VerifiedTargetDataset(Dataset):
    def __init__(self, rows: list[VerifiedTarget]):
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.rows[idx]
        return {
            "fields": row.fields,
            "scalars": row.scalars,
            "target": row.target,
            "advantage": torch.tensor(row.advantage, dtype=torch.float32),
        }


def train_verified_policy(
    actor: DeterministicNeuralOperatorActor,
    supervised_actor: DeterministicNeuralOperatorActor,
    targets: list[VerifiedTarget],
    stats: dict,
    latent_std: torch.Tensor,
    radius: float,
    cfg: dict,
    device: torch.device,
) -> tuple[TrustedPolicy, list[dict[str, float]]]:
    """Fit only to PDE-verified improving targets; no critic gradient reaches the actor."""
    for p in actor.parameters():
        p.requires_grad_(True)
    trusted = TrustedPolicy(actor.to(device), supervised_actor.to(device), latent_std.to(device), radius).to(device)
    if not targets:
        return trusted.eval(), []
    dataset = _VerifiedTargetDataset(targets)
    loader = DataLoader(dataset, batch_size=int(cfg["solver_v2"]["verified_batch_size"]), shuffle=True)
    opt = torch.optim.AdamW(actor.parameters(), lr=float(cfg["solver_v2"]["verified_actor_lr"]), weight_decay=1e-4)
    history: list[dict[str, float]] = []
    mean_adv = max(1e-8, sum(row.advantage for row in targets) / len(targets))
    for epoch in tqdm(range(int(cfg["solver_v2"]["verified_actor_epochs"])), desc="solver_v2:verified_policy"):
        losses, fits, trusts = [], [], []
        trusted.train()
        for batch in loader:
            fields = batch["fields"].to(device)
            scalars = batch["scalars"].to(device)
            target = batch["target"].to(device)
            advantage = batch["advantage"].to(device)
            with torch.no_grad():
                z_sup = supervised_actor(fields, scalars, stats, float(cfg["solver_v2"]["temperature"]))
                target = project_to_trust_region(target, z_sup, latent_std, radius)
            pred = trusted(fields, scalars, stats, float(cfg["solver_v2"]["temperature"]))
            weights = 1.0 + advantage / mean_adv
            fit_loss = (weights * torch.mean((pred - target) ** 2, dim=-1)).mean()
            trust_loss = F.mse_loss(pred, z_sup)
            loss = fit_loss + float(cfg["solver_v2"]["verified_lambda_trust"]) * trust_loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
            fits.append(float(fit_loss.detach().cpu()))
            trusts.append(float(trust_loss.detach().cpu()))
        history.append({
            "epoch": float(epoch),
            "verified_loss": float(sum(losses) / max(1, len(losses))),
            "target_fit_loss": float(sum(fits) / max(1, len(fits))),
            "trust_loss": float(sum(trusts) / max(1, len(trusts))),
            "target_count": float(len(targets)),
            "trust_radius": float(radius),
        })
    return trusted.eval(), history
