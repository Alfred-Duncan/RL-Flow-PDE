from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.solver_v2.data.replay_buffer import GroupedReplayDataset, ReplayDataset, SolverTransition
from src.solver_v2.models.operator_actor import DeterministicNeuralOperatorActor
from src.solver_v2.models.correction_autoencoder import CorrectionOperatorDecoder
from src.solver_v2.models.operator_critic import TwinOperatorCritic


def soft_update(target: torch.nn.Module, source: torch.nn.Module, tau: float) -> None:
    with torch.no_grad():
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.data.mul_(1.0 - tau).add_(sp.data, alpha=tau)


def batch_residual(u_next_norm: torch.Tensor, source_norm: torch.Tensor, stats: dict, diffusion: float, reaction: float) -> torch.Tensor:
    u = u_next_norm * stats["u_std"].to(u_next_norm.device) + stats["u_mean"].to(u_next_norm.device)
    source = source_norm * stats["source_std"].to(source_norm.device) + stats["source_mean"].to(source_norm.device)
    x = stats["x"].to(u.device).reshape(-1)
    t = stats["t"].to(u.device).reshape(-1)
    dx = (x[1] - x[0]).abs().clamp_min(1e-6)
    dt = (t[1] - t[0]).abs().clamp_min(1e-6)
    ut = torch.zeros_like(u)
    ut[:, :, 1:-1] = (u[:, :, 2:] - u[:, :, :-2]) / (2.0 * dt)
    ut[:, :, 0] = (u[:, :, 1] - u[:, :, 0]) / dt
    ut[:, :, -1] = (u[:, :, -1] - u[:, :, -2]) / dt
    uxx = torch.zeros_like(u)
    uxx[:, 1:-1, :] = (u[:, 2:, :] - 2.0 * u[:, 1:-1, :] + u[:, :-2, :]) / (dx * dx)
    return ut - float(diffusion) * uxx - float(reaction) * (u ** 2) - source


class TD3BCTrainer:
    def __init__(
        self,
        actor: DeterministicNeuralOperatorActor,
        critic: TwinOperatorCritic,
        decoder: CorrectionOperatorDecoder,
        cfg: dict,
        stats: dict,
        device: torch.device,
        supervised_actor: DeterministicNeuralOperatorActor | None = None,
        allow_actor_update: bool = True,
        variant: str = "td3_bc",
    ):
        self.actor = actor.to(device)
        for p in self.actor.parameters():
            p.requires_grad_(True)
        self.supervised_actor = supervised_actor.to(device).eval() if supervised_actor is not None else None
        if self.supervised_actor is not None:
            for p in self.supervised_actor.parameters():
                p.requires_grad_(False)
        self.critic = critic.to(device)
        self.decoder = decoder.to(device).eval()
        for p in self.decoder.parameters():
            p.requires_grad_(False)
        self.actor_target = deepcopy(actor).to(device).eval()
        self.critic_target = deepcopy(critic).to(device).eval()
        self.cfg = cfg
        self.stats = stats
        self.device = device
        self.actor_opt = torch.optim.AdamW(self.actor.parameters(), lr=float(cfg["solver_v2"]["td3_actor_lr"]), weight_decay=1e-4)
        self.critic_opt = torch.optim.AdamW(self.critic.parameters(), lr=float(cfg["solver_v2"]["td3_critic_lr"]), weight_decay=1e-4)
        self.total_updates = 0
        self.actor_updates = 0
        self.history: list[dict[str, float]] = []
        self.allow_actor_update = bool(allow_actor_update)
        self.variant = variant

    def _group_critic_loss(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
        """Return, centered advantage, and within-state ranking objectives."""
        fields = batch["state_fields"].to(self.device)
        scalars = batch["state_scalars"].to(self.device)
        actions = batch["action"].to(self.device)
        returns = batch["mc_return"].to(self.device)
        batch_size, candidates = returns.shape
        flat_fields = fields.reshape(batch_size * candidates, *fields.shape[2:])
        flat_scalars = scalars.reshape(batch_size * candidates, -1)
        flat_actions = actions.reshape(batch_size * candidates, -1)
        q1, q2 = self.critic(flat_fields, flat_scalars, flat_actions)
        q1, q2 = q1.view(batch_size, candidates), q2.view(batch_size, candidates)

        return_loss = F.smooth_l1_loss(q1, returns) + F.smooth_l1_loss(q2, returns)
        target_advantage = returns - returns.mean(dim=1, keepdim=True)
        advantage_loss = (
            F.smooth_l1_loss(q1 - q1.mean(dim=1, keepdim=True), target_advantage)
            + F.smooth_l1_loss(q2 - q2.mean(dim=1, keepdim=True), target_advantage)
        )

        delta_return = returns.unsqueeze(2) - returns.unsqueeze(1)
        pair_mask = torch.triu(torch.ones(candidates, candidates, device=self.device, dtype=torch.bool), diagonal=1)
        pair_mask = pair_mask.unsqueeze(0) & (delta_return.abs() >= float(self.cfg["solver_v2"]["critic_return_margin"]))
        if pair_mask.any():
            sign = delta_return.sign()
            ranks = torch.empty_like(returns, dtype=torch.long)
            order = returns.argsort(dim=1, descending=True)
            rank_values = torch.arange(candidates, device=self.device).view(1, -1).expand(batch_size, -1)
            ranks.scatter_(1, order, rank_values)
            quartile = max(1, candidates // 4)
            hard_pair = ((ranks.unsqueeze(2) < quartile) & (ranks.unsqueeze(1) >= candidates - quartile)) | ((ranks.unsqueeze(1) < quartile) & (ranks.unsqueeze(2) >= candidates - quartile))
            weights = torch.where(hard_pair, float(self.cfg["solver_v2"]["critic_hard_pair_weight"]), 1.0)
            def ranking_loss(q: torch.Tensor) -> torch.Tensor:
                delta_q = q.unsqueeze(2) - q.unsqueeze(1)
                values = F.softplus(-sign * delta_q / float(self.cfg["solver_v2"]["critic_rank_temperature"]))
                return (values[pair_mask] * weights[pair_mask]).sum() / weights[pair_mask].sum().clamp_min(1.0)
            rank_loss = ranking_loss(q1) + ranking_loss(q2)
        else:
            rank_loss = torch.zeros((), device=self.device)

        total = (
            float(self.cfg["solver_v2"]["lambda_critic_return"]) * return_loss
            + float(self.cfg["solver_v2"]["lambda_critic_advantage"]) * advantage_loss
            + float(self.cfg["solver_v2"]["lambda_critic_rank"]) * rank_loss
        )
        return total, {
            "return_loss": float(return_loss.detach().cpu()),
            "advantage_loss": float(advantage_loss.detach().cpu()),
            "rank_loss": float(rank_loss.detach().cpu()),
        }

    def _latent_noise(self, action: torch.Tensor) -> torch.Tensor:
        latent_std = self.stats.get("latent_std")
        if latent_std is None:
            scale = torch.ones(action.shape[-1], device=action.device)
        else:
            scale = latent_std.to(action.device)
        noise_std = float(self.cfg["solver_v2"]["target_noise"]) * scale
        noise_clip = float(self.cfg["solver_v2"]["target_noise_clip"]) * scale
        return (torch.randn_like(action) * noise_std).clamp(-noise_clip, noise_clip)

    def pretrain_critic_mc(self, transitions: list[SolverTransition], epochs: int | None = None) -> list[dict[str, float]]:
        grouped = GroupedReplayDataset([tr for tr in transitions if tr.split == "train"])
        if not len(grouped):
            raise ValueError("Critic pretraining requires grouped state-action candidate returns.")
        loader = DataLoader(grouped, batch_size=int(self.cfg["solver_v2"]["critic_group_batch_size"]), shuffle=True, drop_last=False)
        mc_epochs = int(epochs if epochs is not None else self.cfg["solver_v2"].get("critic_mc_epochs", 60))
        for ep in tqdm(range(mc_epochs), desc=f"solver_v2:critic_mc:{self.variant}"):
            losses = []
            components = {"return_loss": [], "advantage_loss": [], "rank_loss": []}
            for batch in loader:
                loss, values = self._group_critic_loss(batch)
                self.critic_opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
                self.critic_opt.step()
                losses.append(float(loss.detach().cpu()))
                for key, value in values.items():
                    components[key].append(value)
            if losses:
                self.history.append({
                    "epoch": float(-mc_epochs + ep),
                    "critic_loss": float(sum(losses) / len(losses)),
                    "actor_loss": 0.0,
                    "actor_updates_enabled": float(self.allow_actor_update),
                    **{key: float(sum(values) / len(values)) for key, values in components.items() if values},
                })
        self.critic_target.load_state_dict(self.critic.state_dict())
        return self.history

    def fit(self, transitions: list[SolverTransition], grouped_transitions: list[SolverTransition], ckpt: Path | None = None) -> tuple[DeterministicNeuralOperatorActor, TwinOperatorCritic, list[dict[str, float]]]:
        if ckpt is not None and ckpt.exists():
            payload = torch.load(ckpt, map_location=self.device)
            self.actor.load_state_dict(payload["actor"])
            self.critic.load_state_dict(payload["critic"])
            self.history = payload.get("history", [])
            self.actor_updates = int(payload.get("actor_updates", 0))
            return self.actor.eval(), self.critic.eval(), self.history
        ds = ReplayDataset([tr for tr in transitions if tr.split == "train"])
        loader = DataLoader(ds, batch_size=int(self.cfg["solver_v2"]["batch_size"]), shuffle=True, drop_last=True)
        grouped = GroupedReplayDataset(grouped_transitions)
        group_loader = DataLoader(grouped, batch_size=int(self.cfg["solver_v2"]["critic_group_batch_size"]), shuffle=True, drop_last=False)
        if not len(grouped):
            raise ValueError("TD3+BC requires grouped candidate transitions for critic updates.")
        tau = float(self.cfg["solver_v2"]["tau"])
        policy_delay = int(self.cfg["solver_v2"]["policy_delay"])
        lambda_q = float(self.cfg["solver_v2"]["lambda_q"])
        lambda_phys = float(self.cfg["solver_v2"]["lambda_phys"])
        lambda_step = float(self.cfg["solver_v2"]["lambda_step"])
        lambda_bc = float(self.cfg["solver_v2"]["lambda_bc_start"])
        epochs = int(self.cfg["solver_v2"]["td3_epochs"])
        for ep in tqdm(range(epochs), desc="solver_v2:td3_bc"):
            group_iter = iter(group_loader)
            for batch in loader:
                fields = batch["state_fields"].to(self.device)
                scalars = batch["state_scalars"].to(self.device)
                action = batch["action"].to(self.device)
                y = batch["mc_return"].to(self.device)
                q1, q2 = self.critic(fields, scalars, action)
                absolute_loss = F.smooth_l1_loss(q1, y) + F.smooth_l1_loss(q2, y)
                try:
                    group_batch = next(group_iter)
                except StopIteration:
                    group_iter = iter(group_loader)
                    group_batch = next(group_iter)
                grouped_loss, grouped_values = self._group_critic_loss(group_batch)
                critic_loss = absolute_loss + grouped_loss
                self.critic_opt.zero_grad(set_to_none=True)
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
                self.critic_opt.step()
                actor_loss_value = torch.tensor(0.0, device=self.device)
                if self.allow_actor_update and self.total_updates % policy_delay == 0:
                    pred_action = self.actor(fields, scalars, self.stats, float(self.cfg["solver_v2"]["temperature"]))
                    q_actor = self.critic.q1(fields, scalars, pred_action)
                    pred_delta = self.decoder(fields, scalars, pred_action, self.stats, float(self.cfg["solver_v2"]["temperature"]))
                    u_next_norm = fields[:, 0] + pred_delta / self.stats["u_std"].to(self.device).clamp_min(1e-6)
                    residual = batch_residual(u_next_norm, fields[:, 2], self.stats, self.cfg["benchmark"]["diffusion"], self.cfg["benchmark"]["reaction"])
                    phys_loss = torch.mean((residual / self.stats["residual_rms"].to(self.device).clamp_min(1e-6)) ** 2)
                    step_loss = torch.mean(pred_delta ** 2)
                    # Replay actions include online exploration noise.  The BC prior is the
                    # deterministic supervised latent policy, never a stored noisy action.
                    if self.supervised_actor is None:
                        raise RuntimeError("TD3+BC requires a frozen supervised actor anchor.")
                    with torch.no_grad():
                        z_sup = self.supervised_actor(fields, scalars, self.stats, float(self.cfg["solver_v2"]["temperature"]))
                    bc_loss = F.mse_loss(pred_action, z_sup)
                    actor_loss = -lambda_q * q_actor.mean() + lambda_bc * bc_loss + lambda_phys * phys_loss + lambda_step * step_loss
                    self.actor_opt.zero_grad(set_to_none=True)
                    actor_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
                    self.actor_opt.step()
                    self.actor_updates += 1
                    soft_update(self.actor_target, self.actor, tau)
                    soft_update(self.critic_target, self.critic, tau)
                    actor_loss_value = actor_loss.detach()
                if self.total_updates % max(1, len(loader)) == 0:
                    self.history.append({
                        "epoch": float(ep),
                        "critic_loss": float(critic_loss.detach().cpu()),
                        "actor_loss": float(actor_loss_value.detach().cpu()),
                        "actor_updates_enabled": float(self.allow_actor_update),
                        "actor_updates": float(self.actor_updates),
                        "return_loss": grouped_values["return_loss"],
                        "advantage_loss": grouped_values["advantage_loss"],
                        "rank_loss": grouped_values["rank_loss"],
                    })
                self.total_updates += 1
        if ckpt is not None:
            torch.save({"actor": self.actor.state_dict(), "critic": self.critic.state_dict(), "history": self.history, "actor_updates": self.actor_updates}, ckpt)
        return self.actor.eval(), self.critic.eval(), self.history
