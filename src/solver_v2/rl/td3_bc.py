from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.solver_v2.data.replay_buffer import ReplayDataset, SolverTransition
from src.solver_v2.models.operator_actor import DeterministicNeuralOperatorActor
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
    def __init__(self, actor: DeterministicNeuralOperatorActor, critic: TwinOperatorCritic, cfg: dict, stats: dict, device: torch.device):
        self.actor = actor.to(device)
        self.critic = critic.to(device)
        self.actor_target = deepcopy(actor).to(device).eval()
        self.critic_target = deepcopy(critic).to(device).eval()
        self.cfg = cfg
        self.stats = stats
        self.device = device
        self.actor_opt = torch.optim.AdamW(self.actor.parameters(), lr=float(cfg["solver_v2"]["td3_actor_lr"]), weight_decay=1e-4)
        self.critic_opt = torch.optim.AdamW(self.critic.parameters(), lr=float(cfg["solver_v2"]["td3_critic_lr"]), weight_decay=1e-4)
        self.total_updates = 0
        self.history: list[dict[str, float]] = []

    def fit(self, transitions: list[SolverTransition], ckpt: Path | None = None) -> tuple[DeterministicNeuralOperatorActor, TwinOperatorCritic, list[dict[str, float]]]:
        if ckpt is not None and ckpt.exists():
            payload = torch.load(ckpt, map_location=self.device)
            self.actor.load_state_dict(payload["actor"])
            self.critic.load_state_dict(payload["critic"])
            self.history = payload.get("history", [])
            return self.actor.eval(), self.critic.eval(), self.history
        ds = ReplayDataset([tr for tr in transitions if tr.split == "train"])
        loader = DataLoader(ds, batch_size=int(self.cfg["solver_v2"]["batch_size"]), shuffle=True, drop_last=True)
        gamma = float(self.cfg["solver_v2"]["gamma"])
        tau = float(self.cfg["solver_v2"]["tau"])
        policy_delay = int(self.cfg["solver_v2"]["policy_delay"])
        noise_std = float(self.cfg["solver_v2"]["target_noise"])
        noise_clip = float(self.cfg["solver_v2"]["target_noise_clip"])
        lambda_q = float(self.cfg["solver_v2"]["lambda_q"])
        lambda_phys = float(self.cfg["solver_v2"]["lambda_phys"])
        lambda_step = float(self.cfg["solver_v2"]["lambda_step"])
        bc_start = float(self.cfg["solver_v2"]["lambda_bc_start"])
        bc_end = float(self.cfg["solver_v2"]["lambda_bc_end"])
        epochs = int(self.cfg["solver_v2"]["td3_epochs"])
        total_batches = max(1, epochs * len(loader))
        for ep in tqdm(range(epochs), desc="solver_v2:td3_bc"):
            for batch in loader:
                fields = batch["state_fields"].to(self.device)
                scalars = batch["state_scalars"].to(self.device)
                action = batch["action"].to(self.device)
                reward = batch["reward"].to(self.device)
                next_fields = batch["next_fields"].to(self.device)
                next_scalars = batch["next_scalars"].to(self.device)
                done = batch["done"].to(self.device)
                with torch.no_grad():
                    next_action = self.actor_target(next_fields, next_scalars, self.stats, float(self.cfg["solver_v2"]["temperature"]))
                    noise = torch.randn_like(next_action).mul(noise_std).clamp(-noise_clip, noise_clip)
                    next_action = (next_action + noise).clamp(-float(self.stats["step_bound"]), float(self.stats["step_bound"]))
                    q1_t, q2_t = self.critic_target(next_fields, next_scalars, next_action)
                    y = reward + gamma * (1.0 - done) * torch.minimum(q1_t, q2_t)
                q1, q2 = self.critic(fields, scalars, action)
                critic_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)
                self.critic_opt.zero_grad(set_to_none=True)
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
                self.critic_opt.step()
                actor_loss_value = torch.tensor(0.0, device=self.device)
                if self.total_updates % policy_delay == 0:
                    pred_action = self.actor(fields, scalars, self.stats, float(self.cfg["solver_v2"]["temperature"]))
                    q_actor = self.critic.q1(fields, scalars, pred_action)
                    progress = min(1.0, self.total_updates / total_batches)
                    lambda_bc = bc_start * (1.0 - progress) + bc_end * progress
                    u_next_norm = fields[:, 0] + pred_action / self.stats["u_std"].to(self.device).clamp_min(1e-6)
                    residual = batch_residual(u_next_norm, fields[:, 2], self.stats, self.cfg["benchmark"]["diffusion"], self.cfg["benchmark"]["reaction"])
                    phys_loss = torch.mean((residual / self.stats["residual_rms"].to(self.device).clamp_min(1e-6)) ** 2)
                    step_loss = torch.mean(pred_action ** 2)
                    bc_loss = F.mse_loss(pred_action, action)
                    actor_loss = -lambda_q * q_actor.mean() + lambda_bc * bc_loss + lambda_phys * phys_loss + lambda_step * step_loss
                    self.actor_opt.zero_grad(set_to_none=True)
                    actor_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
                    self.actor_opt.step()
                    soft_update(self.actor_target, self.actor, tau)
                    soft_update(self.critic_target, self.critic, tau)
                    actor_loss_value = actor_loss.detach()
                if self.total_updates % max(1, len(loader)) == 0:
                    self.history.append({"epoch": float(ep), "critic_loss": float(critic_loss.detach().cpu()), "actor_loss": float(actor_loss_value.detach().cpu())})
                self.total_updates += 1
        if ckpt is not None:
            torch.save({"actor": self.actor.state_dict(), "critic": self.critic.state_dict(), "history": self.history}, ckpt)
        return self.actor.eval(), self.critic.eval(), self.history
