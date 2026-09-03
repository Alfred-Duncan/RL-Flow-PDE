from __future__ import annotations

import torch
import torch.nn as nn


class FlowVelocity(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden: int = 96):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim + 1, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, action_dim),
        )
        self.action_dim = int(action_dim)

    def forward(self, state: torch.Tensor, a_tau: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        if tau.ndim == 1:
            tau = tau[:, None]
        return self.net(torch.cat([state, a_tau, tau], dim=1))


class FlowPolicy(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden: int = 96, steps: int = 8):
        super().__init__()
        self.velocity = FlowVelocity(state_dim, action_dim, hidden)
        self.action_dim = int(action_dim)
        self.steps = int(steps)

    def fm_loss(self, state: torch.Tensor, target_action: torch.Tensor, weight: torch.Tensor | None = None) -> torch.Tensor:
        a0 = torch.randn_like(target_action)
        tau = torch.rand(target_action.shape[0], device=target_action.device)
        a_tau = (1.0 - tau[:, None]) * a0 + tau[:, None] * target_action
        target_v = target_action - a0
        pred_v = self.velocity(state, a_tau, tau)
        loss = (pred_v - target_v).pow(2).mean(dim=1)
        if weight is not None:
            loss = loss * weight.detach()
        return loss.mean()

    @torch.no_grad()
    def sample(self, state: torch.Tensor, n_candidates: int = 1) -> torch.Tensor:
        b = state.shape[0]
        state_rep = state.repeat_interleave(n_candidates, dim=0)
        a = torch.randn(b * n_candidates, self.action_dim, device=state.device)
        dt = 1.0 / max(1, self.steps)
        for i in range(self.steps):
            tau = torch.full((a.shape[0],), (i + 0.5) * dt, device=state.device)
            a = a + dt * self.velocity(state_rep, a, tau)
        return a.view(b, n_candidates, self.action_dim)
