from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TwinQCritic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden: int = 96):
        super().__init__()
        self.action_encoder = nn.Sequential(nn.Linear(action_dim, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU())
        self.q1 = nn.Sequential(nn.Linear(state_dim + hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))
        self.q2 = nn.Sequential(nn.Linear(state_dim + hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        a = self.action_encoder(action)
        x = torch.cat([state, a], dim=1)
        return self.q1(x).squeeze(-1), self.q2(x).squeeze(-1)

    def q_min(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        q1, q2 = self.forward(state, action)
        return torch.minimum(q1, q2)


class SingleQCritic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden: int = 96):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_dim + action_dim, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        q = self.net(torch.cat([state, action], dim=1)).squeeze(-1)
        return q, q

    def q_min(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.forward(state, action)[0]


class ValueNet(nn.Module):
    def __init__(self, state_dim: int, hidden: int = 96):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_dim, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state).squeeze(-1)


def expectile_loss(diff: torch.Tensor, expectile: float) -> torch.Tensor:
    weight = torch.where(diff < 0, 1.0 - expectile, expectile)
    return (weight * diff.pow(2)).mean()


def iql_losses(critic: nn.Module, value: ValueNet, state: torch.Tensor, action: torch.Tensor, reward: torch.Tensor, next_state: torch.Tensor, done: torch.Tensor, gamma: float, expectile: float) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        target = reward + gamma * (1.0 - done) * value(next_state)
    q1, q2 = critic(state, action)
    q_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
    with torch.no_grad():
        q_min = torch.minimum(q1, q2)
    v = value(state)
    v_loss = expectile_loss(q_min - v, expectile)
    return q_loss, v_loss
