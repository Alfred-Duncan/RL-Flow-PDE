from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import Dataset


@dataclass
class RawCorrectionTransition:
    case_id: int
    episode_id: int
    step_idx: int
    split: str
    gt: torch.Tensor
    state_fields: torch.Tensor
    state_scalars: torch.Tensor
    delta_u: torch.Tensor
    reward: float
    next_fields: torch.Tensor
    next_scalars: torch.Tensor
    done: float
    error_before: float
    error_after: float
    residual_before: float
    residual_after: float
    physics_before: float
    physics_after: float
    action_norm: float
    source_policy: str


@dataclass
class SolverTransition(RawCorrectionTransition):
    action: torch.Tensor
    mc_return: float = 0.0


class ReplayDataset(Dataset):
    def __init__(self, transitions: list[SolverTransition]):
        self.transitions = transitions

    def __len__(self) -> int:
        return len(self.transitions)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        tr = self.transitions[idx]
        return {
            "state_fields": tr.state_fields,
            "state_scalars": tr.state_scalars,
            "action": tr.action,
            "delta_u": tr.delta_u,
            "reward": torch.tensor(tr.reward, dtype=torch.float32),
            "mc_return": torch.tensor(tr.mc_return, dtype=torch.float32),
            "next_fields": tr.next_fields,
            "next_scalars": tr.next_scalars,
            "done": torch.tensor(tr.done, dtype=torch.float32),
        }


class ReplayBuffer:
    def __init__(self):
        self.transitions: list[SolverTransition] = []

    def extend(self, rows: list[SolverTransition]) -> None:
        self.transitions.extend(rows)

    def by_split(self, split: str) -> list[SolverTransition]:
        return [tr for tr in self.transitions if tr.split == split]

    def dataset(self, split: str = "train") -> ReplayDataset:
        return ReplayDataset(self.by_split(split))
