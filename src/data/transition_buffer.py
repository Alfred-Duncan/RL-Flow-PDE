from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import Dataset


@dataclass
class Transition:
    case_id: int
    split: str
    kind: str
    trajectory_id: int
    step_idx: int
    state_fields: torch.Tensor
    state_scalars: torch.Tensor
    action: torch.Tensor
    reward: float
    discounted_return: float
    next_fields: torch.Tensor
    next_scalars: torch.Tensor
    done: float
    gt_improvement: float
    physics_improvement: float
    error_before: float
    error_after: float
    energy_before: float
    energy_after: float


class TransitionDataset(Dataset):
    def __init__(self, transitions: list[Transition]):
        self.transitions = transitions

    def __len__(self) -> int:
        return len(self.transitions)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        tr = self.transitions[idx]
        return {
            "state_fields": tr.state_fields,
            "state_scalars": tr.state_scalars,
            "action": tr.action,
            "reward": torch.tensor(tr.reward, dtype=torch.float32),
            "discounted_return": torch.tensor(tr.discounted_return, dtype=torch.float32),
            "next_fields": tr.next_fields,
            "next_scalars": tr.next_scalars,
            "done": torch.tensor(tr.done, dtype=torch.float32),
        }


def split_transitions(transitions: list[Transition], split: str) -> list[Transition]:
    return [tr for tr in transitions if tr.split == split]
