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
    group_id: str = ""
    candidate_name: str = ""


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
            "action": tr.action.reshape(-1),
            "delta_u": tr.delta_u,
            "reward": torch.tensor(tr.reward, dtype=torch.float32),
            "mc_return": torch.tensor(tr.mc_return, dtype=torch.float32),
            "next_fields": tr.next_fields,
            "next_scalars": tr.next_scalars,
            "done": torch.tensor(tr.done, dtype=torch.float32),
        }


class GroupedReplayDataset(Dataset):
    """Candidate action groups used for within-state critic supervision."""

    def __init__(self, transitions: list[SolverTransition]):
        groups: dict[str, list[SolverTransition]] = {}
        for tr in transitions:
            if tr.group_id:
                groups.setdefault(tr.group_id, []).append(tr)
        self.groups = [rows for rows in groups.values() if len(rows) >= 2]

    def __len__(self) -> int:
        return len(self.groups)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        rows = self.groups[idx]
        return {
            "state_fields": torch.stack([tr.state_fields for tr in rows]),
            "state_scalars": torch.stack([tr.state_scalars for tr in rows]),
            "action": torch.stack([tr.action.reshape(-1) for tr in rows]),
            "mc_return": torch.tensor([tr.mc_return for tr in rows], dtype=torch.float32),
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
