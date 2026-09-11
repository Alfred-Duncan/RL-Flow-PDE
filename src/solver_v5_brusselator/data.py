from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


class OfficialBrusselatorV5:
    """Official forcing-driven 2D trajectories with time axis made explicit.

    The official archive stores a scalar forcing value at every physical time.
    It has no separate static condition channel, so ``forcing(case, t)`` is the
    only exogenous operator input exposed to the transition model.
    """

    def __init__(self, path: Path, split_seed: int = 2026):
        with np.load(path) as archive:
            self.train_forcing = torch.from_numpy(np.asarray(archive["inputs_train"], dtype=np.float32).copy())
            self.test_forcing = torch.from_numpy(np.asarray(archive["inputs_test"], dtype=np.float32).copy())
            self.train_states = torch.from_numpy(np.asarray(archive["outputs_train"], dtype=np.float32).copy())
            self.test_states = torch.from_numpy(np.asarray(archive["outputs_test"], dtype=np.float32).copy())
        if self.train_states.shape[1:] != (39, 28, 28) or self.test_states.shape[1:] != (39, 28, 28):
            raise ValueError(f"Expected official physical layout (*,39,28,28), found {tuple(self.train_states.shape)} / {tuple(self.test_states.shape)}")
        self.nt, self.nx, self.ny = 39, 28, 28
        permutation = np.random.default_rng(split_seed).permutation(len(self.test_states))
        offset = len(self.train_states)
        self.splits = {"train": list(range(offset)), "val": (offset + permutation[: len(permutation) // 2]).tolist(), "test": (offset + permutation[len(permutation) // 2 :]).tolist()}
        self.metadata = {"input_shape": tuple(self.train_forcing.shape), "output_shape": tuple(self.train_states.shape), "physical_time_axis": 1, "state_channels": 1, "static_condition": False, "spatial_shape": (28, 28)}

    def case_ids(self, split: str) -> list[int]:
        return self.splits[split]

    def _store(self, case: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        if case < len(self.train_states):
            return self.train_states, self.train_forcing, case
        return self.test_states, self.test_forcing, case - len(self.train_states)

    def frame(self, case: int, time_index: int) -> torch.Tensor:
        states, _, local = self._store(case)
        return states[local, time_index].unsqueeze(0).clone()

    def forcing(self, case: int, time_index: int) -> torch.Tensor:
        _, forcing, local = self._store(case)
        return forcing[local, time_index].clone()
