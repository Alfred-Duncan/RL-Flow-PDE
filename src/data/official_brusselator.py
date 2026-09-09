from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


OFFICIAL_DRIVE_URL = "https://drive.google.com/drive/folders/1x8EYALKl2l9lxpMVy6rfj934kno4V0qB?usp=sharing"
EXPECTED_NAME = "Brusselator_force_train.npz"


def expected_data_path(root: Path) -> Path:
    return root / "data" / "official" / "brusselator" / EXPECTED_NAME


def require_official_data(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(
            f"Official 3D_Brusselator data is required at {path}. Download {EXPECTED_NAME} "
            f"from {OFFICIAL_DRIVE_URL}, or run: python scripts/prepare_brusselator.py --download"
        )
    return path


class OfficialBrusselatorDataset(Dataset):
    """Official LNO 3D_Brusselator data with explicit (case, time, x, y) layout."""

    def __init__(self, path: Path, split: str, max_cases: int | None = None, split_seed: int = 2026):
        path = require_official_data(path)
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unsupported split: {split}")
        with np.load(path) as data:
            required = {"nx", "ny", "n_samples1", "n_samples2", "inputs_train", "inputs_test", "outputs_train", "outputs_test"}
            missing = required.difference(data.files)
            if missing:
                raise ValueError(f"Official NPZ missing required keys: {sorted(missing)}")
            nt, nx, ny = 39, int(data["nx"]), int(data["ny"])
            if (nx, ny) != (28, 28):
                raise ValueError(f"Expected official 28x28 grid, found {nx}x{ny}.")
            train_n, test_n = int(data["n_samples1"]), int(data["n_samples2"])
            train_force = np.asarray(data["inputs_train"], dtype=np.float32).reshape(train_n, nt)
            test_force = np.asarray(data["inputs_test"], dtype=np.float32).reshape(test_n, nt)
            train_gt = np.asarray(data["outputs_train"], dtype=np.float32).reshape(train_n, nt, nx, ny)
            test_gt = np.asarray(data["outputs_test"], dtype=np.float32).reshape(test_n, nt, nx, ny)

        if train_gt.shape[1:] != (39, 28, 28) or test_gt.shape[1:] != (39, 28, 28):
            raise ValueError("Official 3D_Brusselator layout must be (case, 39, 28, 28).")
        # This is exactly the r=2 preprocessing used by the official LNO script.
        train_gt, test_gt = train_gt[:, :, ::2, ::2], test_gt[:, :, ::2, ::2]
        if train_gt.shape[1:] != (39, 14, 14) or test_gt.shape[1:] != (39, 14, 14):
            raise AssertionError("Expected official r=2 layout (case, 39, 14, 14).")

        if split == "train":
            force, gt, ids = train_force, train_gt, np.arange(train_n, dtype=np.int64)
        else:
            permutation = np.random.default_rng(split_seed).permutation(test_n)
            midpoint = test_n // 2
            selected = permutation[:midpoint] if split == "val" else permutation[midpoint:]
            force, gt, ids = test_force[selected], test_gt[selected], selected.astype(np.int64) + train_n
        if max_cases is not None:
            force, gt, ids = force[:max_cases], gt[:max_cases], ids[:max_cases]

        self.forcing = torch.from_numpy(force.copy())
        self.gt = torch.from_numpy(gt.copy())
        self.case_ids = torch.from_numpy(ids.copy())
        self.metadata = {"nt": 39, "nx": 14, "ny": 14, "raw_nx": 28, "raw_ny": 28, "split": split}

    def __len__(self) -> int:
        return self.gt.shape[0]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        forcing = self.forcing[index]
        return {
            "forcing": forcing,
            "forcing_field": forcing[:, None, None].expand(39, 14, 14),
            "gt": self.gt[index],
            "case_id": self.case_ids[index],
        }
