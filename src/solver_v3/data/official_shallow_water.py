from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


OFFICIAL_DRIVE_URL = "https://drive.google.com/drive/folders/1x8EYALKl2l9lxpMVy6rfj934kno4V0qB?usp=sharing"
RAW_FILES = ("shallow-water-256x256x72_1.npz", "shallow-water-256x256x72_2.npz")


class OfficialShallowWater:
    """Lazy adapter for the official LNO 3D_shallow raw trajectory archives."""

    def __init__(self, directory: Path, train_cases: int = 230, validation_cases: int = 20, test_cases: int = 50):
        self.directory = directory
        self.paths = [directory / name for name in RAW_FILES]
        missing = [str(path) for path in self.paths if not path.exists()]
        if missing:
            raise FileNotFoundError(
                "Official LNO Shallow Water raw data is missing: " + "; ".join(missing) +
                f". Download from {OFFICIAL_DRIVE_URL} or run scripts/prepare_official_shallow.py --download."
            )
        self._archives: list[dict[str, np.ndarray] | None] = [None, None]
        metadata = self._load_archive(0)
        if not {"inputs", "outputs"}.issubset(metadata):
            raise ValueError(f"Expected official keys inputs/outputs, found {sorted(metadata)} in {self.paths[0]}.")
        inputs, outputs = metadata["inputs"], metadata["outputs"]
        self.channels, self.nt, self.nx, self.ny = self._infer_output_layout(outputs)
        if (self.nt, self.nx, self.ny) != (72, 256, 256):
            raise ValueError(f"Official shallow-water trajectory must resolve to 72x256x256, found {outputs.shape}.")
        count_a = self._case_count(outputs)
        count_b = self._case_count(self._load_archive(1)["outputs"])
        self.counts = (count_a, count_b)
        self.total_cases = count_a + count_b
        if self.total_cases < train_cases + validation_cases + test_cases:
            raise ValueError(f"Requested {train_cases + validation_cases + test_cases} cases but official data contains {self.total_cases}.")
        self.splits = {
            "train": list(range(train_cases)),
            "val": list(range(train_cases, train_cases + validation_cases)),
            "test": list(range(train_cases + validation_cases, train_cases + validation_cases + test_cases)),
        }
        self.metadata = {"channels": self.channels, "nt": self.nt, "nx": self.nx, "ny": self.ny, "raw_input_shape": tuple(inputs.shape), "raw_output_shape": tuple(outputs.shape)}

    def _load_archive(self, archive_index: int) -> dict[str, np.ndarray]:
        if self._archives[archive_index] is None:
            with np.load(self.paths[archive_index], allow_pickle=False) as archive:
                self._archives[archive_index] = {key: np.asarray(archive[key]) for key in archive.files}
        return self._archives[archive_index]  # type: ignore[return-value]

    @staticmethod
    def _infer_output_layout(outputs: np.ndarray) -> tuple[int, int, int, int]:
        if outputs.ndim == 4:
            # Official scalar layout: (case, time, x, y).
            return 1, int(outputs.shape[1]), int(outputs.shape[2]), int(outputs.shape[3])
        if outputs.ndim == 5:
            # Preserve all channels; official variants may use (case,time,channel,x,y) or (case,channel,time,x,y).
            if outputs.shape[1] == 72:
                return int(outputs.shape[2]), int(outputs.shape[1]), int(outputs.shape[3]), int(outputs.shape[4])
            if outputs.shape[2] == 72:
                return int(outputs.shape[1]), int(outputs.shape[2]), int(outputs.shape[3]), int(outputs.shape[4])
        raise ValueError(f"Unsupported official shallow-water output layout: {outputs.shape}.")

    @staticmethod
    def _case_count(outputs: np.ndarray) -> int:
        return int(outputs.shape[0])

    def _case_location(self, case_id: int) -> tuple[int, int]:
        if not 0 <= case_id < self.total_cases:
            raise IndexError(case_id)
        return (0, case_id) if case_id < self.counts[0] else (1, case_id - self.counts[0])

    def trajectory(self, case_id: int) -> torch.Tensor:
        archive_index, local_index = self._case_location(case_id)
        output = self._load_archive(archive_index)["outputs"][local_index]
        if output.ndim == 3:
            value = output[:, None]
        elif output.shape[0] == 72:
            value = output
        else:
            value = np.moveaxis(output, 0, 1)
        value = np.asarray(value, dtype=np.float32)
        if value.shape != (72, self.channels, 256, 256):
            raise AssertionError(f"Adapter produced unexpected trajectory shape {value.shape}.")
        return torch.from_numpy(value.copy())

    def frame(self, case_id: int, time_index: int) -> torch.Tensor:
        """Read one physical-time frame without materializing a copied full trajectory."""
        if not 0 <= time_index < self.nt:
            raise IndexError(time_index)
        archive_index, local_index = self._case_location(case_id)
        output = self._load_archive(archive_index)["outputs"][local_index]
        if output.ndim == 3:
            value = output[time_index, None]
        elif output.shape[0] == 72:
            value = output[time_index]
        else:
            value = output[:, time_index]
        value = np.asarray(value, dtype=np.float32)
        if value.shape != (self.channels, 256, 256):
            raise AssertionError(f"Adapter produced unexpected frame shape {value.shape}.")
        return torch.from_numpy(value.copy())

    def condition(self, case_id: int) -> torch.Tensor:
        """Return the official operator input field ``a(x,y)`` for one case.

        The benchmark does not document a narrower physical interpretation for
        this array, so callers must treat it as the official condition field.
        """
        archive_index, local_index = self._case_location(case_id)
        value = np.asarray(self._load_archive(archive_index)["inputs"][local_index], dtype=np.float32)
        if value.ndim == 2:
            value = value[None]
        elif value.ndim == 3:
            # Preserve a leading channel layout if a future official variant
            # contains multiple condition fields.
            pass
        else:
            raise AssertionError(f"Unsupported official condition shape {value.shape}.")
        if value.shape[-2:] != (256, 256):
            raise AssertionError(f"Official condition must be 256x256, found {value.shape}.")
        return torch.from_numpy(value.copy())

    def case_ids(self, split: str) -> list[int]:
        return self.splits[split]
