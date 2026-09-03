from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import Dataset


def _fallback_reaction_diffusion(path: Path, n_cases: int, nx: int, nt: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    x = np.linspace(0.0, 1.0, nx, dtype=np.float32)
    t = np.linspace(0.0, 1.0, nt, dtype=np.float32)
    dx = float(x[1] - x[0])
    dt = float(t[1] - t[0])
    d = 1.0 - 0.95 / np.pi**2
    k = 1.0
    f = np.zeros((n_cases, nx, nt), dtype=np.float32)
    u = np.zeros_like(f)
    for n in range(n_cases):
        amps = rng.normal(0.0, 0.25, size=4)
        phases = rng.uniform(0.0, 2 * np.pi, size=4)
        source = np.zeros((nx, nt), dtype=np.float32)
        for j in range(4):
            source += amps[j] * np.sin((j + 1) * np.pi * x[:, None] + phases[j]) * np.cos((j + 1) * np.pi * t[None, :])
        ic = 0.15 * np.sin(np.pi * x + rng.uniform(0, 2 * np.pi)).astype(np.float32)
        uu = np.zeros((nx, nt), dtype=np.float32)
        uu[:, 0] = ic
        uu[0, :] = 0.0
        uu[-1, :] = 0.0
        for m in range(nt - 1):
            lap = np.zeros(nx, dtype=np.float32)
            lap[1:-1] = (uu[2:, m] - 2 * uu[1:-1, m] + uu[:-2, m]) / (dx * dx)
            uu[1:-1, m + 1] = uu[1:-1, m] + dt * (d * lap[1:-1] + k * uu[1:-1, m] ** 2 + source[1:-1, m])
            uu[:, m + 1] = np.clip(uu[:, m + 1], -2.0, 2.0)
        f[n] = source
        u[n] = uu
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, x=x, t=t, f_train=f[: int(0.7 * n_cases)], u_train=u[: int(0.7 * n_cases)],
             f_vali=f[int(0.7 * n_cases): int(0.85 * n_cases)], u_vali=u[int(0.7 * n_cases): int(0.85 * n_cases)],
             f_test=f[int(0.85 * n_cases):], u_test=u[int(0.85 * n_cases):], source="fallback")


def prepare_reaction_diffusion(cfg: dict, root: Path) -> Path:
    out = root / cfg["benchmark"]["prepared_npz"]
    if out.exists():
        return out
    mat_path = root / cfg["benchmark"]["official_mat"]
    out.parent.mkdir(parents=True, exist_ok=True)
    if mat_path.exists():
        data = sio.loadmat(mat_path)
        np.savez(
            out,
            x=np.asarray(data["x"]).reshape(-1).astype(np.float32),
            t=np.asarray(data["t"]).reshape(-1).astype(np.float32),
            f_train=np.asarray(data["f_train"], dtype=np.float32),
            u_train=np.asarray(data["u_train"], dtype=np.float32),
            f_vali=np.asarray(data["f_vali"], dtype=np.float32),
            u_vali=np.asarray(data["u_vali"], dtype=np.float32),
            f_test=np.asarray(data["f_test"], dtype=np.float32),
            u_test=np.asarray(data["u_test"], dtype=np.float32),
            source="official_lno_2d_reac_diffusion",
        )
    else:
        _fallback_reaction_diffusion(out, int(cfg["benchmark"]["fallback_cases"]), int(cfg["benchmark"]["nx"]), int(cfg["benchmark"]["nt"]), int(cfg["seed"]))
    return out


class ReactionDiffusionDataset(Dataset):
    def __init__(self, npz_path: str | Path, split: str, limit: int | None = None):
        data = np.load(npz_path, allow_pickle=True)
        key = "vali" if split == "val" else split
        self.f = torch.from_numpy(np.asarray(data[f"f_{key}"], dtype=np.float32))
        self.u = torch.from_numpy(np.asarray(data[f"u_{key}"], dtype=np.float32))
        if limit is not None:
            self.f = self.f[:limit]
            self.u = self.u[:limit]
        self.x = torch.from_numpy(np.asarray(data["x"], dtype=np.float32).reshape(-1))
        self.t = torch.from_numpy(np.asarray(data["t"], dtype=np.float32).reshape(-1))
        self.split = split

    def __len__(self) -> int:
        return int(self.u.shape[0])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | int]:
        gt = self.u[idx]
        src = self.f[idx]
        return {
            "case_id": idx,
            "source": src,
            "gt": gt,
            "ic": gt[:, 0].clone(),
            "bc_left": gt[0, :].clone(),
            "bc_right": gt[-1, :].clone(),
            "x": self.x,
            "t": self.t,
        }
