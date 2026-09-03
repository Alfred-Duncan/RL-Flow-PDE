from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import torch


def relative_l2(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return torch.linalg.vector_norm(pred - target, dim=(-2, -1)) / (torch.linalg.vector_norm(target, dim=(-2, -1)) + eps)


def scalar_relative_l2(pred: torch.Tensor, target: torch.Tensor) -> float:
    return float(relative_l2(pred.detach().cpu(), target.detach().cpu()).mean())


def mean_std(values: Iterable[float]) -> str:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return "nan +/- nan"
    return f"{arr.mean():.6g} +/- {arr.std(ddof=0):.6g}"


def corr(x: Iterable[float], y: Iterable[float]) -> tuple[float, float]:
    xa = np.asarray(list(x), dtype=np.float64)
    ya = np.asarray(list(y), dtype=np.float64)
    if xa.size < 2 or ya.size < 2 or np.std(xa) < 1e-12 or np.std(ya) < 1e-12:
        return 0.0, 0.0
    pearson = float(np.corrcoef(xa, ya)[0, 1])
    rx = np.argsort(np.argsort(xa))
    ry = np.argsort(np.argsort(ya))
    spearman = float(np.corrcoef(rx, ry)[0, 1])
    return pearson, spearman
