from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from src.utils.metrics import relative_l2


@dataclass
class PDECase:
    source: torch.Tensor
    gt: torch.Tensor
    ic: torch.Tensor
    bc_left: torch.Tensor
    bc_right: torch.Tensor
    x: torch.Tensor
    t: torch.Tensor
    case_id: int = -1


class ReactionDiffusionEnvironment:
    def __init__(self, diffusion: float, reaction: float, action_scale: float = 0.16, modes: int = 8, device: torch.device | None = None):
        self.diffusion = float(diffusion)
        self.reaction = float(reaction)
        self.action_scale = float(action_scale)
        self.modes = int(modes)
        self.device = device or torch.device("cpu")
        self.case: PDECase | None = None

    def reset(self, case: dict) -> torch.Tensor:
        self.case = PDECase(
            source=case["source"].to(self.device),
            gt=case["gt"].to(self.device),
            ic=case["ic"].to(self.device),
            bc_left=case["bc_left"].to(self.device),
            bc_right=case["bc_right"].to(self.device),
            x=case["x"].to(self.device),
            t=case["t"].to(self.device),
            case_id=int(case.get("case_id", -1)),
        )
        return self.project(self.case.gt * 0.0)

    def project(self, u: torch.Tensor) -> torch.Tensor:
        assert self.case is not None
        out = u.clone()
        out[:, 0] = self.case.ic
        out[0, :] = self.case.bc_left
        out[-1, :] = self.case.bc_right
        return out.clamp(-6.0, 6.0)

    def residual(self, u: torch.Tensor) -> torch.Tensor:
        assert self.case is not None
        dx = torch.clamp(self.case.x[1] - self.case.x[0], min=1e-6)
        dt = torch.clamp(self.case.t[1] - self.case.t[0], min=1e-6)
        ut = torch.zeros_like(u)
        ut[:, 1:-1] = (u[:, 2:] - u[:, :-2]) / (2.0 * dt)
        ut[:, 0] = (u[:, 1] - u[:, 0]) / dt
        ut[:, -1] = (u[:, -1] - u[:, -2]) / dt
        uxx = torch.zeros_like(u)
        uxx[1:-1, :] = (u[2:, :] - 2.0 * u[1:-1, :] + u[:-2, :]) / (dx * dx)
        return ut - self.diffusion * uxx - self.reaction * (u ** 2) - self.case.source

    def physics_energy(self, u: torch.Tensor) -> dict[str, torch.Tensor]:
        assert self.case is not None
        res = self.residual(u)
        ic_error = torch.mean((u[:, 0] - self.case.ic) ** 2)
        bc_error = torch.mean((u[0, :] - self.case.bc_left) ** 2) + torch.mean((u[-1, :] - self.case.bc_right) ** 2)
        residual_norm = torch.mean(res ** 2)
        energy = residual_norm + 10.0 * ic_error + 10.0 * bc_error
        return {"residual_norm": residual_norm, "ic_error": ic_error, "bc_error": bc_error, "energy": energy}

    def relative_error(self, u: torch.Tensor, gt: torch.Tensor | None = None) -> torch.Tensor:
        assert self.case is not None
        return relative_l2(u, self.case.gt if gt is None else gt).mean()

    def decode_action(self, action: torch.Tensor, shape: tuple[int, int]) -> torch.Tensor:
        nx, nt = shape
        coeff = action[:-1].view(2, self.modes, self.modes)
        spec = torch.zeros((nx, nt // 2 + 1), dtype=torch.complex64, device=action.device)
        m1 = min(self.modes, nx)
        m2 = min(self.modes, nt // 2 + 1)
        spec[:m1, :m2] = torch.complex(coeff[0, :m1, :m2], coeff[1, :m1, :m2])
        return torch.fft.irfft2(spec, s=(nx, nt)).real

    def lowpass(self, delta: torch.Tensor) -> torch.Tensor:
        spec = torch.fft.rfft2(delta)
        filt = torch.zeros_like(spec)
        m1 = min(self.modes, delta.shape[0])
        m2 = min(self.modes, delta.shape[1] // 2 + 1)
        filt[:m1, :m2] = spec[:m1, :m2]
        return torch.fft.irfft2(filt, s=tuple(delta.shape)).real

    def encode_correction(self, delta: torch.Tensor) -> torch.Tensor:
        spec = torch.fft.rfft2(delta)
        m1 = min(self.modes, delta.shape[0])
        m2 = min(self.modes, delta.shape[1] // 2 + 1)
        real = torch.zeros((self.modes, self.modes), device=delta.device)
        imag = torch.zeros_like(real)
        real[:m1, :m2] = spec[:m1, :m2].real
        imag[:m1, :m2] = spec[:m1, :m2].imag
        vec = torch.cat([real.flatten(), imag.flatten(), torch.zeros(1, device=delta.device)])
        return vec.to(torch.float32)

    def step(self, u: torch.Tensor, correction: torch.Tensor) -> torch.Tensor:
        if correction.ndim == 1 and correction.numel() == 2 * self.modes * self.modes + 1:
            delta = self.decode_action(correction, tuple(u.shape))
        else:
            delta = correction
        return self.project(u + delta)

    def coarse_initialization(self) -> torch.Tensor:
        assert self.case is not None
        u = torch.zeros_like(self.case.gt)
        u[:, 0] = self.case.ic
        dx = torch.clamp(self.case.x[1] - self.case.x[0], min=1e-6)
        dt = torch.clamp(self.case.t[1] - self.case.t[0], min=1e-6)
        for j in range(u.shape[1] - 1):
            lap = torch.zeros_like(u[:, j])
            lap[1:-1] = (u[2:, j] - 2.0 * u[1:-1, j] + u[:-2, j]) / (dx * dx)
            u[1:-1, j + 1] = u[1:-1, j] + 0.25 * dt * (self.diffusion * lap[1:-1] + self.reaction * u[1:-1, j] ** 2 + self.case.source[1:-1, j])
            u = self.project(u)
        return F.avg_pool2d(u[None, None], kernel_size=3, stride=1, padding=1).squeeze()

    def reward(self, u: torch.Tensor, next_u: torch.Tensor, action: torch.Tensor, weights: dict) -> dict[str, float]:
        e0 = self.relative_error(u)
        e1 = self.relative_error(next_u)
        p0 = self.physics_energy(u)["energy"]
        p1 = self.physics_energy(next_u)["energy"]
        eps = 1e-8
        gt_improvement = torch.log((e0 + eps) / (e1 + eps))
        physics_improvement = torch.log((p0 + eps) / (p1 + eps))
        action_penalty = torch.mean(action ** 2)
        r = weights["lambda_gt"] * gt_improvement + weights["lambda_phys"] * physics_improvement - weights["lambda_action"] * action_penalty - weights["lambda_step"]
        return {
            "reward": float(r.detach().cpu()),
            "gt_improvement": float(gt_improvement.detach().cpu()),
            "physics_improvement": float(physics_improvement.detach().cpu()),
            "error_before": float(e0.detach().cpu()),
            "error_after": float(e1.detach().cpu()),
            "energy_before": float(p0.detach().cpu()),
            "energy_after": float(p1.detach().cpu()),
        }
