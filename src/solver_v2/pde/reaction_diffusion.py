from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class ReactionDiffusionCase:
    source: torch.Tensor
    gt: torch.Tensor
    ic: torch.Tensor
    bc_left: torch.Tensor
    bc_right: torch.Tensor
    x: torch.Tensor
    t: torch.Tensor
    case_id: int


class ReactionDiffusionPDE:
    """Reaction-diffusion PDE utilities for the official 2D_Reac_diffusion grid.

    Tensors use shape (nx, nt), where axis 0 is space x and axis 1 is time t.
    The residual convention is u_t - D u_xx - k u^2 - f(x,t).
    """

    def __init__(self, diffusion: float, reaction: float):
        self.diffusion = float(diffusion)
        self.reaction = float(reaction)

    def make_case(self, item: dict, device: torch.device) -> ReactionDiffusionCase:
        gt = item["gt"].to(device)
        return ReactionDiffusionCase(
            source=item["source"].to(device),
            gt=gt,
            ic=item["ic"].to(device),
            bc_left=item["bc_left"].to(device),
            bc_right=item["bc_right"].to(device),
            x=item["x"].to(device).reshape(-1),
            t=item["t"].to(device).reshape(-1),
            case_id=int(item.get("case_id", -1)),
        )

    def project_hard_constraints(self, u: torch.Tensor, case: ReactionDiffusionCase) -> torch.Tensor:
        out = u.clone()
        out[:, 0] = case.ic
        out[0, :] = case.bc_left
        out[-1, :] = case.bc_right
        return out.clamp(-8.0, 8.0)

    def residual(self, u: torch.Tensor, case: ReactionDiffusionCase) -> torch.Tensor:
        dx = (case.x[1] - case.x[0]).abs().clamp_min(1e-6)
        dt = (case.t[1] - case.t[0]).abs().clamp_min(1e-6)
        ut = torch.zeros_like(u)
        ut[:, 1:-1] = (u[:, 2:] - u[:, :-2]) / (2.0 * dt)
        ut[:, 0] = (u[:, 1] - u[:, 0]) / dt
        ut[:, -1] = (u[:, -1] - u[:, -2]) / dt
        uxx = torch.zeros_like(u)
        uxx[1:-1, :] = (u[2:, :] - 2.0 * u[1:-1, :] + u[:-2, :]) / (dx * dx)
        return ut - self.diffusion * uxx - self.reaction * (u ** 2) - case.source

    def relative_l2(self, u: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        return torch.norm(u - gt) / torch.norm(gt).clamp_min(1e-8)

    def physics_metrics(self, u: torch.Tensor, case: ReactionDiffusionCase) -> dict[str, torch.Tensor]:
        res = self.residual(u, case)
        ic_error = torch.mean((u[:, 0] - case.ic) ** 2)
        bc_error = torch.mean((u[0, :] - case.bc_left) ** 2) + torch.mean((u[-1, :] - case.bc_right) ** 2)
        residual_norm = torch.mean(res ** 2)
        energy = residual_norm + 10.0 * ic_error + 10.0 * bc_error
        return {"residual_norm": residual_norm, "ic_error": ic_error, "bc_error": bc_error, "energy": energy}

    def step(self, u: torch.Tensor, delta_u: torch.Tensor, case: ReactionDiffusionCase) -> torch.Tensor:
        return self.project_hard_constraints(u + delta_u, case)

    def reward(self, u: torch.Tensor, next_u: torch.Tensor, delta_u: torch.Tensor, case: ReactionDiffusionCase, lambda_action: float) -> dict[str, float]:
        e0 = self.relative_l2(u, case.gt)
        e1 = self.relative_l2(next_u, case.gt)
        r_acc = torch.log((e0 + 1e-8) / (e1 + 1e-8))
        action_cost = torch.mean(delta_u ** 2)
        reward = r_acc - float(lambda_action) * action_cost
        p0 = self.physics_metrics(u, case)
        p1 = self.physics_metrics(next_u, case)
        return {
            "reward": float(reward.detach().cpu()),
            "accuracy_reward": float(r_acc.detach().cpu()),
            "error_before": float(e0.detach().cpu()),
            "error_after": float(e1.detach().cpu()),
            "residual_before": float(p0["residual_norm"].detach().cpu()),
            "residual_after": float(p1["residual_norm"].detach().cpu()),
            "physics_before": float(p0["energy"].detach().cpu()),
            "physics_after": float(p1["energy"].detach().cpu()),
            "action_norm": float(torch.sqrt(action_cost).detach().cpu()),
        }

