from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


def relative_l2(value: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.linalg.vector_norm((value - target).flatten(1), dim=1) / torch.linalg.vector_norm(target.flatten(1), dim=1).clamp_min(1e-8)


class BrusselatorRefiner:
    """Generic-size frozen-proposal environment for forcing-driven 2D states."""

    def __init__(self, coarse: torch.nn.Module, local: torch.nn.Module, grid: int, core: int, halo: int, mean: torch.Tensor, std: torch.Tensor, force_mean: torch.Tensor, force_std: torch.Tensor):
        self.coarse, self.local, self.grid, self.core, self.halo = coarse, local, grid, core, halo
        self.mean, self.std, self.force_mean, self.force_std = mean, std, force_mean, force_std

    def norm(self, value: torch.Tensor) -> torch.Tensor: return (value - self.mean.to(value.device)) / self.std.to(value.device)
    def stabilize(self, value: torch.Tensor) -> torch.Tensor:
        """Keep autoregressive operator states finite without tightening the solver range."""
        mean, std = self.mean.to(value.device), self.std.to(value.device)
        limit = 8.0 * std
        return torch.nan_to_num(value, nan=float(mean), posinf=float(mean + limit), neginf=float(mean - limit)).clamp(mean - limit, mean + limit)
    def coarse_next(self, force: torch.Tensor, previous: torch.Tensor, current: torch.Tensor, time: float) -> torch.Tensor:
        f = ((force.to(current.device) - self.force_mean.to(current.device)) / self.force_std.to(current.device)).view(-1, 1, 1, 1).expand_as(current[:, :1])
        t = torch.full_like(current[:, :1], time)
        value = self.coarse(torch.cat([f, self.norm(previous), self.norm(current), self.norm(current - previous), t], 1)) * self.std.to(current.device) + self.mean.to(current.device)
        return self.stabilize(value)
    def bounds(self, patch: int) -> tuple[int, int, int, int]:
        row, col = divmod(patch, self.grid); return row * self.core - self.halo, (row + 1) * self.core + self.halo, col * self.core - self.halo, (col + 1) * self.core + self.halo
    def extract(self, value: torch.Tensor, patch: int) -> torch.Tensor:
        top, bottom, left, right = self.bounds(patch); padded = F.pad(value, (self.halo, self.halo, self.halo, self.halo), mode="reflect")
        return padded[:, :, top + self.halo:bottom + self.halo, left + self.halo:right + self.halo]
    def inputs(self, force: torch.Tensor, previous: torch.Tensor, current: torch.Tensor, provisional: torch.Tensor, time: float) -> torch.Tensor:
        rows=[]
        for patch in range(self.grid * self.grid):
            prev, now, nxt = (self.extract(v, patch) for v in (previous, current, provisional)); f = torch.full_like(nxt[:, :1], float(force.flatten()[0])); t=torch.full_like(nxt[:, :1], time)
            rows.append(torch.cat([f, prev, now, nxt, now-prev, nxt-now, t], 1))
        return torch.cat(rows)
    def window(self, size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        ramp=torch.sin(torch.linspace(0, torch.pi/2, self.halo+1, device=device,dtype=dtype))[1:]; axis=torch.cat([ramp,torch.ones(size-2*self.halo,device=device,dtype=dtype),ramp.flip(0)]); return axis[:,None]*axis[None,:]
    def canvases(self, provisional: torch.Tensor, corrections: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        n,h,w=self.grid*self.grid,*provisional.shape[-2:]; contribution=torch.zeros((n,*provisional.shape[1:]),device=provisional.device,dtype=provisional.dtype); weight=torch.zeros((n,1,h,w),device=provisional.device,dtype=provisional.dtype)
        for patch, correction in enumerate(corrections.split(1,0)):
            top,bottom,left,right=self.bounds(patch); r0,r1,c0,c1=max(top,0),min(bottom,h),max(left,0),min(right,w); rs,cs=slice(r0-top,r1-top),slice(c0-left,c1-left); win=self.window(bottom-top, provisional.device, provisional.dtype)[None,None]
            contribution[patch:patch+1,:,r0:r1,c0:c1]+=correction[:,:,rs,cs]*win[:,:,rs,cs]; weight[patch:patch+1,:,r0:r1,c0:c1]+=win[:,:,rs,cs]
        return contribution,weight


@dataclass
class FrozenBundle:
    force: torch.Tensor; previous: torch.Tensor; current: torch.Tensor; provisional: torch.Tensor; inputs: torch.Tensor; corrections: torch.Tensor; contributions: torch.Tensor; weights: torch.Tensor; refiner: BrusselatorRefiner
    @classmethod
    @torch.no_grad()
    def create(cls, refiner: BrusselatorRefiner, force: torch.Tensor, previous: torch.Tensor, current: torch.Tensor, time: float) -> "FrozenBundle":
        provisional=refiner.coarse_next(force,previous,current,time); inputs=refiner.inputs(force,previous,current,provisional,time); corrections=torch.nan_to_num(refiner.local(inputs)); contributions,weights=refiner.canvases(provisional,corrections); return cls(force,previous,current,provisional,inputs,corrections,contributions,weights,refiner)
    def apply_set(self, selected: list[int]) -> torch.Tensor:
        if not selected:return self.provisional
        ids=torch.tensor(selected,device=self.provisional.device); return self.refiner.stabilize(self.provisional+self.contributions[ids].sum(0,keepdim=True)/self.weights[ids].sum(0,keepdim=True).clamp_min(1e-6))
    def candidate_fields(self, selected: list[int]) -> torch.Tensor:
        if selected:
            ids=torch.tensor(selected,device=self.provisional.device); value,weight=self.contributions[ids].sum(0,keepdim=True),self.weights[ids].sum(0,keepdim=True)
        else:value,weight=torch.zeros_like(self.provisional),torch.zeros_like(self.provisional[:,:1])
        return self.refiner.stabilize(self.provisional+(value+self.contributions)/(weight+self.weights).clamp_min(1e-6))
