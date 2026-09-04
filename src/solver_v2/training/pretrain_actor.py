from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.solver_v2.data.replay_buffer import ReplayDataset, SolverTransition
from src.solver_v2.models.operator_actor import DeterministicNeuralOperatorActor


def pretrain_actor(
    actor: DeterministicNeuralOperatorActor,
    transitions: list[SolverTransition],
    cfg: dict,
    stats: dict,
    device: torch.device,
    ckpt: Path,
) -> DeterministicNeuralOperatorActor:
    if ckpt.exists():
        actor.load_state_dict(torch.load(ckpt, map_location=device)["actor"])
        return actor.eval()
    ds = ReplayDataset([tr for tr in transitions if tr.split == "train"])
    loader = DataLoader(ds, batch_size=int(cfg["solver_v2"]["batch_size"]), shuffle=True)
    opt = torch.optim.AdamW(actor.parameters(), lr=float(cfg["solver_v2"]["lr"]), weight_decay=1e-4)
    for _ in tqdm(range(int(cfg["solver_v2"]["pretrain_epochs"])), desc="solver_v2:pretrain_actor"):
        actor.train()
        for batch in loader:
            fields = batch["state_fields"].to(device)
            scalars = batch["state_scalars"].to(device)
            action = batch["action"].to(device)
            pred = actor(fields, scalars, stats, float(cfg["solver_v2"]["temperature"]))
            loss = F.mse_loss(pred, action)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
            opt.step()
    torch.save({"actor": actor.state_dict()}, ckpt)
    return actor.eval()
