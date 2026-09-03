from __future__ import annotations

import json
import time
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.baselines.refinement import DeterministicCorrection, gradient_residual_refine, pinn_style_refine
from src.data.official_reaction_diffusion import ReactionDiffusionDataset, prepare_reaction_diffusion
from src.data.transition_buffer import Transition, TransitionDataset, split_transitions
from src.models.flow_policy import FlowPolicy
from src.models.lno_baseline import SmallLNOBaseline
from src.models.operator_state_encoder import OperatorStateEncoder, state_tensors
from src.pde.environment import ReactionDiffusionEnvironment
from src.rl.iql import SingleQCritic, TwinQCritic, ValueNet, iql_losses
from src.utils.metrics import corr, mean_std
from src.utils.seed import get_device, set_seed


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_VERSION = "protocol_v2"


def load_config(path: str | Path = "configs/default.yaml") -> dict:
    with open(ROOT / path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dirs() -> None:
    for p in ["checkpoints", "data/processed", "results/tables", "results/figures", "results/trajectories", "docs"]:
        (ROOT / p).mkdir(parents=True, exist_ok=True)


def as_case(item: dict, case_id: int | None = None) -> dict:
    case = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in item.items()}
    if case_id is not None:
        case["case_id"] = int(case_id)
    return case


def env_from_cfg(cfg: dict, device: torch.device) -> ReactionDiffusionEnvironment:
    b = cfg["benchmark"]
    p = cfg["paper"]
    return ReactionDiffusionEnvironment(float(b["diffusion"]), float(b["reaction"]), float(p["action_scale"]), int(p["modes"]), device)


def scalar_context(env: ReactionDiffusionEnvironment, u: torch.Tensor, k_frac: float, no_residual: bool = False, no_physics: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    e = env.physics_energy(u)
    res = torch.zeros_like(env.residual(u)) if no_residual else env.residual(u)
    vals = [
        float(torch.log1p(e["residual_norm"]).detach().cpu()),
        float(torch.log1p(e["ic_error"]).detach().cpu()),
        float(torch.log1p(e["bc_error"]).detach().cpu()),
        float(torch.log1p(e["energy"]).detach().cpu()) if not no_physics else 0.0,
        float(u.mean().detach().cpu()),
        float(u.std().detach().cpu()),
        float(k_frac),
        1.0,
    ]
    return torch.tensor(vals, dtype=torch.float32, device=u.device), res


def make_state(env: ReactionDiffusionEnvironment, u: torch.Tensor, k_frac: float, no_residual: bool = False, no_physics: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    assert env.case is not None
    scalars, res = scalar_context(env, u, k_frac, no_residual, no_physics)
    bc = torch.zeros_like(u)
    bc[0, :] = env.case.bc_left
    bc[-1, :] = env.case.bc_right
    return state_tensors(u, res, env.case.ic, bc, scalars)


def normalize_state(fields: torch.Tensor, scalars: torch.Tensor, stats: dict) -> tuple[torch.Tensor, torch.Tensor]:
    fields = fields.clone()
    fields[..., 1, :, :] = fields[..., 1, :, :] / stats["residual_scale"].to(fields.device).clamp_min(1e-8)
    scalars = (scalars - stats["scalar_mean"].to(scalars.device)) / stats["scalar_std"].to(scalars.device).clamp_min(1e-6)
    return fields, scalars


def make_state_normalized(env: ReactionDiffusionEnvironment, u: torch.Tensor, k_frac: float, stats: dict, no_residual: bool = False, no_physics: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    fields, scalars = make_state(env, u, k_frac, no_residual, no_physics)
    return normalize_state(fields, scalars, stats)


def normalize_action(action: torch.Tensor, stats: dict) -> torch.Tensor:
    return (action - stats["action_mean"].to(action.device)) / stats["action_std"].to(action.device).clamp_min(1e-6)


def denormalize_action(action: torch.Tensor, stats: dict) -> torch.Tensor:
    return action * stats["action_std"].to(action.device).clamp_min(1e-6) + stats["action_mean"].to(action.device)


def train_lno(cfg: dict, npz_path: Path, device: torch.device) -> SmallLNOBaseline:
    ckpt = ROOT / "checkpoints" / "lno_reacdiff.pt"
    model = SmallLNOBaseline(width=int(cfg["paper"]["state_width"]), modes=int(cfg["paper"]["modes"])).to(device)
    if ckpt.exists():
        model.load_state_dict(torch.load(ckpt, map_location=device))
        model.eval()
        return model
    ds = ReactionDiffusionDataset(npz_path, "train", int(cfg["paper"]["train_cases"]))
    loader = DataLoader(ds, batch_size=int(cfg["paper"]["batch_size"]), shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["paper"]["lr"]), weight_decay=1e-4)
    for _ in tqdm(range(int(cfg["paper"]["lno_epochs"])), desc="lno"):
        for batch in loader:
            src = batch["source"].to(device)
            gt = batch["gt"].to(device)
            pred = model(src)
            loss = F.mse_loss(pred, gt) + 0.1 * torch.mean(torch.abs(pred[:, :, 0] - gt[:, :, 0]))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    torch.save(model.state_dict(), ckpt)
    model.eval()
    return model


def candidate_states(env: ReactionDiffusionEnvironment, lno: SmallLNOBaseline, case: dict, device: torch.device) -> list[tuple[str, torch.Tensor]]:
    env.reset(case)
    source = case["source"].to(device).unsqueeze(0)
    with torch.no_grad():
        lno_u = env.project(lno(source).squeeze(0))
    coarse = env.coarse_initialization()
    gt = case["gt"].to(device)
    grad, _, _ = gradient_residual_refine(env, lno_u, 1, lr=0.01)
    pinn, _, _ = pinn_style_refine(env, lno_u, 1, lr=0.01)
    perturbed = env.project(gt + 0.10 * torch.randn_like(gt))
    return [("lno", lno_u), ("coarse", coarse), ("perturbed", perturbed), ("gradient", grad), ("pinn", pinn)]


def append_transition(rows: list[Transition], env: ReactionDiffusionEnvironment, split: str, kind: str, case_id: int, trajectory_id: int, step_idx: int, u: torch.Tensor, action: torch.Tensor, reward_weights: dict, step_frac: float, done: float, no_residual: bool = False, no_physics: bool = False) -> torch.Tensor:
    fields, scalars = make_state(env, u, step_frac, no_residual, no_physics)
    next_u = env.step(u, action)
    next_fields, next_scalars = make_state(env, next_u, min(1.0, step_frac + 0.1), no_residual, no_physics)
    r = env.reward(u, next_u, action, reward_weights)
    rows.append(
        Transition(
            case_id=case_id,
            split=split,
            kind=kind,
            trajectory_id=trajectory_id,
            step_idx=step_idx,
            state_fields=fields.squeeze(0).detach().cpu(),
            state_scalars=scalars.squeeze(0).detach().cpu(),
            action=action.detach().cpu(),
            reward=r["reward"],
            discounted_return=0.0,
            next_fields=next_fields.squeeze(0).detach().cpu(),
            next_scalars=next_scalars.squeeze(0).detach().cpu(),
            done=float(done),
            gt_improvement=r["gt_improvement"],
            physics_improvement=r["physics_improvement"],
            error_before=r["error_before"],
            error_after=r["error_after"],
            energy_before=r["energy_before"],
            energy_after=r["energy_after"],
        )
    )
    return next_u.detach()


def assign_discounted_returns(rows: list[Transition], gamma: float) -> None:
    by_traj: dict[int, list[int]] = defaultdict(list)
    for idx, tr in enumerate(rows):
        by_traj[int(tr.trajectory_id)].append(idx)
    for idxs in by_traj.values():
        idxs.sort(key=lambda j: rows[j].step_idx)
        running = 0.0
        for j in reversed(idxs):
            tr = rows[j]
            running = float(tr.reward) + gamma * running * (1.0 - float(tr.done))
            tr.discounted_return = running


def append_gt_sequence(rows, env, split, case_id, traj_id, u0, reward_weights, steps, no_resid, no_phys, label: str) -> None:
    assert env.case is not None
    u = u0.detach()
    for k in range(steps):
        frac = k / max(1, steps)
        remaining = max(1, steps - k)
        delta = env.lowpass(env.case.gt - u) / float(remaining)
        action = env.encode_correction(delta)
        u = append_transition(rows, env, split, f"seq_{label}_gt_lowpass", case_id, traj_id, k, u, action, reward_weights, frac, float(k == steps - 1), no_resid, no_phys)


def append_refine_sequence(rows, env, split, case_id, traj_id, u0, reward_weights, steps, no_resid, no_phys, method: str) -> None:
    u = u0.detach()
    for k in range(steps):
        frac = k / max(1, steps)
        if method == "gradient":
            next_u, _, _ = gradient_residual_refine(env, u, 1, lr=0.01)
        elif method == "pinn":
            next_u, _, _ = pinn_style_refine(env, u, 1, lr=0.008)
        else:
            assert env.case is not None
            direction = env.lowpass(env.case.gt - u) / float(max(1, steps - k))
            direction = direction if k % 3 != 1 else -direction
            next_u = env.project(u + direction + 0.03 * env.lowpass(torch.randn_like(u)))
        action = env.encode_correction(next_u - u)
        u = append_transition(rows, env, split, f"seq_{method}", case_id, traj_id, k, u, action, reward_weights, frac, float(k == steps - 1), no_resid, no_phys)


def build_transition_buffer(cfg: dict, npz_path: Path, lno: SmallLNOBaseline, device: torch.device, encoder_variant: str = "full", reward_variant: str = "full") -> list[Transition]:
    cache = ROOT / "checkpoints" / f"transitions_{encoder_variant}_{reward_variant}_{PROTOCOL_VERSION}.pt"
    if cache.exists():
        return torch.load(cache, weights_only=False)
    reward_weights = dict(cfg["reward"])
    if reward_variant == "no_physics":
        reward_weights["lambda_phys"] = 0.0
    if reward_variant == "no_gt":
        reward_weights["lambda_gt"] = 0.0
    env = env_from_cfg(cfg, device)
    rows: list[Transition] = []
    specs = [("train", int(cfg["paper"]["train_cases"])), ("val", int(cfg["paper"]["val_cases"])), ("test", int(cfg["paper"]["test_cases"]))]
    offset = 0
    traj_id = 0
    seq_steps = int(cfg["paper"].get("sequential_steps", 5))
    for split, limit in specs:
        ds = ReactionDiffusionDataset(npz_path, split, limit)
        for i in tqdm(range(len(ds)), desc=f"buffer:{split}:{encoder_variant}:{reward_variant}"):
            case_id = offset + i
            case = as_case(ds[i], case_id)
            env.reset(case)
            gt = case["gt"].to(device)
            no_resid = encoder_variant == "no_residual"
            no_phys = reward_variant == "no_physics"
            starts = candidate_states(env, lno, case, device)
            for kind, u in starts:
                good = env.encode_correction(env.lowpass(gt - u))
                bad = -good
                zero = torch.zeros_like(good)
                noise = 0.35 * torch.randn_like(good)
                over = 2.0 * good
                for name, action in [("good", good), ("bad", bad), ("zero", zero), ("noise", noise), ("over", over)]:
                    append_transition(rows, env, split, f"{kind}_{name}_terminal", case_id, traj_id, 0, u, action, reward_weights, 0.0, 1.0, no_resid, no_phys)
                    traj_id += 1
            for start_name, u in starts[:3]:
                append_gt_sequence(rows, env, split, case_id, traj_id, u, reward_weights, seq_steps, no_resid, no_phys, start_name)
                traj_id += 1
            append_refine_sequence(rows, env, split, case_id, traj_id, starts[0][1], reward_weights, seq_steps, no_resid, no_phys, "gradient")
            traj_id += 1
            append_refine_sequence(rows, env, split, case_id, traj_id, starts[1][1], reward_weights, seq_steps, no_resid, no_phys, "pinn")
            traj_id += 1
            append_refine_sequence(rows, env, split, case_id, traj_id, starts[2][1], reward_weights, seq_steps, no_resid, no_phys, "mixed")
            traj_id += 1
        offset += 10000
    assign_discounted_returns(rows, float(cfg["paper"]["gamma"]))
    torch.save(rows, cache)
    return rows


def fit_normalization_stats(transitions: list[Transition]) -> dict:
    train = split_transitions(transitions, "train")
    scalars = torch.stack([tr.state_scalars for tr in train])
    residual = torch.cat([tr.state_fields[1].reshape(-1) for tr in train])
    actions = torch.stack([tr.action for tr in train])
    rewards = torch.tensor([tr.reward for tr in train], dtype=torch.float32)
    returns = torch.tensor([tr.discounted_return for tr in train], dtype=torch.float32)
    stats = {
        "scalar_mean": scalars.mean(0),
        "scalar_std": scalars.std(0).clamp_min(1e-6),
        "residual_scale": torch.sqrt(torch.mean(residual.pow(2))).clamp_min(1e-6),
        "action_mean": actions.mean(0),
        "action_std": actions.std(0).clamp_min(1e-6),
        "reward_mean": rewards.mean(),
        "reward_std": rewards.std().clamp_min(1e-6),
        "return_mean": returns.mean(),
        "return_std": returns.std().clamp_min(1e-6),
    }
    ckpt = ROOT / "checkpoints"
    torch.save({"mean": stats["scalar_mean"], "std": stats["scalar_std"], "residual_scale": stats["residual_scale"]}, ckpt / "state_scalar_stats.pt")
    torch.save({"mean": stats["action_mean"], "std": stats["action_std"]}, ckpt / "action_stats.pt")
    torch.save({"reward_mean": stats["reward_mean"], "reward_std": stats["reward_std"], "return_mean": stats["return_mean"], "return_std": stats["return_std"]}, ckpt / "reward_stats.pt")
    return stats


def normalize_transitions(transitions: list[Transition], stats: dict) -> list[Transition]:
    out = []
    for tr in transitions:
        sf, ss = normalize_state(tr.state_fields.unsqueeze(0), tr.state_scalars.unsqueeze(0), stats)
        nsf, nss = normalize_state(tr.next_fields.unsqueeze(0), tr.next_scalars.unsqueeze(0), stats)
        out.append(
            replace(
                tr,
                state_fields=sf.squeeze(0).detach().cpu(),
                state_scalars=ss.squeeze(0).detach().cpu(),
                action=normalize_action(tr.action, stats).detach().cpu(),
                reward=float((torch.tensor(tr.reward) - stats["reward_mean"]) / stats["reward_std"]),
                discounted_return=float((torch.tensor(tr.discounted_return) - stats["return_mean"]) / stats["return_std"]),
                next_fields=nsf.squeeze(0).detach().cpu(),
                next_scalars=nss.squeeze(0).detach().cpu(),
            )
        )
    return out


def batch_state_embeddings(encoder: OperatorStateEncoder, dataset: TransitionDataset, device: torch.device, batch_size: int):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    states, actions, rewards, returns, next_states, dones = [], [], [], [], [], []
    encoder.eval()
    with torch.no_grad():
        for batch in loader:
            sf = batch["state_fields"].to(device)
            ss = batch["state_scalars"].to(device)
            nsf = batch["next_fields"].to(device)
            nss = batch["next_scalars"].to(device)
            states.append(encoder(sf, ss).detach().cpu())
            next_states.append(encoder(nsf, nss).detach().cpu())
            actions.append(batch["action"])
            rewards.append(batch["reward"])
            returns.append(batch["discounted_return"])
            dones.append(batch["done"])
    return torch.cat(states), torch.cat(actions), torch.cat(rewards), torch.cat(returns), torch.cat(next_states), torch.cat(dones)


def train_shared_encoder_fm(cfg: dict, transitions: list[Transition], device: torch.device) -> tuple[OperatorStateEncoder, FlowPolicy]:
    p = cfg["paper"]
    ckpt = ROOT / "checkpoints" / f"flow_supervised_{PROTOCOL_VERSION}.pt"
    action_dim = 2 * int(p["modes"]) * int(p["modes"]) + 1
    encoder = OperatorStateEncoder(width=int(p["state_width"]), modes=int(p["modes"]), state_dim=int(p["state_dim"])).to(device)
    policy = FlowPolicy(int(p["state_dim"]), action_dim, int(p["hidden"]), int(p["fm_steps"])).to(device)
    if ckpt.exists():
        payload = torch.load(ckpt, map_location=device)
        encoder.load_state_dict(payload["encoder"])
        policy.load_state_dict(payload["policy"])
        encoder.eval()
        policy.eval()
        for par in encoder.parameters():
            par.requires_grad_(False)
        return encoder, policy
    ds = TransitionDataset(split_transitions(transitions, "train"))
    loader = DataLoader(ds, batch_size=int(p["batch_size"]), shuffle=True)
    opt = torch.optim.AdamW(list(encoder.parameters()) + list(policy.parameters()), lr=float(p["lr"]), weight_decay=1e-4)
    for _ in tqdm(range(int(p["fm_pretrain_epochs"])), desc="flow:supervised"):
        for batch in loader:
            state = encoder(batch["state_fields"].to(device), batch["state_scalars"].to(device))
            loss = policy.fm_loss(state, batch["action"].to(device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    torch.save({"encoder": encoder.state_dict(), "policy": policy.state_dict()}, ckpt)
    encoder.eval()
    policy.eval()
    for par in encoder.parameters():
        par.requires_grad_(False)
    return encoder, policy


def train_awfm(cfg: dict, transitions: list[Transition], encoder: OperatorStateEncoder, init_policy: FlowPolicy, critic, value, suffix: str, device: torch.device) -> FlowPolicy:
    p = cfg["paper"]
    ckpt = ROOT / "checkpoints" / f"flow_{suffix}_{PROTOCOL_VERSION}.pt"
    action_dim = 2 * int(p["modes"]) * int(p["modes"]) + 1
    policy = FlowPolicy(int(p["state_dim"]), action_dim, int(p["hidden"]), int(p["fm_steps"])).to(device)
    if ckpt.exists():
        policy.load_state_dict(torch.load(ckpt, map_location=device)["policy"])
        return policy.eval()
    policy.load_state_dict(init_policy.state_dict())
    ds = TransitionDataset(split_transitions(transitions, "train"))
    loader = DataLoader(ds, batch_size=int(p["batch_size"]), shuffle=True)
    opt = torch.optim.AdamW(policy.parameters(), lr=float(p["lr"]), weight_decay=1e-4)
    encoder.eval()
    for _ in tqdm(range(int(p["awfm_epochs"])), desc=f"flow:{suffix}"):
        for batch in loader:
            sf = batch["state_fields"].to(device)
            ss = batch["state_scalars"].to(device)
            a = batch["action"].to(device)
            with torch.no_grad():
                state = encoder(sf, ss)
                adv = critic.q_min(state, a) - value(state)
                weight = torch.exp(adv / float(p["aw_beta"])).clamp(0.05, float(p["aw_wmax"]))
            loss = policy.fm_loss(state, a, weight)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    torch.save({"policy": policy.state_dict()}, ckpt)
    return policy.eval()


def train_deterministic(cfg: dict, transitions: list[Transition], encoder: OperatorStateEncoder, device: torch.device) -> DeterministicCorrection:
    p = cfg["paper"]
    ckpt = ROOT / "checkpoints" / f"deterministic_{PROTOCOL_VERSION}.pt"
    action_dim = 2 * int(p["modes"]) * int(p["modes"]) + 1
    model = DeterministicCorrection(int(p["state_dim"]), action_dim, int(p["hidden"])).to(device)
    if ckpt.exists():
        model.load_state_dict(torch.load(ckpt, map_location=device)["model"])
        return model.eval()
    ds = TransitionDataset([tr for tr in split_transitions(transitions, "train") if "good" in tr.kind or tr.kind.startswith("seq_")])
    loader = DataLoader(ds, batch_size=int(p["batch_size"]), shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=float(p["lr"]))
    encoder.eval()
    for _ in tqdm(range(int(p["deterministic_epochs"])), desc="deterministic"):
        for batch in loader:
            with torch.no_grad():
                z = encoder(batch["state_fields"].to(device), batch["state_scalars"].to(device))
            pred = model(z)
            loss = F.mse_loss(pred, batch["action"].to(device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    torch.save({"model": model.state_dict()}, ckpt)
    return model.eval()


def calibrate_stop_margin(cfg: dict, transitions: list[Transition], encoder: OperatorStateEncoder, critic, value, device: torch.device) -> tuple[float, float]:
    p = cfg["paper"]
    val = [tr for tr in split_transitions(transitions, "val") if tr.kind.startswith("seq_")]
    if len(val) == 0:
        return 0.0, 0.0
    s, a, _, ret, _, _ = batch_state_embeddings(encoder, TransitionDataset(val), device, int(p["batch_size"]))
    with torch.no_grad():
        adv = (critic.q_min(s.to(device), a.to(device)) - value(s.to(device))).detach().cpu()
    targets = ret > torch.quantile(ret, 0.55)
    best_margin, best_score, best_stop = float(torch.quantile(adv, 0.25)), -1e9, 0.0
    for q in torch.linspace(0.05, 0.75, 21):
        margin = float(torch.quantile(adv, float(q)))
        execute = adv > margin
        tp = torch.logical_and(execute, targets).sum().item()
        fp = torch.logical_and(execute, ~targets).sum().item()
        fn = torch.logical_and(~execute, targets).sum().item()
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        stop_rate = float((~execute).float().mean())
        score = precision + 0.5 * recall - 0.15 * abs(stop_rate - 0.25)
        if score > best_score:
            best_margin, best_score, best_stop = margin, score, stop_rate
    return best_margin, best_stop


def train_iql(cfg: dict, transitions: list[Transition], encoder: OperatorStateEncoder, suffix: str, device: torch.device, single_q: bool = False) -> tuple[torch.nn.Module, ValueNet, float, float]:
    p = cfg["paper"]
    ckpt = ROOT / "checkpoints" / f"iql_{suffix}_{PROTOCOL_VERSION}.pt"
    action_dim = 2 * int(p["modes"]) * int(p["modes"]) + 1
    critic = (SingleQCritic if single_q else TwinQCritic)(int(p["state_dim"]), action_dim, int(p["hidden"])).to(device)
    value = ValueNet(int(p["state_dim"]), int(p["hidden"])).to(device)
    if ckpt.exists():
        payload = torch.load(ckpt, map_location=device)
        critic.load_state_dict(payload["critic"])
        value.load_state_dict(payload["value"])
        return critic.eval(), value.eval(), float(payload["stop_margin"]), float(payload.get("val_stop_rate", 0.0))
    train_ds = TransitionDataset(split_transitions(transitions, "train"))
    s, a, r, _, ns, d = batch_state_embeddings(encoder, train_ds, device, int(p["batch_size"]))
    s, a, r, ns, d = s.to(device), a.to(device), r.to(device), ns.to(device), d.to(device)
    opt = torch.optim.AdamW(list(critic.parameters()) + list(value.parameters()), lr=float(p["lr"]), weight_decay=1e-4)
    n = s.shape[0]
    bs = int(p["batch_size"])
    for _ in tqdm(range(int(p["critic_epochs"])), desc=f"iql:{suffix}"):
        perm = torch.randperm(n, device=device)
        for j in range(0, n, bs):
            idx = perm[j:j + bs]
            q_loss, v_loss = iql_losses(critic, value, s[idx], a[idx], r[idx], ns[idx], d[idx], float(p["gamma"]), float(p["expectile"]))
            loss = q_loss + v_loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    stop_margin, val_stop_rate = calibrate_stop_margin(cfg, transitions, encoder, critic, value, device)
    torch.save({"critic": critic.state_dict(), "value": value.state_dict(), "stop_margin": stop_margin, "val_stop_rate": val_stop_rate}, ckpt)
    return critic.eval(), value.eval(), stop_margin, val_stop_rate


def calibrate_policy_stop_margin(cfg: dict, npz_path: Path, lno, encoder, policy, critic, value, stats: dict, device: torch.device) -> tuple[float, float]:
    p = cfg["paper"]
    env = env_from_cfg(cfg, device)
    val = ReactionDiffusionDataset(npz_path, "val", min(int(p["val_cases"]), 16))
    pairs = []
    for i in range(len(val)):
        case = as_case(val[i], i)
        env.reset(case)
        with torch.no_grad():
            u = env.project(lno(case["source"].to(device).unsqueeze(0)).squeeze(0))
            e0 = float(env.relative_error(u).detach().cpu())
            fields, scalars = make_state_normalized(env, u, 0.0, stats)
            z = encoder(fields.to(device), scalars.to(device))
            acts_norm = policy.sample(z, int(p["candidates"]))[0]
            z_rep = z.repeat(acts_norm.shape[0], 1)
            q = critic.q_min(z_rep, acts_norm)
            adv = q - value(z).repeat(acts_norm.shape[0])
            best = int(torch.argmax(q))
            e1 = float(env.relative_error(env.step(u, denormalize_action(acts_norm[best].detach(), stats))).detach().cpu())
            pairs.append((float(adv.max().detach().cpu()), e0 - e1))
    advs = torch.tensor([p[0] for p in pairs], dtype=torch.float32)
    improvements = torch.tensor([p[1] for p in pairs], dtype=torch.float32)
    best_margin, best_score, best_stop = float(torch.quantile(advs, 0.50)), -1e9, 0.0
    for q in torch.linspace(0.05, 0.95, 37):
        margin = float(torch.quantile(advs, float(q)))
        execute = advs > margin
        if float(execute.float().mean()) < 0.60:
            continue
        score = torch.where(execute, improvements, torch.zeros_like(improvements)).mean().item()
        score -= 0.001 * float(execute.float().mean())
        if score > best_score:
            best_margin = margin
            best_score = score
            best_stop = float((~execute).float().mean())
    return best_margin, best_stop


def policy_rollout(env, encoder, policy, critic, value, u0: torch.Tensor, max_steps: int, cfg: dict, stats: dict, stop_margin: float, no_residual: bool = False, use_q_selection: bool = True) -> tuple[torch.Tensor, dict]:
    p = cfg["paper"]
    u = u0.detach()
    errors, energies, advantages = [], [], []
    steps = 0
    stopped = False
    start = time.perf_counter()
    device = next(encoder.parameters()).device
    for k in range(max_steps):
        fields, scalars = make_state_normalized(env, u, k / max(1, max_steps), stats, no_residual=no_residual)
        with torch.no_grad():
            z = encoder(fields.to(device), scalars.to(device))
            acts_norm = policy.sample(z, int(p["candidates"]))[0]
            if use_q_selection:
                z_rep = z.repeat(acts_norm.shape[0], 1)
                q = critic.q_min(z_rep, acts_norm)
                v = value(z).repeat(acts_norm.shape[0])
                adv = q - v
                best = int(torch.argmax(q))
                best_adv = float(adv[best].detach().cpu())
                max_adv = float(adv.max().detach().cpu())
            else:
                best = 0
                best_adv = 0.0
                max_adv = 1.0
            advantages.append(best_adv)
            if use_q_selection and max_adv <= stop_margin:
                stopped = True
                break
            action = denormalize_action(acts_norm[best].detach(), stats)
        u = env.step(u, action)
        steps += 1
        errors.append(float(env.relative_error(u).detach().cpu()))
        energies.append(float(env.physics_energy(u)["energy"].detach().cpu()))
    return u, {"wall_time": time.perf_counter() - start, "steps": steps, "stopped": stopped, "errors": errors, "energies": energies, "advantages": advantages}


def add_policy_rollout_transitions(cfg: dict, npz_path: Path, lno, transitions: list[Transition], encoder, policy, critic, value, stats: dict, device: torch.device, round_idx: int) -> list[Transition]:
    p = cfg["paper"]
    env = env_from_cfg(cfg, device)
    ds = ReactionDiffusionDataset(npz_path, "train", int(p["rollout_add_cases"]))
    rows = list(transitions)
    reward_weights = dict(cfg["reward"])
    traj_start = max([tr.trajectory_id for tr in rows], default=0) + 1
    for i in tqdm(range(len(ds)), desc=f"rollout-buffer:{round_idx}"):
        case_id = 50000 + 1000 * round_idx + i
        case = as_case(ds[i], case_id)
        env.reset(case)
        with torch.no_grad():
            u = env.project(lno(case["source"].to(device).unsqueeze(0)).squeeze(0))
        traj_id = traj_start + i
        for k in range(min(3, max(p["rollout_steps"]))):
            fields, scalars = make_state_normalized(env, u, k / max(1, max(p["rollout_steps"])), stats)
            with torch.no_grad():
                z = encoder(fields.to(device), scalars.to(device))
                acts_norm = policy.sample(z, int(p["candidates"]))[0]
                z_rep = z.repeat(acts_norm.shape[0], 1)
                action_norm = acts_norm[int(torch.argmax(critic.q_min(z_rep, acts_norm)))].detach()
                action = denormalize_action(action_norm, stats)
            u = append_transition(rows, env, "train", f"seq_policy_rollout_r{round_idx}", case_id, traj_id, k, u, action, reward_weights, k / max(1, max(p["rollout_steps"])), float(k == 2))
    assign_discounted_returns(rows, float(p["gamma"]))
    return rows


def supervised_fm_rollout(env, encoder, policy, u0: torch.Tensor, max_steps: int, cfg: dict, stats: dict, selection: str = "single") -> tuple[torch.Tensor, dict]:
    p = cfg["paper"]
    u = u0.detach()
    start = time.perf_counter()
    errors, energies = [], []
    device = next(encoder.parameters()).device
    for k in range(max_steps):
        fields, scalars = make_state_normalized(env, u, k / max(1, max_steps), stats)
        with torch.no_grad():
            z = encoder(fields.to(device), scalars.to(device))
            acts_norm = policy.sample(z, int(p["candidates"]))[0]
            if selection == "physics":
                candidates = [env.step(u, denormalize_action(a.detach(), stats)) for a in acts_norm]
                scores = [float(env.physics_energy(c)["energy"].detach().cpu()) for c in candidates]
                u = candidates[int(np.argmin(scores))]
            else:
                u = env.step(u, denormalize_action(acts_norm[0].detach(), stats))
        errors.append(float(env.relative_error(u).detach().cpu()))
        energies.append(float(env.physics_energy(u)["energy"].detach().cpu()))
    return u, {"wall_time": time.perf_counter() - start, "steps": max_steps, "stopped": False, "errors": errors, "energies": energies}


def deterministic_rollout(env, encoder, model, u0: torch.Tensor, max_steps: int, stats: dict) -> tuple[torch.Tensor, dict]:
    u = u0.detach()
    start = time.perf_counter()
    device = next(encoder.parameters()).device
    for k in range(max_steps):
        fields, scalars = make_state_normalized(env, u, k / max(1, max_steps), stats)
        with torch.no_grad():
            z = encoder(fields.to(device), scalars.to(device))
            a = denormalize_action(model(z)[0].detach(), stats)
        u = env.step(u, a)
    return u, {"wall_time": time.perf_counter() - start, "steps": max_steps, "stopped": False}


def summarize(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    metrics = ["Relative L2", "PDE residual norm", "BC error", "IC error", "Physics energy", "Wall time", "Refinement steps", "STOP", "Failure", "Per-step improvement"]
    rows = []
    for group, g in df.groupby(keys, dropna=False):
        if not isinstance(group, tuple):
            group = (group,)
        row = dict(zip(keys, group))
        for m in metrics:
            if m in g:
                row[m] = mean_std(g[m].astype(float).tolist())
        row["Samples"] = len(g)
        rows.append(row)
    return pd.DataFrame(rows)


def critic_diagnostics(cfg: dict, transitions_norm: list[Transition], encoder, critic, value, device: torch.device) -> pd.DataFrame:
    held = split_transitions(transitions_norm, "test")
    s, a, r, ret, _, _ = batch_state_embeddings(encoder, TransitionDataset(held), device, int(cfg["paper"]["batch_size"]))
    with torch.no_grad():
        q = critic.q_min(s.to(device), a.to(device)).detach().cpu().numpy()
        v = value(s.to(device)).detach().cpu().numpy()
    p_qi, s_qi = corr(q, r.numpy())
    p_ai, s_ai = corr(q - v, r.numpy())
    p_qr, s_qr = corr(q, ret.numpy())
    p_ar, s_ar = corr(q - v, ret.numpy())
    return pd.DataFrame(
        [
            {
                "Q Immediate Pearson": p_qi,
                "Q Immediate Spearman": s_qi,
                "Advantage Immediate Pearson": p_ai,
                "Advantage Immediate Spearman": s_ai,
                "Q Return Pearson": p_qr,
                "Q Return Spearman": s_qr,
                "Advantage Return Pearson": p_ar,
                "Advantage Return Spearman": s_ar,
                "Samples": len(held),
            }
        ]
    )


def residual_error_quadrants(transitions: list[Transition]) -> pd.DataFrame:
    rows = []
    for split in ["train", "val", "test"]:
        sub = split_transitions(transitions, split)
        for name, pred in [
            ("physics better / accuracy better", lambda tr: tr.energy_after < tr.energy_before and tr.error_after < tr.error_before),
            ("physics better / accuracy worse", lambda tr: tr.energy_after < tr.energy_before and tr.error_after >= tr.error_before),
            ("physics worse / accuracy better", lambda tr: tr.energy_after >= tr.energy_before and tr.error_after < tr.error_before),
            ("both worse", lambda tr: tr.energy_after >= tr.energy_before and tr.error_after >= tr.error_before),
        ]:
            count = sum(1 for tr in sub if pred(tr))
            rows.append({"Split": split, "Quadrant": name, "Count": count, "Fraction": count / max(1, len(sub))})
    return pd.DataFrame(rows)


def codec_diagnostics(cfg: dict, npz_path: Path, lno, device: torch.device) -> pd.DataFrame:
    env = env_from_cfg(cfg, device)
    ds = ReactionDiffusionDataset(npz_path, "test", min(8, int(cfg["paper"]["eval_cases"])))
    rows = []
    for i in range(len(ds)):
        case = as_case(ds[i], i)
        env.reset(case)
        with torch.no_grad():
            u = env.project(lno(case["source"].to(device).unsqueeze(0)).squeeze(0))
        delta = case["gt"].to(device) - u
        low = env.lowpass(delta)
        recon = env.decode_action(env.encode_correction(delta), tuple(delta.shape))
        rows.append(
            {
                "Case": i,
                "Codec reconstruction error": float(torch.norm(recon - low) / torch.norm(low).clamp_min(1e-8)),
                "Spectral energy retention": float(low.pow(2).sum() / delta.pow(2).sum().clamp_min(1e-8)),
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "results" / "tables" / "codec_diagnostics.csv", index=False)
    return df


def evaluate_all(cfg: dict, npz_path: Path, lno, encoder, fm_sup, fm_rl, critic, value, stop_margin, det_model, stats: dict, transitions_raw, transitions_norm, device: torch.device) -> dict[str, pd.DataFrame]:
    p = cfg["paper"]
    env = env_from_cfg(cfg, device)
    test = ReactionDiffusionDataset(npz_path, "test", int(p["eval_cases"]))
    main_rows, conv_rows, init_rows, dist_rows = [], [], [], []
    methods = ["LNO only", "Gradient", "PINN", "Deterministic", "Supervised FM single", "Supervised FM + physics selection", "RL-FM", "RL-FM without Q selection", "Coarse + RL-FM"]
    for steps in tqdm([1, 2, 5, 10], desc="evaluate"):
        per_method = {m: [] for m in methods}
        for i in range(len(test)):
            case = as_case(test[i], i)
            env.reset(case)
            with torch.no_grad():
                lno_u = env.project(lno(case["source"].to(device).unsqueeze(0)).squeeze(0))
            coarse = env.coarse_initialization()
            gt = case["gt"].to(device)
            base_err = float(env.relative_error(lno_u).detach().cpu())
            candidates = {
                "LNO only": (lno_u, {"wall_time": 0.0, "steps": 0, "stopped": True}),
                "Gradient": gradient_residual_refine(env, lno_u, steps, lr=0.015)[:2],
                "PINN": pinn_style_refine(env, lno_u, steps, lr=0.012)[:2],
                "Deterministic": deterministic_rollout(env, encoder, det_model, lno_u, steps, stats),
                "Supervised FM single": supervised_fm_rollout(env, encoder, fm_sup, lno_u, steps, cfg, stats, "single"),
                "Supervised FM + physics selection": supervised_fm_rollout(env, encoder, fm_sup, lno_u, steps, cfg, stats, "physics"),
                "RL-FM": policy_rollout(env, encoder, fm_rl, critic, value, lno_u, steps, cfg, stats, stop_margin, use_q_selection=True),
                "RL-FM without Q selection": policy_rollout(env, encoder, fm_rl, critic, value, lno_u, steps, cfg, stats, stop_margin, use_q_selection=False),
                "Coarse + RL-FM": policy_rollout(env, encoder, fm_rl, critic, value, coarse, steps, cfg, stats, stop_margin, use_q_selection=True),
            }
            for name, payload in candidates.items():
                if isinstance(payload[1], float):
                    u, wall = payload
                    meta = {"wall_time": wall, "steps": steps, "stopped": False}
                else:
                    u, meta = payload
                phys = env.physics_energy(u)
                err = float(env.relative_error(u).detach().cpu())
                row = {
                    "Method": name,
                    "Steps": steps,
                    "Relative L2": err,
                    "PDE residual norm": float(phys["residual_norm"].detach().cpu()),
                    "BC error": float(phys["bc_error"].detach().cpu()),
                    "IC error": float(phys["ic_error"].detach().cpu()),
                    "Physics energy": float(phys["energy"].detach().cpu()),
                    "Wall time": float(meta["wall_time"]),
                    "Refinement steps": int(meta["steps"]),
                    "STOP": float(meta.get("stopped", False)),
                    "Failure": float(not torch.isfinite(u).all() or err > 10.0),
                    "Per-step improvement": (base_err - err) / max(1, steps),
                }
                per_method[name].append(row)
                if name in ["Gradient", "PINN", "Supervised FM single", "Supervised FM + physics selection", "RL-FM"]:
                    conv_rows.append(row | {"Case": i})
                if steps == 10 and name in ["LNO only", "RL-FM", "Coarse + RL-FM"]:
                    init_rows.append(row | {"Initial quality": "coarse" if "Coarse" in name else "lno"})
            dist_rows.append(
                {
                    "Case": i,
                    "LNO error": base_err,
                    "Supervised FM single improvement": base_err - float((candidates["Supervised FM single"][0].detach() - gt).norm() / gt.norm().clamp_min(1e-8)),
                    "RL-FM improvement": base_err - float((candidates["RL-FM"][0].detach() - gt).norm() / gt.norm().clamp_min(1e-8)),
                }
            )
        for rows in per_method.values():
            main_rows.extend(rows)
    return {
        "raw_main": pd.DataFrame(main_rows),
        "main": summarize(pd.DataFrame(main_rows), ["Method", "Steps"]),
        "convergence": summarize(pd.DataFrame(conv_rows), ["Method", "Steps"]),
        "initialization": summarize(pd.DataFrame(init_rows), ["Method", "Initial quality"]),
        "critic": critic_diagnostics(cfg, transitions_norm, encoder, critic, value, device),
        "improvement_distribution": pd.DataFrame(dist_rows),
        "residual_error_quadrants": residual_error_quadrants(transitions_raw),
    }


def run_ablations(cfg: dict, npz_path: Path, lno, transitions_norm, encoder, fm_sup, fm_rl, critic, value, stop_margin, det_model, stats, device: torch.device) -> pd.DataFrame:
    rows = []
    test = ReactionDiffusionDataset(npz_path, "test", min(12, int(cfg["paper"]["eval_cases"])))
    env = env_from_cfg(cfg, device)
    variants = [
        ("Full RL-FM", "rl_q"),
        ("without candidate Q selection", "rl_single"),
        ("without RL weighting = Supervised FM", "sup"),
        ("single Q instead of Twin-Q", "single_q"),
        ("deterministic correction instead of Flow Matching", "det"),
    ]
    single_q_critic, single_q_value, single_q_margin, _ = train_iql(cfg, transitions_norm, encoder, "singleq_ablation", device, single_q=True)
    for name, mode in tqdm(variants, desc="ablation"):
        for i in range(len(test)):
            case = as_case(test[i], i)
            env.reset(case)
            with torch.no_grad():
                lno_u = env.project(lno(case["source"].to(device).unsqueeze(0)).squeeze(0))
            if mode == "rl_q":
                pred, meta = policy_rollout(env, encoder, fm_rl, critic, value, lno_u, 5, cfg, stats, stop_margin, use_q_selection=True)
            elif mode == "rl_single":
                pred, meta = policy_rollout(env, encoder, fm_rl, critic, value, lno_u, 5, cfg, stats, stop_margin, use_q_selection=False)
            elif mode == "sup":
                pred, meta = supervised_fm_rollout(env, encoder, fm_sup, lno_u, 5, cfg, stats, "single")
            elif mode == "single_q":
                pred, meta = policy_rollout(env, encoder, fm_rl, single_q_critic, single_q_value, lno_u, 5, cfg, stats, single_q_margin, use_q_selection=True)
            else:
                pred, meta = deterministic_rollout(env, encoder, det_model, lno_u, 5, stats)
            phys = env.physics_energy(pred)
            rows.append(
                {
                    "Ablation": name,
                    "Relative L2": float(env.relative_error(pred).detach().cpu()),
                    "Physics energy": float(phys["energy"].detach().cpu()),
                    "Wall time": meta["wall_time"],
                    "STOP": float(meta.get("stopped", False)),
                    "Refinement steps": meta["steps"],
                    "Failure": 0.0,
                    "Per-step improvement": 0.0,
                    "PDE residual norm": float(phys["residual_norm"].detach().cpu()),
                    "BC error": float(phys["bc_error"].detach().cpu()),
                    "IC error": float(phys["ic_error"].detach().cpu()),
                }
            )
    return summarize(pd.DataFrame(rows), ["Ablation"])


def save_tables(results: dict[str, pd.DataFrame], ablation: pd.DataFrame) -> None:
    table_dir = ROOT / "results" / "tables"
    results["main"].to_csv(table_dir / "main_results.csv", index=False)
    results["convergence"].to_csv(table_dir / "convergence.csv", index=False)
    results["critic"].to_csv(table_dir / "critic_results.csv", index=False)
    results["initialization"].to_csv(table_dir / "initialization_results.csv", index=False)
    results["improvement_distribution"].to_csv(table_dir / "improvement_distribution.csv", index=False)
    results["residual_error_quadrants"].to_csv(table_dir / "residual_error_quadrants.csv", index=False)
    ablation.to_csv(table_dir / "ablation.csv", index=False)


def generate_figures(results: dict[str, pd.DataFrame], transitions_raw: list[Transition]) -> None:
    fig_dir = ROOT / "results" / "figures"
    raw = results["raw_main"]
    conv = raw[raw["Method"].isin(["Gradient", "PINN", "Supervised FM single", "Supervised FM + physics selection", "RL-FM"])]

    def line(metric: str, path: str):
        plt.figure(figsize=(6, 4))
        for name, g in conv.groupby("Method"):
            gg = g.groupby("Steps")[metric].mean()
            plt.plot(gg.index, gg.values, marker="o", label=name)
        plt.xlabel("Refinement iterations")
        plt.ylabel(metric)
        plt.legend()
        plt.tight_layout()
        plt.savefig(fig_dir / path, dpi=180)
        plt.close()

    plt.figure(figsize=(8, 2.4))
    blocks = ["PDE case", "Shared state encoder", "Flow policy", "IQL Q/V critic", "Candidate selection", "Refined PDE"]
    for i, b in enumerate(blocks):
        plt.text(i, 0.5, b, ha="center", va="center", bbox=dict(boxstyle="round,pad=0.3", fc="#edf2f7", ec="#2d3748"))
        if i < len(blocks) - 1:
            plt.arrow(i + 0.35, 0.5, 0.28, 0, head_width=0.04, color="#2d3748")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(fig_dir / "figure1_architecture.png", dpi=180)
    plt.close()
    line("Relative L2", "figure3_iteration_vs_l2.png")
    line("Physics energy", "figure4_iteration_vs_energy.png")
    plt.figure(figsize=(6, 4))
    for name, g in conv.groupby("Method"):
        plt.scatter(g["Wall time"], g["Relative L2"], s=20, label=name, alpha=0.7)
    plt.xlabel("Wall time (s)")
    plt.ylabel("Relative L2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "figure5_walltime_vs_l2.png", dpi=180)
    plt.close()
    dist = results["improvement_distribution"]
    plt.figure(figsize=(5, 4))
    plt.hist(dist["Supervised FM single improvement"], bins=16, alpha=0.6, label="Supervised FM single")
    plt.hist(dist["RL-FM improvement"], bins=16, alpha=0.6, label="RL-FM")
    plt.xlabel("Improvement over LNO")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "figure6_fm_vs_rlfm_distribution.png", dpi=180)
    plt.close()
    crit = results["critic"]
    plt.figure(figsize=(4.5, 3.5))
    vals = [float(crit["Q Return Spearman"].iloc[0]), float(crit["Advantage Return Spearman"].iloc[0])]
    plt.bar(["Q", "Advantage"], vals)
    plt.ylabel("Spearman with discounted return")
    plt.tight_layout()
    plt.savefig(fig_dir / "figure7_critic_return_correlation.png", dpi=180)
    plt.close()
    sample = [tr for tr in transitions_raw if tr.split == "test"][:1200]
    plt.figure(figsize=(5.5, 4))
    x = [tr.energy_after - tr.energy_before for tr in sample]
    y = [tr.error_after - tr.error_before for tr in sample]
    plt.axhline(0, color="black", linewidth=0.8)
    plt.axvline(0, color="black", linewidth=0.8)
    plt.scatter(x, y, s=8, alpha=0.45)
    plt.xlabel("Delta physics energy")
    plt.ylabel("Delta ground-truth error")
    plt.tight_layout()
    plt.savefig(fig_dir / "figure8_residual_error_quadrants.png", dpi=180)
    plt.close()
    init = raw[raw["Method"].isin(["LNO only", "RL-FM", "Coarse + RL-FM"])]
    plt.figure(figsize=(6, 4))
    for name, g in init.groupby("Method"):
        gg = g.groupby("Steps")["Relative L2"].mean()
        plt.plot(gg.index, gg.values, marker="o", label=name)
    plt.xlabel("Refinement iterations")
    plt.ylabel("Relative L2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "figure9_initial_quality_convergence.png", dpi=180)
    plt.close()


def write_docs(results: dict[str, pd.DataFrame]) -> None:
    docs = ROOT / "docs"
    main = results["main"]

    def table(name: str) -> str:
        path = ROOT / "results" / "tables" / name
        return pd.read_csv(path).to_markdown(index=False) if path.exists() else "not generated"

    lno10 = main[(main.Method == "LNO only") & (main.Steps == 10)]["Relative L2"].iloc[0]
    sup10 = main[(main.Method == "Supervised FM single") & (main.Steps == 10)]["Relative L2"].iloc[0]
    rl10 = main[(main.Method == "RL-FM") & (main.Steps == 10)]["Relative L2"].iloc[0]
    grad10 = main[(main.Method == "Gradient") & (main.Steps == 10)]["Relative L2"].iloc[0]
    critic = pd.read_csv(ROOT / "results" / "tables" / "critic_results.csv")
    claims = ["# Paper Claims", ""]
    if rl10 < sup10 and rl10 < lno10:
        claims.append(f"- RL-guided Flow Matching improves over both supervised FM and LNO at 10 steps: RL-FM {rl10}, Supervised FM {sup10}, LNO {lno10}.")
    else:
        claims.append(f"- RL-guided Flow Matching does not support a broad improvement claim at 10 steps: RL-FM {rl10}, Supervised FM {sup10}, LNO {lno10}.")
    if grad10 < rl10:
        claims.append(f"- The gradient residual baseline remains stronger than RL-FM at 10 steps in this run: Gradient {grad10}, RL-FM {rl10}.")
    claims.append(f"- Critic evaluation is reported on held-out transitions with Q Return Spearman = {critic['Q Return Spearman'].iloc[0]:.4f} and Advantage Return Spearman = {critic['Advantage Return Spearman'].iloc[0]:.4f}.")
    claims.append("- The final protocol uses one shared frozen state encoder after supervised FM, normalized actions, train-only state/reward statistics, sequential transition trajectories, and validation-calibrated STOP.")
    claims.append("- GT is used for offline transition reward construction and evaluation only, not policy/critic inference inputs.")
    (docs / "paper_claims.md").write_text("\n".join(claims) + "\n", encoding="utf-8")
    sections = [
        "# Results",
        "",
        "Interpretation:",
        "This paper-mode run reports the measured outcome without promoting unsupported gains. The protocol fixes action-codec symmetry, shared-latent training, normalized offline RL inputs, sequential transition returns, candidate-Q selection, and validation-calibrated STOP.",
        "",
        "Main results:",
        table("main_results.csv"),
        "",
        "Critic:",
        table("critic_results.csv"),
        "",
        "Convergence:",
        table("convergence.csv"),
        "",
        "Ablation:",
        table("ablation.csv"),
        "",
        "Residual/error quadrants:",
        table("residual_error_quadrants.csv"),
        "",
        "Codec diagnostics:",
        table("codec_diagnostics.csv"),
    ]
    (docs / "paper_results.md").write_text("\n\n".join(sections) + "\n", encoding="utf-8")
    (docs / "method.md").write_text(
        "# Method\n\nRL-Flow-PDE refines an initial reaction-diffusion prediction with a residual-conditioned Flow Matching correction policy. Supervised FM trains a shared state encoder and initial policy on low-frequency spectral corrections. The encoder is then frozen and reused by the critic, value network, AWFM policy, deterministic baseline, rollouts, and evaluation. Offline trajectories include terminal contrastive corrections and multi-step coarse-to-improved sequences with discounted returns. At inference, policy samples normalized actions, the critic ranks normalized candidates, and actions are denormalized only before PDE environment stepping.\n",
        encoding="utf-8",
    )
    (docs / "limitations.md").write_text(
        "# Limitations\n\nThe system is intentionally compact enough for a local 8 GB GPU. It uses the official LNO 2D_Reac_diffusion data when available and reports negative results when RL guidance does not beat supervised FM, LNO-only, or gradient refinement. Residual reduction and ground-truth accuracy improvement can disagree, so the residual/error quadrant table is part of the final evidence rather than a diagnostic afterthought.\n",
        encoding="utf-8",
    )


def run_pipeline(mode: str = "paper") -> None:
    cfg = load_config()
    ensure_dirs()
    set_seed(int(cfg["seed"]))
    device = get_device(str(cfg.get("device", "cuda")))
    npz_path = prepare_reaction_diffusion(cfg, ROOT)
    lno = train_lno(cfg, npz_path, device)
    transitions_raw = build_transition_buffer(cfg, npz_path, lno, device)
    stats = fit_normalization_stats(transitions_raw)
    transitions_norm = normalize_transitions(transitions_raw, stats)
    codec_df = codec_diagnostics(cfg, npz_path, lno, device)
    encoder, fm_sup = train_shared_encoder_fm(cfg, transitions_norm, device)
    critic, value, stop_margin, val_stop_rate = train_iql(cfg, transitions_norm, encoder, "main_twinq", device, single_q=False)
    fm_rl = train_awfm(cfg, transitions_norm, encoder, fm_sup, critic, value, "rl_guided", device)
    stop_margin, val_stop_rate = calibrate_policy_stop_margin(cfg, npz_path, lno, encoder, fm_rl, critic, value, stats, device)
    for round_idx in range(int(cfg["paper"]["policy_improvement_rounds"])):
        transitions_raw = add_policy_rollout_transitions(cfg, npz_path, lno, transitions_raw, encoder, fm_rl, critic, value, stats, device, round_idx + 1)
        transitions_norm = normalize_transitions(transitions_raw, stats)
        critic, value, stop_margin, val_stop_rate = train_iql(cfg, transitions_norm, encoder, f"main_twinq_round{round_idx + 1}", device, single_q=False)
        fm_rl = train_awfm(cfg, transitions_norm, encoder, fm_rl, critic, value, f"rl_guided_round{round_idx + 1}", device)
        stop_margin, val_stop_rate = calibrate_policy_stop_margin(cfg, npz_path, lno, encoder, fm_rl, critic, value, stats, device)
    det_model = train_deterministic(cfg, transitions_norm, encoder, device)
    results = evaluate_all(cfg, npz_path, lno, encoder, fm_sup, fm_rl, critic, value, stop_margin, det_model, stats, transitions_raw, transitions_norm, device)
    ablation = run_ablations(cfg, npz_path, lno, transitions_norm, encoder, fm_sup, fm_rl, critic, value, stop_margin, det_model, stats, device)
    save_tables(results, ablation)
    generate_figures(results, transitions_raw)
    write_docs(results)
    main = results["main"]
    critic_df = results["critic"]
    summary = {
        "mode": mode,
        "device": str(device),
        "prepared_data": str(npz_path),
        "protocol_version": PROTOCOL_VERSION,
        "stop_margin": stop_margin,
        "validation_stop_rate": val_stop_rate,
        "codec_reconstruction_error": float(codec_df["Codec reconstruction error"].mean()),
        "spectral_energy_retention": float(codec_df["Spectral energy retention"].mean()),
        "tables": sorted(p.name for p in (ROOT / "results" / "tables").glob("*.csv")),
    }
    (ROOT / "results" / "tables" / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    def get(method: str, steps: int, col: str) -> str:
        return str(main[(main.Method == method) & (main.Steps == steps)][col].iloc[0])

    print("Pipeline completed.")
    print(f"LNO L2: {get('LNO only', 10, 'Relative L2')}")
    print(f"Supervised FM 10-step L2: {get('Supervised FM single', 10, 'Relative L2')}")
    print(f"RL-FM 10-step L2: {get('RL-FM', 10, 'Relative L2')}")
    print(f"Gradient 10-step L2: {get('Gradient', 10, 'Relative L2')}")
    print(f"Q Return Spearman: {critic_df['Q Return Spearman'].iloc[0]:.6f}")
    print(f"Advantage Return Spearman: {critic_df['Advantage Return Spearman'].iloc[0]:.6f}")
    print(f"RL-FM STOP rate: {get('RL-FM', 10, 'STOP')}")
    print(f"Codec reconstruction error: {summary['codec_reconstruction_error']:.8f}")
    print(f"Spectral energy retention: {summary['spectral_energy_retention']:.6f}")
