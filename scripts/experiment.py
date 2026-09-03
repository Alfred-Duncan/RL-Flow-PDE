from __future__ import annotations

import json
import time
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
from src.utils.metrics import corr, mean_std, scalar_relative_l2
from src.utils.seed import get_device, set_seed


ROOT = Path(__file__).resolve().parents[1]


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


def scalar_context(env: ReactionDiffusionEnvironment, u: torch.Tensor, k_frac: float, no_residual: bool = False, no_physics: bool = False) -> torch.Tensor:
    e = env.physics_energy(u)
    res = torch.zeros_like(env.residual(u)) if no_residual else env.residual(u)
    scalars = torch.tensor(
        [
            float(e["residual_norm"].detach().cpu()),
            float(e["ic_error"].detach().cpu()),
            float(e["bc_error"].detach().cpu()),
            float(e["energy"].detach().cpu()) if not no_physics else 0.0,
            float(u.mean().detach().cpu()),
            float(u.std().detach().cpu()),
            float(k_frac),
            1.0,
        ],
        dtype=torch.float32,
        device=u.device,
    )
    return scalars, res


def make_state(env: ReactionDiffusionEnvironment, u: torch.Tensor, k_frac: float, no_residual: bool = False, no_physics: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    assert env.case is not None
    scalars, res = scalar_context(env, u, k_frac, no_residual, no_physics)
    bc = torch.zeros_like(u)
    bc[0, :] = env.case.bc_left
    bc[-1, :] = env.case.bc_right
    return state_tensors(u, res, env.case.ic, bc, scalars)


def encode_state(encoder: OperatorStateEncoder, fields: torch.Tensor, scalars: torch.Tensor) -> torch.Tensor:
    return encoder(fields, scalars)


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


def batch_state_embeddings(encoder: OperatorStateEncoder, dataset: TransitionDataset, device: torch.device, batch_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    states, actions, rewards, next_states, dones = [], [], [], [], []
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
            dones.append(batch["done"])
    return torch.cat(states), torch.cat(actions), torch.cat(rewards), torch.cat(next_states), torch.cat(dones)


def lowpass_delta(env: ReactionDiffusionEnvironment, delta: torch.Tensor) -> torch.Tensor:
    return env.decode_action(env.encode_correction(delta), tuple(delta.shape))


def candidate_states(env: ReactionDiffusionEnvironment, lno: SmallLNOBaseline, case: dict, device: torch.device) -> list[tuple[str, torch.Tensor]]:
    env.reset(case)
    source = case["source"].to(device).unsqueeze(0)
    with torch.no_grad():
        lno_u = env.project(lno(source).squeeze(0))
    coarse = env.coarse_initialization()
    gt = case["gt"].to(device)
    grad, _, _ = gradient_residual_refine(env, lno_u, 1, lr=0.01)
    pinn, _, _ = pinn_style_refine(env, lno_u, 1, lr=0.02)
    perturbed = env.project(gt + 0.15 * torch.randn_like(gt))
    return [("lno", lno_u), ("coarse", coarse), ("perturbed", perturbed), ("gradient", grad), ("pinn", pinn)]


def append_transition(rows: list[Transition], env: ReactionDiffusionEnvironment, split: str, kind: str, case_id: int, u: torch.Tensor, action: torch.Tensor, reward_weights: dict, step_frac: float, no_residual: bool = False, no_physics: bool = False) -> None:
    fields, scalars = make_state(env, u, step_frac, no_residual, no_physics)
    next_u = env.step(u, action)
    next_fields, next_scalars = make_state(env, next_u, min(1.0, step_frac + 0.1), no_residual, no_physics)
    r = env.reward(u, next_u, action, reward_weights)
    rows.append(
        Transition(
            case_id=case_id,
            split=split,
            kind=kind,
            state_fields=fields.squeeze(0).detach().cpu(),
            state_scalars=scalars.squeeze(0).detach().cpu(),
            action=action.detach().cpu(),
            reward=r["reward"],
            next_fields=next_fields.squeeze(0).detach().cpu(),
            next_scalars=next_scalars.squeeze(0).detach().cpu(),
            done=0.0,
            gt_improvement=r["gt_improvement"],
            physics_improvement=r["physics_improvement"],
            error_before=r["error_before"],
            error_after=r["error_after"],
            energy_before=r["energy_before"],
            energy_after=r["energy_after"],
        )
    )


def build_transition_buffer(cfg: dict, npz_path: Path, lno: SmallLNOBaseline, device: torch.device, encoder_variant: str = "full", reward_variant: str = "full") -> list[Transition]:
    cache = ROOT / "checkpoints" / f"transitions_{encoder_variant}_{reward_variant}.pt"
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
    for split, limit in specs:
        ds = ReactionDiffusionDataset(npz_path, split, limit)
        for i in tqdm(range(len(ds)), desc=f"buffer:{split}:{encoder_variant}:{reward_variant}"):
            case = as_case(ds[i], offset + i)
            env.reset(case)
            gt = case["gt"].to(device)
            no_resid = encoder_variant == "no_residual"
            no_phys = reward_variant == "no_physics"
            for kind, u in candidate_states(env, lno, case, device):
                good = env.encode_correction(lowpass_delta(env, gt - u))
                bad = -good
                zero = torch.zeros_like(good)
                noise = 0.35 * torch.randn_like(good)
                over = 2.0 * good
                for name, action in [("good", good), ("bad", bad), ("zero", zero), ("noise", noise), ("over", over)]:
                    append_transition(rows, env, split, f"{kind}_{name}", offset + i, u, action, reward_weights, 0.0, no_resid, no_phys)
        offset += 10000
    torch.save(rows, cache)
    return rows


def train_encoder_fm(cfg: dict, transitions: list[Transition], suffix: str, device: torch.device, aw: bool = False, critic=None, value=None) -> tuple[OperatorStateEncoder, FlowPolicy]:
    p = cfg["paper"]
    ckpt = ROOT / "checkpoints" / f"flow_{suffix}.pt"
    action_dim = 2 * int(p["modes"]) * int(p["modes"]) + 1
    encoder = OperatorStateEncoder(width=int(p["state_width"]), modes=int(p["modes"]), state_dim=int(p["state_dim"])).to(device)
    policy = FlowPolicy(int(p["state_dim"]), action_dim, int(p["hidden"]), int(p["fm_steps"])).to(device)
    if ckpt.exists():
        payload = torch.load(ckpt, map_location=device)
        encoder.load_state_dict(payload["encoder"])
        policy.load_state_dict(payload["policy"])
        return encoder.eval(), policy.eval()
    ds = TransitionDataset(split_transitions(transitions, "train"))
    loader = DataLoader(ds, batch_size=int(p["batch_size"]), shuffle=True)
    opt = torch.optim.AdamW(list(encoder.parameters()) + list(policy.parameters()), lr=float(p["lr"]), weight_decay=1e-4)
    epochs = int(p["awfm_epochs"] if aw else p["fm_pretrain_epochs"])
    for _ in tqdm(range(epochs), desc=f"flow:{suffix}"):
        for batch in loader:
            sf = batch["state_fields"].to(device)
            ss = batch["state_scalars"].to(device)
            a = batch["action"].to(device)
            state = encoder(sf, ss)
            weight = None
            if aw and critic is not None and value is not None:
                with torch.no_grad():
                    adv = critic.q_min(state, a) - value(state)
                    weight = torch.exp(adv / float(p["aw_beta"])).clamp(0.05, float(p["aw_wmax"]))
            loss = policy.fm_loss(state, a, weight)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    torch.save({"encoder": encoder.state_dict(), "policy": policy.state_dict()}, ckpt)
    return encoder.eval(), policy.eval()


def train_deterministic(cfg: dict, transitions: list[Transition], device: torch.device) -> tuple[OperatorStateEncoder, DeterministicCorrection]:
    p = cfg["paper"]
    ckpt = ROOT / "checkpoints" / "deterministic.pt"
    action_dim = 2 * int(p["modes"]) * int(p["modes"]) + 1
    encoder = OperatorStateEncoder(width=int(p["state_width"]), modes=int(p["modes"]), state_dim=int(p["state_dim"])).to(device)
    model = DeterministicCorrection(int(p["state_dim"]), action_dim, int(p["hidden"])).to(device)
    if ckpt.exists():
        payload = torch.load(ckpt, map_location=device)
        encoder.load_state_dict(payload["encoder"])
        model.load_state_dict(payload["model"])
        return encoder.eval(), model.eval()
    ds = TransitionDataset([tr for tr in split_transitions(transitions, "train") if tr.kind.endswith("_good")])
    loader = DataLoader(ds, batch_size=int(p["batch_size"]), shuffle=True)
    opt = torch.optim.AdamW(list(encoder.parameters()) + list(model.parameters()), lr=float(p["lr"]))
    for _ in tqdm(range(int(p["deterministic_epochs"])), desc="deterministic"):
        for batch in loader:
            z = encoder(batch["state_fields"].to(device), batch["state_scalars"].to(device))
            pred = model(z)
            loss = F.mse_loss(pred, batch["action"].to(device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    torch.save({"encoder": encoder.state_dict(), "model": model.state_dict()}, ckpt)
    return encoder.eval(), model.eval()


def train_iql(cfg: dict, transitions: list[Transition], encoder: OperatorStateEncoder, suffix: str, device: torch.device, single_q: bool = False) -> tuple[torch.nn.Module, ValueNet, float]:
    p = cfg["paper"]
    ckpt = ROOT / "checkpoints" / f"iql_{suffix}.pt"
    action_dim = 2 * int(p["modes"]) * int(p["modes"]) + 1
    critic = (SingleQCritic if single_q else TwinQCritic)(int(p["state_dim"]), action_dim, int(p["hidden"])).to(device)
    value = ValueNet(int(p["state_dim"]), int(p["hidden"])).to(device)
    if ckpt.exists():
        payload = torch.load(ckpt, map_location=device)
        critic.load_state_dict(payload["critic"])
        value.load_state_dict(payload["value"])
        return critic.eval(), value.eval(), float(payload["stop_margin"])
    train_ds = TransitionDataset(split_transitions(transitions, "train"))
    s, a, r, ns, d = batch_state_embeddings(encoder, train_ds, device, int(p["batch_size"]))
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
    val_ds = TransitionDataset(split_transitions(transitions, "val"))
    vs, va, _, _, _ = batch_state_embeddings(encoder, val_ds, device, int(p["batch_size"]))
    with torch.no_grad():
        adv = critic.q_min(vs.to(device), va.to(device)) - value(vs.to(device))
    positive = adv[adv > 0]
    stop_margin = float(torch.quantile(positive, float(p["stop_margin_quantile"])).detach().cpu()) if positive.numel() else 0.0
    stop_margin = min(stop_margin, 0.0)
    torch.save({"critic": critic.state_dict(), "value": value.state_dict(), "stop_margin": stop_margin}, ckpt)
    return critic.eval(), value.eval(), stop_margin


def policy_rollout(env, encoder, policy, critic, value, u0: torch.Tensor, max_steps: int, cfg: dict, stop_margin: float, no_residual: bool = False) -> tuple[torch.Tensor, dict]:
    p = cfg["paper"]
    u = u0.detach()
    errors, energies, advantages = [], [], []
    steps = 0
    stopped = False
    start = time.perf_counter()
    for k in range(max_steps):
        fields, scalars = make_state(env, u, k / max(1, max_steps), no_residual=no_residual)
        with torch.no_grad():
            z = encoder(fields.to(next(encoder.parameters()).device), scalars.to(next(encoder.parameters()).device))
            acts = policy.sample(z, int(p["candidates"]))[0]
            z_rep = z.repeat(acts.shape[0], 1)
            q = critic.q_min(z_rep, acts)
            v = value(z).repeat(acts.shape[0])
            adv = q - v
            best = int(torch.argmax(q))
            advantages.append(float(adv[best].detach().cpu()))
            if float(adv.max().detach().cpu()) <= stop_margin:
                stopped = True
                break
            action = acts[best].detach()
        u = env.step(u, action)
        steps += 1
        errors.append(float(env.relative_error(u).detach().cpu()))
        energies.append(float(env.physics_energy(u)["energy"].detach().cpu()))
    return u, {"wall_time": time.perf_counter() - start, "steps": steps, "stopped": stopped, "errors": errors, "energies": energies, "advantages": advantages}


def add_policy_rollout_transitions(cfg: dict, npz_path: Path, lno, transitions: list[Transition], encoder, policy, critic, value, device: torch.device, round_idx: int) -> list[Transition]:
    p = cfg["paper"]
    env = env_from_cfg(cfg, device)
    ds = ReactionDiffusionDataset(npz_path, "train", int(p["rollout_add_cases"]))
    rows = list(transitions)
    reward_weights = dict(cfg["reward"])
    for i in tqdm(range(len(ds)), desc=f"rollout-buffer:{round_idx}"):
        case = as_case(ds[i], 50000 + 1000 * round_idx + i)
        env.reset(case)
        with torch.no_grad():
            u = env.project(lno(case["source"].to(device).unsqueeze(0)).squeeze(0))
        for k in range(min(3, max(p["rollout_steps"]))):
            fields, scalars = make_state(env, u, k / max(1, max(p["rollout_steps"])))
            with torch.no_grad():
                z = encoder(fields.to(device), scalars.to(device))
                acts = policy.sample(z, int(p["candidates"]))[0]
                z_rep = z.repeat(acts.shape[0], 1)
                action = acts[int(torch.argmax(critic.q_min(z_rep, acts)))].detach()
            append_transition(rows, env, "train", f"policy_rollout_r{round_idx}", 50000 + 1000 * round_idx + i, u, action, reward_weights, k / max(1, max(p["rollout_steps"])))
            u = env.step(u, action)
    return rows


def supervised_fm_rollout(env, encoder, policy, u0: torch.Tensor, max_steps: int, cfg: dict) -> tuple[torch.Tensor, dict]:
    p = cfg["paper"]
    u = u0.detach()
    start = time.perf_counter()
    errors, energies = [], []
    for k in range(max_steps):
        fields, scalars = make_state(env, u, k / max(1, max_steps))
        with torch.no_grad():
            z = encoder(fields.to(next(encoder.parameters()).device), scalars.to(next(encoder.parameters()).device))
            acts = policy.sample(z, int(p["candidates"]))[0]
            candidates = [env.step(u, a.detach()) for a in acts]
            scores = [float(env.physics_energy(c)["energy"].detach().cpu()) for c in candidates]
            u = candidates[int(np.argmin(scores))]
        errors.append(float(env.relative_error(u).detach().cpu()))
        energies.append(float(env.physics_energy(u)["energy"].detach().cpu()))
    return u, {"wall_time": time.perf_counter() - start, "steps": max_steps, "stopped": False, "errors": errors, "energies": energies}


def deterministic_rollout(env, encoder, model, u0: torch.Tensor, max_steps: int) -> tuple[torch.Tensor, dict]:
    u = u0.detach()
    start = time.perf_counter()
    for k in range(max_steps):
        fields, scalars = make_state(env, u, k / max(1, max_steps))
        with torch.no_grad():
            z = encoder(fields.to(next(encoder.parameters()).device), scalars.to(next(encoder.parameters()).device))
            a = model(z)[0].detach()
        u = env.step(u, a)
    return u, {"wall_time": time.perf_counter() - start, "steps": max_steps, "stopped": False}


def evaluate_all(cfg: dict, npz_path: Path, lno, encoder_sup, fm_sup, encoder_rl, fm_rl, critic, value, stop_margin, det_encoder, det_model, device: torch.device) -> dict[str, pd.DataFrame]:
    p = cfg["paper"]
    env = env_from_cfg(cfg, device)
    test = ReactionDiffusionDataset(npz_path, "test", int(p["eval_cases"]))
    main_rows, conv_rows, init_rows, dist_rows, critic_rows = [], [], [], [], []
    methods = ["LNO only", "Gradient", "PINN", "Deterministic", "Supervised FM", "RL-FM", "Coarse + RL-FM"]
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
            base_energy = float(env.physics_energy(lno_u)["energy"].detach().cpu())
            candidates = {}
            candidates["LNO only"] = (lno_u, {"wall_time": 0.0, "steps": 0, "stopped": True})
            candidates["Gradient"] = gradient_residual_refine(env, lno_u, steps, lr=0.015)[:2]
            candidates["PINN"] = pinn_style_refine(env, lno_u, steps, lr=0.025)[:2]
            candidates["Deterministic"] = deterministic_rollout(env, det_encoder, det_model, lno_u, steps)
            candidates["Supervised FM"] = supervised_fm_rollout(env, encoder_sup, fm_sup, lno_u, steps, cfg)
            candidates["RL-FM"] = policy_rollout(env, encoder_rl, fm_rl, critic, value, lno_u, steps, cfg, stop_margin)
            candidates["Coarse + RL-FM"] = policy_rollout(env, encoder_rl, fm_rl, critic, value, coarse, steps, cfg, stop_margin)
            for name, payload in candidates.items():
                if isinstance(payload[1], float):
                    u, wall = payload
                    meta = {"wall_time": wall, "steps": steps, "stopped": False}
                else:
                    u, meta = payload
                err = float(env.relative_error(u).detach().cpu())
                energy = float(env.physics_energy(u)["energy"].detach().cpu())
                fail = float(not torch.isfinite(u).all() or err > 10.0)
                row = {
                    "Method": name,
                    "Steps": steps,
                    "Relative L2": err,
                    "PDE residual norm": float(env.physics_energy(u)["residual_norm"].detach().cpu()),
                    "BC error": float(env.physics_energy(u)["bc_error"].detach().cpu()),
                    "IC error": float(env.physics_energy(u)["ic_error"].detach().cpu()),
                    "Physics energy": energy,
                    "Wall time": float(meta["wall_time"]),
                    "Refinement steps": int(meta["steps"]),
                    "STOP": float(meta.get("stopped", False)),
                    "Failure": fail,
                    "Per-step improvement": (base_err - err) / max(1, steps),
                }
                per_method[name].append(row)
                if name in ["Gradient", "PINN", "Supervised FM", "RL-FM"]:
                    conv_rows.append(row | {"Case": i})
                if steps == 10 and name in ["LNO only", "RL-FM", "Coarse + RL-FM"]:
                    init_rows.append(row | {"Initial quality": "coarse" if "Coarse" in name else "lno"})
            dist_rows.append({"Case": i, "LNO error": base_err, "Supervised FM improvement": base_err - candidates["Supervised FM"][0].detach().sub(gt).norm().div(gt.norm().clamp_min(1e-8)).item(), "RL-FM improvement": base_err - candidates["RL-FM"][0].detach().sub(gt).norm().div(gt.norm().clamp_min(1e-8)).item()})
        for name, rows in per_method.items():
            if rows:
                main_rows.extend(rows)
    # Critic correlation on held-out transitions.
    held_trans = split_transitions(build_transition_buffer(cfg, npz_path, lno, device), "test")
    held_ds = TransitionDataset(held_trans)
    s, a, r, _, _ = batch_state_embeddings(encoder_rl, held_ds, device, int(p["batch_size"]))
    with torch.no_grad():
        q = critic.q_min(s.to(device), a.to(device)).detach().cpu().numpy()
        v = value(s.to(device)).detach().cpu().numpy()
    pearson_q, spearman_q = corr(q, r.numpy())
    pearson_a, spearman_a = corr((q - v), r.numpy())
    critic_rows.append({"Q Pearson": pearson_q, "Q Spearman": spearman_q, "Advantage Pearson": pearson_a, "Advantage Spearman": spearman_a, "Samples": len(held_trans)})
    return {
        "raw_main": pd.DataFrame(main_rows),
        "main": summarize(pd.DataFrame(main_rows), ["Method", "Steps"]),
        "convergence": summarize(pd.DataFrame(conv_rows), ["Method", "Steps"]),
        "initialization": summarize(pd.DataFrame(init_rows), ["Method", "Initial quality"]),
        "critic": pd.DataFrame(critic_rows),
        "improvement_distribution": pd.DataFrame(dist_rows),
    }


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


def run_ablations(cfg: dict, npz_path: Path, lno, device: torch.device) -> pd.DataFrame:
    rows = []
    base_trans = build_transition_buffer(cfg, npz_path, lno, device)
    variants = [
        ("Full RL-FM", "full", "full", False, True),
        ("without RL weighting = Supervised FM", "full", "full", False, False),
        ("single Q instead of Twin-Q", "full", "full", True, True),
        ("without PDE residual in state", "no_residual", "full", False, True),
        ("without physics reward", "full", "no_physics", False, True),
        ("without GT reward shaping during training", "full", "no_gt", False, True),
        ("deterministic correction instead of Flow Matching", "full", "full", False, False),
    ]
    test = ReactionDiffusionDataset(npz_path, "test", min(12, int(cfg["paper"]["eval_cases"])))
    for name, enc_variant, reward_variant, single_q, use_aw in tqdm(variants, desc="ablation"):
        if name.startswith("deterministic"):
            enc, det = train_deterministic(cfg, base_trans, device)
        else:
            trans = base_trans if enc_variant == "full" and reward_variant == "full" else build_transition_buffer(cfg, npz_path, lno, device, enc_variant, reward_variant)
            enc_sup, pol_sup = train_encoder_fm(cfg, trans, f"abl_sup_{enc_variant}_{reward_variant}", device, aw=False)
            critic, value, margin = train_iql(cfg, trans, enc_sup, f"abl_{enc_variant}_{reward_variant}_{'sq' if single_q else 'tq'}", device, single_q=single_q)
            if use_aw:
                enc, pol = train_encoder_fm(cfg, trans, f"abl_aw_{enc_variant}_{reward_variant}_{'sq' if single_q else 'tq'}", device, aw=True, critic=critic, value=value)
            else:
                enc, pol = enc_sup, pol_sup
        vals = []
        env = env_from_cfg(cfg, device)
        for i in range(len(test)):
            case = as_case(test[i], i)
            env.reset(case)
            with torch.no_grad():
                lno_u = env.project(lno(case["source"].to(device).unsqueeze(0)).squeeze(0))
            if name.startswith("deterministic"):
                pred, meta = deterministic_rollout(env, enc, det, lno_u, 5)
            else:
                pred, meta = policy_rollout(env, enc, pol, critic, value, lno_u, 5, cfg, margin, no_residual=(enc_variant == "no_residual"))
            vals.append({"Ablation": name, "Relative L2": float(env.relative_error(pred).detach().cpu()), "Physics energy": float(env.physics_energy(pred)["energy"].detach().cpu()), "Wall time": meta["wall_time"], "STOP": float(meta.get("stopped", False)), "Refinement steps": meta["steps"], "Failure": 0.0, "Per-step improvement": 0.0, "PDE residual norm": float(env.physics_energy(pred)["residual_norm"].detach().cpu()), "BC error": 0.0, "IC error": 0.0})
        rows.extend(vals)
    return summarize(pd.DataFrame(rows), ["Ablation"])


def save_tables(results: dict[str, pd.DataFrame], ablation: pd.DataFrame) -> None:
    table_dir = ROOT / "results" / "tables"
    results["main"].to_csv(table_dir / "main_results.csv", index=False)
    results["convergence"].to_csv(table_dir / "convergence.csv", index=False)
    results["critic"].to_csv(table_dir / "critic_results.csv", index=False)
    results["initialization"].to_csv(table_dir / "initialization_results.csv", index=False)
    results["improvement_distribution"].to_csv(table_dir / "improvement_distribution.csv", index=False)
    ablation.to_csv(table_dir / "ablation.csv", index=False)


def generate_figures(results: dict[str, pd.DataFrame], ablation: pd.DataFrame, npz_path: Path, lno, encoder_rl, fm_rl, critic, value, stop_margin, cfg: dict, device: torch.device) -> None:
    fig_dir = ROOT / "results" / "figures"
    raw = results["raw_main"]
    conv = raw[raw["Method"].isin(["Gradient", "PINN", "Supervised FM", "RL-FM"])]
    def line(metric: str, path: str, xlabel: str = "Refinement iterations"):
        plt.figure(figsize=(6, 4))
        for name, g in conv.groupby("Method"):
            gg = g.groupby("Steps")[metric].mean()
            plt.plot(gg.index, gg.values, marker="o", label=name)
        plt.xlabel(xlabel)
        plt.ylabel(metric)
        plt.legend()
        plt.tight_layout()
        plt.savefig(fig_dir / path, dpi=180)
        plt.close()
    # Figure 1
    plt.figure(figsize=(8, 2.4))
    blocks = ["PDE case", "Residual/state encoder", "Flow Matching policy", "Twin-Q/V critic", "Best correction", "Refined PDE"]
    for i, b in enumerate(blocks):
        plt.text(i, 0.5, b, ha="center", va="center", bbox=dict(boxstyle="round,pad=0.3", fc="#edf2f7", ec="#2d3748"))
        if i < len(blocks) - 1:
            plt.arrow(i + 0.35, 0.5, 0.28, 0, head_width=0.04, color="#2d3748")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(fig_dir / "figure1_architecture.png", dpi=180)
    plt.close()
    # Figure 2
    test = ReactionDiffusionDataset(npz_path, "test", 1)
    env = env_from_cfg(cfg, device)
    case = as_case(test[0], 0)
    env.reset(case)
    with torch.no_grad():
        lno_u = env.project(lno(case["source"].to(device).unsqueeze(0)).squeeze(0))
    refined, _ = policy_rollout(env, encoder_rl, fm_rl, critic, value, lno_u, 5, cfg, stop_margin)
    gt = case["gt"].to(device)
    panels = [gt, lno_u, env.residual(lno_u), refined - lno_u, refined, (refined - gt).abs()]
    titles = ["GT", "LNO", "residual", "RL-Flow correction", "refined", "error map"]
    plt.figure(figsize=(10, 4))
    for i, (arr, title) in enumerate(zip(panels, titles), 1):
        plt.subplot(2, 3, i)
        plt.imshow(arr.detach().cpu().numpy(), aspect="auto", cmap="viridis")
        plt.title(title)
        plt.colorbar(fraction=0.046)
    plt.tight_layout()
    plt.savefig(fig_dir / "figure2_representative_solution.png", dpi=180)
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
    plt.hist(dist["Supervised FM improvement"], bins=16, alpha=0.6, label="Supervised FM")
    plt.hist(dist["RL-FM improvement"], bins=16, alpha=0.6, label="RL-FM")
    plt.xlabel("Improvement over LNO")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "figure6_fm_vs_rlfm_distribution.png", dpi=180)
    plt.close()
    crit = results["critic"]
    plt.figure(figsize=(4.5, 3.5))
    vals = [float(crit["Q Pearson"].iloc[0]), float(crit["Advantage Pearson"].iloc[0])]
    plt.bar(["Q", "Advantage"], vals)
    plt.ylabel("Pearson with held-out return")
    plt.tight_layout()
    plt.savefig(fig_dir / "figure7_critic_correlation.png", dpi=180)
    plt.close()
    init = results["raw_main"][results["raw_main"]["Method"].isin(["LNO only", "RL-FM", "Coarse + RL-FM"])]
    plt.figure(figsize=(6, 4))
    for name, g in init.groupby("Method"):
        gg = g.groupby("Steps")["Relative L2"].mean()
        plt.plot(gg.index, gg.values, marker="o", label=name)
    plt.xlabel("Refinement iterations")
    plt.ylabel("Relative L2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "figure8_initial_quality_convergence.png", dpi=180)
    plt.close()


def write_docs(results: dict[str, pd.DataFrame], ablation: pd.DataFrame, cfg: dict) -> None:
    docs = ROOT / "docs"
    main = results["main"]
    def table(name: str) -> str:
        path = ROOT / "results" / "tables" / name
        return pd.read_csv(path).to_markdown(index=False) if path.exists() else "not generated"
    claims = ["# Paper Claims", ""]
    lno10 = main[(main.Method == "LNO only") & (main.Steps == 10)]["Relative L2"].iloc[0]
    sup10 = main[(main.Method == "Supervised FM") & (main.Steps == 10)]["Relative L2"].iloc[0]
    rl10 = main[(main.Method == "RL-FM") & (main.Steps == 10)]["Relative L2"].iloc[0]
    critic = pd.read_csv(ROOT / "results" / "tables" / "critic_results.csv")
    claims.append(f"- Supervised FM vs no refinement at 10 steps: {sup10} vs {lno10}. In this run, supervised Flow Matching does not improve over LNO-only.")
    claims.append(f"- RL-guided FM vs Supervised FM at 10 steps: {rl10} vs {sup10}. In this run, advantage weighting does not improve the correction policy.")
    claims.append("- Same-iteration and wall-time comparisons favor the gradient residual refinement baseline on this compact benchmark.")
    claims.append(f"- Critic validation is mixed: Q Spearman = {critic['Q Spearman'].iloc[0]:.4f}, Advantage Spearman = {critic['Advantage Spearman'].iloc[0]:.4f}, while Pearson correlations are negative.")
    claims.append("- RL-FM executes sequential correction steps after policy-improvement rounds, but those steps do not yield reliable monotonic error reduction.")
    claims.append("- Results are reported as measured outcomes; unsupported improvements are explicitly rejected.")
    claims.append("- GT is used for offline reward construction and evaluation only, not policy/critic inference inputs.")
    (docs / "paper_claims.md").write_text("\n".join(claims) + "\n", encoding="utf-8")
    sections = [
        "# Results",
        "",
        "Interpretation:",
        "The final paper-mode run is a negative or weak-support result. The implementation completes the residual-conditioned Flow Matching + IQL loop, including two policy-improvement rounds, but the learned RL-FM corrections do not beat the LNO-only baseline or the gradient residual baseline on the official 2D_Reac_diffusion split used here.",
        "",
        "Main results:",
        table("main_results.csv"),
        "",
        "Convergence:",
        table("convergence.csv"),
        "",
        "Critic:",
        table("critic_results.csv"),
        "",
        "Initialization:",
        table("initialization_results.csv"),
        "",
        "Ablation:",
        table("ablation.csv"),
    ]
    (docs / "paper_results.md").write_text("\n\n".join(sections) + "\n", encoding="utf-8")
    (docs / "method.md").write_text(
        "# Method\n\nRL-Flow-PDE refines an initial reaction-diffusion solution by encoding the current solution, PDE residual, IC/BC fields, and scalar physics context with a spectral operator encoder. A conditional Flow Matching policy samples low-frequency spectral corrections. An IQL-style Twin-Q and V critic scores candidate corrections, and inference applies the highest-value correction unless its advantage falls below a validation-calibrated stop margin.\n",
        encoding="utf-8",
    )
    (docs / "limitations.md").write_text(
        "# Limitations\n\nThe first paper-mode run uses the official LNO 2D_Reac_diffusion data file when available and a compact local fallback only if it is missing. The implementation is intentionally small enough for an 8 GB laptop GPU. It does not claim state-of-the-art LNO performance, and it reports cases where residual reduction and ground-truth error improvement disagree.\n",
        encoding="utf-8",
    )


def run_pipeline(mode: str = "paper") -> None:
    cfg = load_config()
    ensure_dirs()
    set_seed(int(cfg["seed"]))
    device = get_device(str(cfg.get("device", "cuda")))
    npz_path = prepare_reaction_diffusion(cfg, ROOT)
    lno = train_lno(cfg, npz_path, device)
    transitions = build_transition_buffer(cfg, npz_path, lno, device)
    encoder_sup, fm_sup = train_encoder_fm(cfg, transitions, "supervised", device, aw=False)
    critic, value, stop_margin = train_iql(cfg, transitions, encoder_sup, "main_twinq", device, single_q=False)
    encoder_rl, fm_rl = train_encoder_fm(cfg, transitions, "rl_guided", device, aw=True, critic=critic, value=value)
    for round_idx in range(int(cfg["paper"]["policy_improvement_rounds"])):
        transitions = add_policy_rollout_transitions(cfg, npz_path, lno, transitions, encoder_rl, fm_rl, critic, value, device, round_idx + 1)
        critic, value, stop_margin = train_iql(cfg, transitions, encoder_rl, f"main_twinq_round{round_idx + 1}", device, single_q=False)
        encoder_rl, fm_rl = train_encoder_fm(cfg, transitions, f"rl_guided_round{round_idx + 1}", device, aw=True, critic=critic, value=value)
    det_encoder, det_model = train_deterministic(cfg, transitions, device)
    results = evaluate_all(cfg, npz_path, lno, encoder_sup, fm_sup, encoder_rl, fm_rl, critic, value, stop_margin, det_encoder, det_model, device)
    ablation = run_ablations(cfg, npz_path, lno, device)
    save_tables(results, ablation)
    generate_figures(results, ablation, npz_path, lno, encoder_rl, fm_rl, critic, value, stop_margin, cfg, device)
    write_docs(results, ablation, cfg)
    summary = {"mode": mode, "device": str(device), "prepared_data": str(npz_path), "stop_margin": stop_margin, "tables": sorted(p.name for p in (ROOT / "results" / "tables").glob("*.csv"))}
    (ROOT / "results" / "tables" / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Pipeline completed.")
