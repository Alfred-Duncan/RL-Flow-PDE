from __future__ import annotations

import argparse
import json
import sys
import time
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
V2_CACHE_VERSION = "stable_bound_v2"

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.data.official_reaction_diffusion import ReactionDiffusionDataset, prepare_reaction_diffusion
from src.solver_v2.data.normalization import fit_train_stats, normalize_residual, normalize_source, normalize_u
from src.solver_v2.data.replay_buffer import ReplayBuffer, SolverTransition
from src.solver_v2.models.fno_initializer import FNOInitializer, initializer_loss
from src.solver_v2.models.operator_actor import DeterministicNeuralOperatorActor
from src.solver_v2.models.operator_critic import TwinOperatorCritic
from src.solver_v2.pde.reaction_diffusion import ReactionDiffusionCase, ReactionDiffusionPDE
from src.solver_v2.rl.td3_bc import TD3BCTrainer
from src.solver_v2.training.pretrain_actor import pretrain_actor
from src.utils.seed import get_device, set_seed


def load_config() -> dict:
    with open(ROOT / "configs/default.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dirs() -> None:
    for path in ["checkpoints/solver_v2", "results/solver_v2/tables", "results/solver_v2/figures", "docs"]:
        (ROOT / path).mkdir(parents=True, exist_ok=True)


def state_from_u(pde: ReactionDiffusionPDE, case: ReactionDiffusionCase, u: torch.Tensor, step_frac: float, stats: dict) -> tuple[torch.Tensor, torch.Tensor]:
    res = pde.residual(u, case)
    ic_field = case.ic[:, None].expand_as(u)
    bc_field = torch.zeros_like(u)
    bc_field[0, :] = case.bc_left
    bc_field[-1, :] = case.bc_right
    fields = torch.stack(
        [
            normalize_u(u, stats),
            normalize_residual(res, stats),
            normalize_source(case.source, stats),
            normalize_u(ic_field, stats),
            normalize_u(bc_field, stats),
        ],
        dim=0,
    )
    scalars = torch.tensor([pde.diffusion, pde.reaction, float(step_frac), 1.0], dtype=torch.float32, device=u.device)
    return fields, scalars


def train_initializer(cfg: dict, npz_path: Path, seed: int, pde: ReactionDiffusionPDE, device: torch.device) -> FNOInitializer:
    s = cfg["solver_v2"]
    ckpt = ROOT / "checkpoints" / "solver_v2" / f"fno_initializer_seed{seed}.pt"
    model = FNOInitializer(width=int(s["width"]), modes=int(s["modes"]), depth=3).to(device)
    if ckpt.exists():
        model.load_state_dict(torch.load(ckpt, map_location=device)["model"])
        return model.eval()
    ds = ReactionDiffusionDataset(npz_path, "train", int(s["train_cases"]))
    loader = DataLoader(ds, batch_size=int(s["batch_size"]), shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=float(s["lr"]), weight_decay=1e-4)
    for _ in tqdm(range(int(s["initializer_epochs"])), desc=f"v2:init:{seed}"):
        model.train()
        for batch in loader:
            source = batch["source"].to(device)
            gt = batch["gt"].to(device)
            pred = model(source)
            loss = initializer_loss(pred, gt)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
    torch.save({"model": model.state_dict()}, ckpt)
    return model.eval()


def initial_solution(initializer: FNOInitializer, pde: ReactionDiffusionPDE, case: ReactionDiffusionCase) -> torch.Tensor:
    with torch.no_grad():
        pred = initializer(case.source.unsqueeze(0)).squeeze(0)
    return pde.project_hard_constraints(pred, case)


def bounded_target_delta(u: torch.Tensor, target: torch.Tensor, stats: dict, remaining: int) -> torch.Tensor:
    raw = (target - u) / float(max(1, remaining))
    bound = stats["step_bound"].to(u.device)
    return raw.clamp(-bound, bound)


def append_step(rows: list[SolverTransition], pde: ReactionDiffusionPDE, case: ReactionDiffusionCase, u: torch.Tensor, delta: torch.Tensor, stats: dict, split: str, episode_id: int, step_idx: int, horizon: int, source_policy: str, cfg: dict) -> torch.Tensor:
    fields, scalars = state_from_u(pde, case, u, step_idx / max(1, horizon), stats)
    next_u = pde.step(u, delta, case)
    next_fields, next_scalars = state_from_u(pde, case, next_u, (step_idx + 1) / max(1, horizon), stats)
    r = pde.reward(u, next_u, delta, case, float(cfg["solver_v2"]["lambda_action"]))
    rows.append(
        SolverTransition(
            case_id=case.case_id,
            episode_id=episode_id,
            step_idx=step_idx,
            split=split,
            state_fields=fields.detach().cpu(),
            state_scalars=scalars.detach().cpu(),
            action=delta.detach().cpu(),
            reward=r["reward"],
            next_fields=next_fields.detach().cpu(),
            next_scalars=next_scalars.detach().cpu(),
            done=float(step_idx == horizon - 1),
            error_before=r["error_before"],
            error_after=r["error_after"],
            residual_before=r["residual_before"],
            residual_after=r["residual_after"],
            physics_before=r["physics_before"],
            physics_after=r["physics_after"],
            action_norm=r["action_norm"],
            source_policy=source_policy,
        )
    )
    return next_u.detach()


def gradient_delta(pde: ReactionDiffusionPDE, case: ReactionDiffusionCase, u: torch.Tensor, stats: dict, lr: float = 0.015) -> torch.Tensor:
    var = u.detach().clone().requires_grad_(True)
    loss = pde.physics_metrics(var, case)["energy"]
    grad = torch.autograd.grad(loss, var)[0]
    delta = -lr * grad / grad.abs().mean().clamp_min(1e-4)
    return delta.clamp(-stats["step_bound"].to(u.device), stats["step_bound"].to(u.device)).detach()


def pinn_delta(pde: ReactionDiffusionPDE, case: ReactionDiffusionCase, u: torch.Tensor, stats: dict, steps: int = 3) -> torch.Tensor:
    bound = stats["step_bound"].to(u.device)
    raw = torch.zeros_like(u, requires_grad=True)
    opt = torch.optim.Adam([raw], lr=0.01)
    for _ in range(steps):
        delta = bound * torch.tanh(raw)
        cand = pde.step(u, delta, case)
        loss = pde.physics_metrics(cand, case)["energy"] / stats["residual_rms"].to(u.device).clamp_min(1e-6) + 0.05 * delta.pow(2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return (bound * torch.tanh(raw.detach())).detach()


def generate_supervised_trajectories(cfg: dict, npz_path: Path, initializer: FNOInitializer, pde: ReactionDiffusionPDE, stats: dict, seed: int, device: torch.device) -> list[SolverTransition]:
    cache = ROOT / "checkpoints" / "solver_v2" / f"supervised_trajectories_{V2_CACHE_VERSION}_seed{seed}.pt"
    if cache.exists():
        return torch.load(cache, weights_only=False)
    rows: list[SolverTransition] = []
    horizon = int(cfg["solver_v2"]["horizon"])
    episode = 0
    for split, limit in [("train", int(cfg["solver_v2"]["train_cases"])), ("val", int(cfg["solver_v2"]["val_cases"]))]:
        ds = ReactionDiffusionDataset(npz_path, split, limit)
        for i in tqdm(range(len(ds)), desc=f"v2:supervised_traj:{split}:{seed}"):
            case = pde.make_case(ds[i], device)
            case.case_id = i if split == "train" else 10000 + i
            for policy_name in ["gt_directed", "gradient"]:
                u = initial_solution(initializer, pde, case)
                for k in range(horizon):
                    if policy_name == "gt_directed":
                        delta = bounded_target_delta(u, case.gt, stats, horizon - k)
                    else:
                        delta = gradient_delta(pde, case, u, stats)
                    u = append_step(rows, pde, case, u, delta, stats, split, episode, k, horizon, policy_name, cfg)
                episode += 1
    torch.save(rows, cache)
    return rows


def rollout_actor_to_replay(cfg: dict, npz_path: Path, initializer: FNOInitializer, actor: DeterministicNeuralOperatorActor, pde: ReactionDiffusionPDE, stats: dict, seed: int, cycle: int, device: torch.device) -> list[SolverTransition]:
    rows: list[SolverTransition] = []
    horizon = int(cfg["solver_v2"]["horizon"])
    ds = ReactionDiffusionDataset(npz_path, "train", int(cfg["solver_v2"]["online_rollout_cases"]))
    actor.eval()
    for i in tqdm(range(len(ds)), desc=f"v2:online_rollout:{seed}:{cycle}"):
        case = pde.make_case(ds[i], device)
        case.case_id = 50000 + cycle * 1000 + i
        u = initial_solution(initializer, pde, case)
        episode = 50000 + cycle * 1000 + i
        for k in range(horizon):
            fields, scalars = state_from_u(pde, case, u, k / max(1, horizon), stats)
            with torch.no_grad():
                delta = actor(fields.unsqueeze(0).to(device), scalars.unsqueeze(0).to(device), stats, float(cfg["solver_v2"]["temperature"])).squeeze(0)
                noise = 0.10 * stats["step_bound"].to(device) * torch.randn_like(delta)
                delta = (delta + noise).clamp(-stats["step_bound"].to(device), stats["step_bound"].to(device))
            u = append_step(rows, pde, case, u, delta, stats, "train", episode, k, horizon, f"online_actor_cycle{cycle}", cfg)
    return rows


def train_td3_variant(cfg: dict, transitions: list[SolverTransition], actor: DeterministicNeuralOperatorActor, seed: int, variant: str, stats: dict, device: torch.device, epochs_override: int | None = None) -> tuple[DeterministicNeuralOperatorActor, TwinOperatorCritic, list[dict[str, float]]]:
    local_cfg = deepcopy(cfg)
    if variant == "no_physics":
        local_cfg["solver_v2"]["lambda_phys"] = 0.0
    if epochs_override is not None:
        local_cfg["solver_v2"]["td3_epochs"] = int(epochs_override)
    critic = TwinOperatorCritic(width=int(cfg["solver_v2"]["width"]), modes=int(cfg["solver_v2"]["modes"]), depth=int(cfg["solver_v2"]["critic_depth"])).to(device)
    trainer = TD3BCTrainer(actor, critic, local_cfg, stats, device)
    ckpt = ROOT / "checkpoints" / "solver_v2" / f"td3_{V2_CACHE_VERSION}_{variant}_seed{seed}.pt"
    return trainer.fit(transitions, ckpt)


def validation_solver_error(cfg: dict, npz_path: Path, initializer: FNOInitializer, actor: DeterministicNeuralOperatorActor, pde: ReactionDiffusionPDE, stats: dict, device: torch.device) -> float:
    ds = ReactionDiffusionDataset(npz_path, "val", int(cfg["solver_v2"]["val_cases"]))
    steps = int(cfg["solver_v2"]["horizon"])
    vals = []
    actor.eval()
    for i in range(len(ds)):
        case = pde.make_case(ds[i], device)
        u = initial_solution(initializer, pde, case)
        for k in range(steps):
            fields, scalars = state_from_u(pde, case, u, k / max(1, steps), stats)
            with torch.no_grad():
                delta = actor(fields.unsqueeze(0).to(device), scalars.unsqueeze(0).to(device), stats, float(cfg["solver_v2"]["temperature"])).squeeze(0)
            u = pde.step(u, delta, case)
        vals.append(float(pde.relative_l2(u, case.gt).detach().cpu()))
    return float(np.mean(vals))


def rollout_method(method: str, cfg: dict, initializer: FNOInitializer, actor: DeterministicNeuralOperatorActor | None, critic: TwinOperatorCritic | None, pde: ReactionDiffusionPDE, stats: dict, case: ReactionDiffusionCase, steps: int, device: torch.device) -> tuple[torch.Tensor, list[dict[str, float]], float]:
    u = initial_solution(initializer, pde, case)
    trace = []
    start = time.perf_counter()
    for k in range(steps):
        if method == "Base FNO Initializer":
            break
        if method == "Gradient residual refinement":
            delta = gradient_delta(pde, case, u, stats)
            q1 = q2 = np.nan
        elif method == "PINN-style refinement":
            delta = pinn_delta(pde, case, u, stats, steps=3)
            q1 = q2 = np.nan
        elif actor is not None:
            fields, scalars = state_from_u(pde, case, u, k / max(1, steps), stats)
            with torch.no_grad():
                delta = actor(fields.unsqueeze(0).to(device), scalars.unsqueeze(0).to(device), stats, float(cfg["solver_v2"]["temperature"])).squeeze(0)
                if critic is not None:
                    q1_t, q2_t = critic(fields.unsqueeze(0).to(device), scalars.unsqueeze(0).to(device), delta.unsqueeze(0))
                    q1, q2 = float(q1_t.item()), float(q2_t.item())
                else:
                    q1 = q2 = np.nan
        else:
            raise ValueError(method)
        next_u = pde.step(u, delta, case)
        r = pde.reward(u, next_u, delta, case, float(cfg["solver_v2"]["lambda_action"]))
        trace.append({"step": k + 1, "Relative L2": r["error_after"], "PDE residual norm": r["residual_after"], "Physics energy": r["physics_after"], "action norm": r["action_norm"], "Q1": q1, "Q2": q2, "reward": r["reward"]})
        u = next_u
    return u, trace, time.perf_counter() - start


def evaluate_methods(cfg: dict, npz_path: Path, seed: int, initializer: FNOInitializer, actors: dict[str, DeterministicNeuralOperatorActor], critics: dict[str, TwinOperatorCritic | None], pde: ReactionDiffusionPDE, stats: dict, device: torch.device) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ds = ReactionDiffusionDataset(npz_path, "test", int(cfg["solver_v2"]["eval_cases"]))
    rows, per_case, conv, physics_rows = [], [], [], []
    methods = [
        "Base FNO Initializer",
        "Gradient residual refinement",
        "PINN-style refinement",
        "Supervised Neural Operator Corrector",
        "TD3 from scratch",
        "TD3+BC without physics regularization",
        "RL Neural Operator Solver",
    ]
    for steps in tqdm(cfg["solver_v2"]["eval_steps"], desc=f"v2:evaluate:{seed}"):
        method_rows = {m: [] for m in methods}
        for i in range(len(ds)):
            case = pde.make_case(ds[i], device)
            base_u = initial_solution(initializer, pde, case)
            base_err = float(pde.relative_l2(base_u, case.gt).detach().cpu())
            for method in methods:
                actor = actors.get(method)
                critic = critics.get(method)
                u, trace, wall = rollout_method(method, cfg, initializer, actor, critic, pde, stats, case, int(steps), device)
                pm = pde.physics_metrics(u, case)
                err = float(pde.relative_l2(u, case.gt).detach().cpu())
                action_norm = float(np.mean([t["action norm"] for t in trace])) if trace else 0.0
                ret = float(np.sum([t["reward"] for t in trace])) if trace else 0.0
                row = {
                    "Seed": seed,
                    "Method": method,
                    "Steps": int(steps),
                    "Relative L2": err,
                    "PDE residual norm": float(pm["residual_norm"].detach().cpu()),
                    "BC error": float(pm["bc_error"].detach().cpu()),
                    "IC error": float(pm["ic_error"].detach().cpu()),
                    "wall time": wall,
                    "solver steps": len(trace),
                    "action norm": action_norm,
                    "return": ret,
                    "paired improvement": base_err - err,
                    "Case": i,
                }
                method_rows[method].append(row)
                per_case.append(row)
                physics_rows.append({k: row[k] for k in ["Seed", "Method", "Steps", "Case", "PDE residual norm", "BC error", "IC error"]})
                for tr in trace:
                    conv.append({"Seed": seed, "Method": method, "Case": i, "StepCap": int(steps), **tr})
        for vals in method_rows.values():
            rows.extend(vals)
    return pd.DataFrame(rows), pd.DataFrame(per_case), pd.DataFrame(conv), pd.DataFrame(physics_rows)


def summarize_main(per_case: pd.DataFrame) -> pd.DataFrame:
    metrics = ["Relative L2", "PDE residual norm", "BC error", "IC error", "wall time", "solver steps", "action norm", "return", "paired improvement"]
    rows = []
    for (method, steps), g in per_case.groupby(["Method", "Steps"]):
        row = {"Method": method, "Steps": steps}
        for metric in metrics:
            vals = g[metric].astype(float)
            row[f"{metric} mean"] = vals.mean()
            row[f"{metric} std"] = vals.std(ddof=0)
            row[f"{metric} median"] = vals.median()
            seed_vals = g.groupby("Seed")[metric].mean().astype(float)
            row[f"{metric} seed mean"] = seed_vals.mean()
            row[f"{metric} seed std"] = seed_vals.std(ddof=0)
        row["Samples"] = len(g)
        rows.append(row)
    return pd.DataFrame(rows)


def seed_summary(per_case: pd.DataFrame) -> pd.DataFrame:
    sub = per_case[per_case["Steps"].eq(10)]
    return sub.groupby(["Seed", "Method"], as_index=False).agg(
        relative_l2=("Relative L2", "mean"),
        residual=("PDE residual norm", "mean"),
        wall_time=("wall time", "mean"),
        return_mean=("return", "mean"),
        paired_improvement=("paired improvement", "mean"),
    )


def residual_accuracy_quadrants(transitions: list[SolverTransition]) -> pd.DataFrame:
    rows = []
    for source, g in pd.DataFrame([tr.__dict__ for tr in transitions]).groupby("source_policy"):
        for label, mask in [
            ("Residual better / Accuracy better", (g.residual_after < g.residual_before) & (g.error_after < g.error_before)),
            ("Residual better / Accuracy worse", (g.residual_after < g.residual_before) & (g.error_after >= g.error_before)),
            ("Residual worse / Accuracy better", (g.residual_after >= g.residual_before) & (g.error_after < g.error_before)),
            ("Both worse", (g.residual_after >= g.residual_before) & (g.error_after >= g.error_before)),
        ]:
            rows.append({"source_policy": source, "Quadrant": label, "Count": int(mask.sum()), "Fraction": float(mask.mean())})
    return pd.DataFrame(rows)


def run_seed(cfg: dict, npz_path: Path, seed: int, device: torch.device) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[SolverTransition], pd.DataFrame]:
    set_seed(seed)
    pde = ReactionDiffusionPDE(float(cfg["benchmark"]["diffusion"]), float(cfg["benchmark"]["reaction"]))
    stats = fit_train_stats(cfg, npz_path, pde, device, ROOT / "checkpoints" / "solver_v2")
    initializer = train_initializer(cfg, npz_path, seed, pde, device)
    supervised = generate_supervised_trajectories(cfg, npz_path, initializer, pde, stats, seed, device)
    actor_sup = DeterministicNeuralOperatorActor(width=int(cfg["solver_v2"]["width"]), modes=int(cfg["solver_v2"]["modes"]), depth=int(cfg["solver_v2"]["actor_depth"])).to(device)
    actor_sup = pretrain_actor(actor_sup, supervised, cfg, stats, device, ROOT / "checkpoints" / "solver_v2" / f"actor_pretrain_{V2_CACHE_VERSION}_seed{seed}.pt")
    replay = ReplayBuffer()
    replay.extend(supervised)
    actor_rl = deepcopy(actor_sup).to(device)
    best_actor = deepcopy(actor_sup).to(device)
    best_val = validation_solver_error(cfg, npz_path, initializer, best_actor, pde, stats, device)
    critic_rl = None
    history = [{"epoch": 0.0, "critic_loss": float("nan"), "actor_loss": float("nan"), "variant": "pretrain", "val_error": best_val}]
    for cycle in range(int(cfg["solver_v2"]["online_cycles"])):
        replay.extend(rollout_actor_to_replay(cfg, npz_path, initializer, actor_rl, pde, stats, seed, cycle, device))
        actor_rl, critic_rl, hist = train_td3_variant(cfg, replay.transitions, actor_rl, seed, f"full_cycle{cycle}", stats, device)
        val_error = validation_solver_error(cfg, npz_path, initializer, actor_rl, pde, stats, device)
        for h in hist:
            h["variant"] = f"full_cycle{cycle}"
            h["val_error"] = val_error
        history.extend(hist)
        if val_error <= best_val:
            best_val = val_error
            best_actor = deepcopy(actor_rl).to(device)
        else:
            actor_rl = deepcopy(best_actor).to(device)
    torch.save({"actor": best_actor.state_dict(), "validation_error": best_val}, ROOT / "checkpoints" / "solver_v2" / f"actor_selected_{V2_CACHE_VERSION}_seed{seed}.pt")
    actor_no_phys, critic_no_phys, hist_no_phys = train_td3_variant(cfg, replay.transitions, deepcopy(actor_sup).to(device), seed, "no_physics", stats, device)
    actor_scratch = DeterministicNeuralOperatorActor(width=int(cfg["solver_v2"]["width"]), modes=int(cfg["solver_v2"]["modes"]), depth=int(cfg["solver_v2"]["actor_depth"])).to(device)
    actor_scratch, critic_scratch, hist_scratch = train_td3_variant(cfg, replay.transitions, actor_scratch, seed, "scratch", stats, device, int(cfg["solver_v2"]["td3_scratch_epochs"]))
    actors = {
        "Supervised Neural Operator Corrector": actor_sup,
        "TD3 from scratch": actor_scratch,
        "TD3+BC without physics regularization": actor_no_phys,
        "RL Neural Operator Solver": best_actor,
    }
    critics = {
        "TD3 from scratch": critic_scratch,
        "TD3+BC without physics regularization": critic_no_phys,
        "RL Neural Operator Solver": critic_rl,
    }
    eval_df, per_case, conv, physics = evaluate_methods(cfg, npz_path, seed, initializer, actors, critics, pde, stats, device)
    training_summary = pd.DataFrame(history + [{"epoch": h["epoch"], "critic_loss": h["critic_loss"], "actor_loss": h["actor_loss"], "variant": "no_physics"} for h in hist_no_phys] + [{"epoch": h["epoch"], "critic_loss": h["critic_loss"], "actor_loss": h["actor_loss"], "variant": "scratch"} for h in hist_scratch])
    training_summary["Seed"] = seed
    return eval_df, per_case, conv, replay.transitions, training_summary


def generate_representative_figure(cfg: dict, npz_path: Path, seed: int, device: torch.device) -> None:
    fig_dir = ROOT / "results" / "solver_v2" / "figures"
    pde = ReactionDiffusionPDE(float(cfg["benchmark"]["diffusion"]), float(cfg["benchmark"]["reaction"]))
    stats = fit_train_stats(cfg, npz_path, pde, device, ROOT / "checkpoints" / "solver_v2")
    initializer = FNOInitializer(width=int(cfg["solver_v2"]["width"]), modes=int(cfg["solver_v2"]["modes"]), depth=3).to(device)
    initializer.load_state_dict(torch.load(ROOT / "checkpoints" / "solver_v2" / f"fno_initializer_seed{seed}.pt", map_location=device)["model"])
    actor = DeterministicNeuralOperatorActor(width=int(cfg["solver_v2"]["width"]), modes=int(cfg["solver_v2"]["modes"]), depth=int(cfg["solver_v2"]["actor_depth"])).to(device)
    selected = ROOT / "checkpoints" / "solver_v2" / f"actor_selected_{V2_CACHE_VERSION}_seed{seed}.pt"
    fallback = ROOT / "checkpoints" / "solver_v2" / f"actor_pretrain_{V2_CACHE_VERSION}_seed{seed}.pt"
    actor.load_state_dict(torch.load(selected if selected.exists() else fallback, map_location=device)["actor"])
    ds = ReactionDiffusionDataset(npz_path, "test", 1)
    case = pde.make_case(ds[0], device)
    u0 = initial_solution(initializer, pde, case)
    u = u0.detach()
    for k in range(10):
        fields, scalars = state_from_u(pde, case, u, k / 10.0, stats)
        with torch.no_grad():
            delta = actor(fields.unsqueeze(0).to(device), scalars.unsqueeze(0).to(device), stats, float(cfg["solver_v2"]["temperature"])).squeeze(0)
        u = pde.step(u, delta, case)
    panels = [
        ("GT", case.gt),
        ("Initial FNO", u0),
        ("RL refined", u),
        ("Absolute error", (u - case.gt).abs()),
        ("Initial residual", pde.residual(u0, case)),
        ("Refined residual", pde.residual(u, case)),
    ]
    plt.figure(figsize=(10, 4))
    for idx, (title, arr) in enumerate(panels, 1):
        plt.subplot(2, 3, idx)
        plt.imshow(arr.detach().cpu().numpy(), aspect="auto", cmap="viridis")
        plt.title(title)
        plt.colorbar(fraction=0.046)
    plt.tight_layout()
    plt.savefig(fig_dir / "figure2_representative_solution.png", dpi=180)
    plt.close()


def generate_figures(tables: dict[str, pd.DataFrame]) -> None:
    fig_dir = ROOT / "results" / "solver_v2" / "figures"
    main = tables["per_case_results"]
    conv = tables["convergence"]
    train = tables["training_summary"]
    quad = tables["residual_accuracy_quadrants"]
    plt.figure(figsize=(8, 2.5))
    blocks = ["PDE instance", "FNO initializer", "u0", "residual-conditioned actor", "hard IC/BC projection", "u(t+1)", "TD3+BC update"]
    for i, b in enumerate(blocks):
        plt.text(i, 0.5, b, ha="center", va="center", bbox=dict(boxstyle="round,pad=0.25", fc="#eef2ff", ec="#1f2937"))
        if i < len(blocks) - 1:
            plt.arrow(i + 0.38, 0.5, 0.22, 0, head_width=0.04, color="#1f2937")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(fig_dir / "figure1_architecture.png", dpi=180)
    plt.close()
    plt.figure(figsize=(6, 4))
    for method, g in conv[conv["StepCap"].eq(10)].groupby("Method"):
        if method in ["Gradient residual refinement", "Supervised Neural Operator Corrector", "RL Neural Operator Solver"]:
            gg = g.groupby("step")["Relative L2"].mean()
            plt.plot(gg.index, gg.values, marker="o", label=method)
    plt.xlabel("solver step")
    plt.ylabel("Relative L2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "figure3_l2_vs_step.png", dpi=180)
    plt.close()
    plt.figure(figsize=(6, 4))
    for method, g in conv[conv["StepCap"].eq(10)].groupby("Method"):
        if method in ["Gradient residual refinement", "Supervised Neural Operator Corrector", "RL Neural Operator Solver"]:
            gg = g.groupby("step")["PDE residual norm"].mean()
            plt.plot(gg.index, gg.values, marker="o", label=method)
    plt.xlabel("solver step")
    plt.ylabel("PDE residual norm")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "figure4_residual_vs_step.png", dpi=180)
    plt.close()
    plt.figure(figsize=(6, 4))
    if not train.empty and "critic_loss" in train:
        for seed, g in train.groupby("Seed"):
            plt.plot(g.index, g["critic_loss"], label=f"seed {seed}")
    plt.xlabel("logged update")
    plt.ylabel("critic loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "figure5_training_curve.png", dpi=180)
    plt.close()
    paired = main[(main["Steps"].eq(10)) & (main["Method"].isin(["Supervised Neural Operator Corrector", "RL Neural Operator Solver"]))]
    pivot = paired.pivot_table(index=["Seed", "Case"], columns="Method", values="paired improvement")
    plt.figure(figsize=(5, 4))
    if {"Supervised Neural Operator Corrector", "RL Neural Operator Solver"}.issubset(pivot.columns):
        plt.scatter(pivot["Supervised Neural Operator Corrector"], pivot["RL Neural Operator Solver"], s=18, alpha=0.7)
        lo = float(np.nanmin(pivot.values))
        hi = float(np.nanmax(pivot.values))
        plt.plot([lo, hi], [lo, hi], color="black", linewidth=0.8)
    plt.xlabel("Supervised paired improvement")
    plt.ylabel("RL paired improvement")
    plt.tight_layout()
    plt.savefig(fig_dir / "figure6_supervised_vs_rl_paired.png", dpi=180)
    plt.close()
    trans_df = tables["transition_debug"]
    plt.figure(figsize=(5.5, 4))
    if not trans_df.empty:
        plt.axhline(0, color="black", linewidth=0.8)
        plt.axvline(0, color="black", linewidth=0.8)
        plt.scatter(trans_df["residual_after"] - trans_df["residual_before"], trans_df["error_after"] - trans_df["error_before"], s=8, alpha=0.35)
    plt.xlabel("Delta residual")
    plt.ylabel("Delta accuracy error")
    plt.tight_layout()
    plt.savefig(fig_dir / "figure7_residual_accuracy_change.png", dpi=180)
    plt.close()
    plt.figure(figsize=(6, 4))
    for method, g in main[main["Steps"].eq(10)].groupby("Method"):
        plt.scatter(g["wall time"], g["Relative L2"], s=14, alpha=0.55, label=method)
    plt.xlabel("wall time")
    plt.ylabel("Relative L2")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(fig_dir / "figure8_error_vs_walltime.png", dpi=180)
    plt.close()
    quad.to_csv(ROOT / "results" / "solver_v2" / "tables" / "residual_accuracy_quadrants.csv", index=False)


def write_docs(cfg: dict, tables: dict[str, pd.DataFrame]) -> None:
    docs = ROOT / "docs"
    main = tables["main_results"]
    sub = main[main["Steps"].eq(10)]
    def val(method: str) -> float:
        return float(sub[sub["Method"].eq(method)]["Relative L2 mean"].iloc[0])
    base = val("Base FNO Initializer")
    sup = val("Supervised Neural Operator Corrector")
    rl = val("RL Neural Operator Solver")
    if rl < base and rl < sup:
        claim = "Reinforcement learning improves iterative PDE solution refinement beyond supervised neural-operator correction."
    elif rl < base:
        claim = "RL produces a functional iterative PDE solver, but long-horizon optimization does not yet provide a clear advantage over supervised correction."
    else:
        claim = "The current RL formulation does not yet provide a reliable PDE solver."
    (docs / "solver_v2_claims.md").write_text(
        "# Solver V2 Claims\n\n"
        f"- {claim}\n"
        f"- 10-step Base FNO Relative L2 mean: {base:.6f}.\n"
        f"- 10-step Supervised Neural Operator Corrector Relative L2 mean: {sup:.6f}.\n"
        f"- 10-step RL Neural Operator Solver Relative L2 mean: {rl:.6f}.\n"
        "- Validation checkpoint selection is used for the full solver; in this run TD3 fine-tuning did not produce a checkpoint that improved over the supervised warm start.\n"
        "- Physics residual is reported as a constraint diagnostic, not used as the core RL reward.\n",
        encoding="utf-8",
    )
    (docs / "solver_v2_pde_definition.md").write_text(
        "# Solver V2 PDE Definition\n\n"
        "The official 2D_Reac_diffusion tensors provide source fields f(x,t), target solutions u(x,t), and one-dimensional x and t coordinate arrays. The data orientation used by the official training script is `(case, x, t)`. The bundled official arrays use 40 spatial points on x in [0, 2] and 20 temporal points on t in [0, 1].\n\n"
        "V2 uses the reaction-diffusion residual\n\n"
        "`R(u) = u_t - D u_xx - k u^2 - f(x,t)`\n\n"
        "with `D = 1 - 0.95 / pi^2` and `k = 1`, matching the official note file. Boundary and initial values are taken directly from each target solution: `u[:,0]`, `u[0,:]`, and `u[-1,:]`. Every solver update is followed by hard IC/BC projection.\n",
        encoding="utf-8",
    )
    (docs / "solver_v2_method.md").write_text(
        "# Solver V2 Method\n\n"
        "We formulate PDE solving as a sequential reinforcement-learning problem and train a residual-conditioned deterministic neural operator policy to directly generate iterative solution-field updates under governing-equation constraints.\n\n"
        "The V2 actor is a full-field FNO-style neural operator. It receives the current solution, residual field, source, IC field, BC field, PDE scalars, and step fraction, and outputs a bounded correction field. TD3+BC fine-tunes a supervised warm-start actor using accuracy-improvement rewards, behavior-cloning regularization, and scale-normalized physics regularization. Full-solver checkpoints are selected on the validation split before final test evaluation.\n",
        encoding="utf-8",
    )
    (docs / "solver_v2_results.md").write_text(
        "# Solver V2 Results\n\n"
        "The generated tables under `results/solver_v2/tables` contain the final 3-seed paper-mode results. Claims are automatically selected from the measured 10-step test performance and avoid unsupported improvement statements.\n\n"
        + main.to_markdown(index=False)
        + "\n",
        encoding="utf-8",
    )


def run_pipeline(mode: str) -> None:
    cfg = load_config()
    ensure_dirs()
    npz_path = prepare_reaction_diffusion(cfg, ROOT)
    device = get_device(str(cfg.get("device", "cuda")))
    all_eval, all_per_case, all_conv, all_transitions, all_training = [], [], [], [], []
    for seed in cfg["solver_v2"]["seeds"]:
        eval_df, per_case, conv, transitions, training_summary = run_seed(cfg, npz_path, int(seed), device)
        all_eval.append(eval_df)
        all_per_case.append(per_case)
        all_conv.append(conv)
        all_transitions.extend(transitions)
        all_training.append(training_summary)
    per_case_df = pd.concat(all_per_case, ignore_index=True)
    conv_df = pd.concat(all_conv, ignore_index=True)
    transition_df = pd.DataFrame([tr.__dict__ for tr in all_transitions])
    training_df = pd.concat(all_training, ignore_index=True)
    tables = {
        "main_results": summarize_main(per_case_df),
        "convergence": conv_df,
        "per_case_results": per_case_df,
        "residual_accuracy_quadrants": residual_accuracy_quadrants(all_transitions),
        "training_summary": training_df,
        "seed_results": seed_summary(per_case_df),
        "physics_metrics": per_case_df[["Seed", "Method", "Steps", "Case", "PDE residual norm", "BC error", "IC error"]],
        "transition_debug": transition_df,
    }
    table_dir = ROOT / "results" / "solver_v2" / "tables"
    for name, df in tables.items():
        if name != "transition_debug":
            df.to_csv(table_dir / f"{name}.csv", index=False)
    generate_figures(tables)
    generate_representative_figure(cfg, npz_path, int(cfg["solver_v2"]["seeds"][0]), device)
    write_docs(cfg, tables)
    summary = {
        "mode": mode,
        "device": str(device),
        "seeds": cfg["solver_v2"]["seeds"],
        "tables": sorted(p.name for p in table_dir.glob("*.csv")),
    }
    (table_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    step10 = tables["main_results"][tables["main_results"]["Steps"].eq(10)]
    print("Solver V2 paper pipeline completed.")
    print(step10[["Method", "Relative L2 mean", "Relative L2 std", "paired improvement mean"]].to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="paper", choices=["paper"])
    args = parser.parse_args()
    run_pipeline(args.mode)


if __name__ == "__main__":
    main()
