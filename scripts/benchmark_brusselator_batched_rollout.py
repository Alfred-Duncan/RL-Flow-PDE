from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.benchmark_batched_brusselator import bundle_equivalence, load_policy
from scripts.run_solver_v5_brusselator import (
    ACTIONS, beam_teacher, build_selector_data, cfg, data, feasible, future_return,
    models, paths, refiner, run_case, select, stats, train_bc, train_coarse,
    train_local, train_selector,
)
from src.solver_v5_brusselator.batched_rollout import BatchedFrozenBundle, batched_future_return, select_batch
from src.solver_v5_brusselator.env import FrozenBundle
from src.utils.seed import get_device


def load_stack():
    config = cfg()
    device = get_device(config["device"])
    dataset = data(config)
    out, checkpoint_dir = paths()
    summary_stats = stats(dataset)
    coarse = train_coarse(dataset, config, summary_stats, device, checkpoint_dir)
    local = train_local(dataset, coarse, config, summary_stats, device, checkpoint_dir)
    refiner_model = refiner(coarse, local, config, summary_stats)
    selector, selector_stats = train_selector(build_selector_data(dataset, refiner_model, config, checkpoint_dir), config, device, checkpoint_dir)
    bc = train_bc(beam_teacher(dataset, refiner_model, selector, selector_stats, config, checkpoint_dir), config, device, checkpoint_dir)
    return config, device, dataset, out, refiner_model, selector, selector_stats, bc, load_policy(checkpoint_dir, device, bc)


@torch.no_grad()
def strict_states(dataset, refiner_model, selector, selector_stats, policy, bc, state_count: int):
    step_grid = (0, 5, 10, 15, 20, 25, 30, 35, 37)
    sources = ("RVPI", "BeamBC", "RandomMacro", "SetAwareMyopicMacro")
    traces = {}
    pool = dataset.case_ids("train")
    rows = []
    for source_index, source in enumerate(sources):
        source_policy = bc if source == "BeamBC" else policy
        for case in (pool[source_index], pool[source_index + 16]):
            _, _, trace = run_case(dataset, refiner_model, selector, selector_stats, source_policy, case, source, 7100 + source_index * 100 + case, keep=True)
            traces[source, case] = trace
        for step_index, step in enumerate(step_grid):
            case = pool[source_index] if step_index % 2 == 0 else pool[source_index + 16]
            state = dict(traces[source, case][step])
            state["Source"] = source
            rows.append(state)
            if len(rows) >= state_count:
                return rows
    return rows


def state_bundle(dataset, refiner_model, state):
    device = next(refiner_model.coarse.parameters()).device
    return FrozenBundle.create(
        refiner_model,
        dataset.forcing(state["case"], state["step"]).reshape(1).to(device),
        state["previous"].to(device),
        state["current"].to(device),
        state["step"] / 38,
    )


def jaccard(left: list[int], right: list[int]) -> float:
    union = set(left) | set(right)
    return 1.0 if not union else len(set(left) & set(right)) / len(union)


@torch.no_grad()
def strict_equivalence(dataset, refiner_model, selector, selector_stats, policy, bc, out: Path, batch_size: int, state_count: int):
    states = strict_states(dataset, refiner_model, selector, selector_stats, policy, bc, state_count)
    jobs = []
    for state_index, state in enumerate(states):
        before_action_budget = state["remaining"] + state["q"]
        for action in feasible(before_action_budget, 37 - state["step"]):
            bundle = state_bundle(dataset, refiner_model, state)
            serial_patches, _, _ = select(selector, selector_stats, bundle, action, state["step"] / 38)
            jobs.append({"state_index": state_index, "state": state, "q": action, "serial_patches": serial_patches})

    grouped = defaultdict(list)
    for job in jobs:
        grouped[job["state"]["step"]].append(job)
    for step, group in grouped.items():
        for start in range(0, len(group), batch_size):
            chunk = group[start : start + batch_size]
            device = next(refiner_model.coarse.parameters()).device
            previous = torch.cat([job["state"]["previous"].to(device) for job in chunk])
            current = torch.cat([job["state"]["current"].to(device) for job in chunk])
            force = torch.stack([dataset.forcing(job["state"]["case"], step) for job in chunk]).to(device)
            time_vector = torch.full((len(chunk),), step / 38, device=device, dtype=current.dtype)
            bundle = BatchedFrozenBundle.create(refiner_model, force, previous, current, time_vector)
            counts = torch.tensor([job["q"] for job in chunk], device=device)
            patch_sets, _ = select_batch(selector, selector_stats, bundle, counts, time_vector)
            returns = batched_future_return(dataset, refiner_model, selector, selector_stats, policy, [job["state"] for job in chunk], [job["q"] for job in chunk], feasible)
            for job, patches, value in zip(chunk, patch_sets, returns):
                job["batched_patches"] = patches
                job["batched_return"] = value

    records = []
    for job in jobs:
        serial = future_return(dataset, refiner_model, selector, selector_stats, policy, job["state"], job["q"])
        state = job["state"]
        records.append({
            "State": job["state_index"], "Source": state["Source"], "Case": state["case"], "Step": state["step"], "Q": job["q"],
            "SerialReturn": serial, "BatchedReturn": job["batched_return"], "AbsDifference": abs(serial - job["batched_return"]),
            "ExactPatchSetAgreement": job["serial_patches"] == job["batched_patches"],
            "JaccardAgreement": jaccard(job["serial_patches"], job["batched_patches"]),
        })
    frame = pd.DataFrame(records)
    frame.to_csv(out / "batched_equivalence.csv", index=False)
    best, pairs = [], []
    for _, group in frame.groupby("State"):
        serial_best = int(group.loc[group.SerialReturn.idxmax(), "Q"])
        batched_best = int(group.loc[group.BatchedReturn.idxmax(), "Q"])
        best.append(serial_best == batched_best)
        values = group.sort_values("Q")
        serial_values, batched_values = values.SerialReturn.to_numpy(), values.BatchedReturn.to_numpy()
        for left in range(len(values)):
            for right in range(left + 1, len(values)):
                pairs.append(np.sign(serial_values[left] - serial_values[right]) == np.sign(batched_values[left] - batched_values[right]))
    summary = {
        "states": int(frame.State.nunique()), "candidate_continuations": len(frame), "batch_size": batch_size,
        "field_max_abs_diff": bundle_equivalence(refiner_model, dataset, next(refiner_model.coarse.parameters()).device),
        "field_diff_over_train_state_std": bundle_equivalence(refiner_model, dataset, next(refiner_model.coarse.parameters()).device) / float(dataset.train_states.std()),
        "best_action_agreement": float(np.mean(best)), "pairwise_action_order_agreement": float(np.mean(pairs)),
        "exact_patch_set_agreement": float(frame.ExactPatchSetAgreement.mean()), "mean_jaccard_agreement": float(frame.JaccardAgreement.mean()),
        "max_return_abs_difference": float(frame.AbsDifference.max()),
    }
    summary["passed"] = bool(
        summary["field_max_abs_diff"] <= 5e-4
        and summary["best_action_agreement"] >= 0.99
        and summary["pairwise_action_order_agreement"] >= 0.99
        and summary["exact_patch_set_agreement"] >= 0.98
        and summary["mean_jaccard_agreement"] >= 0.995
    )
    (out / "batched_equivalence_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Strict Brusselator batched-rollout equivalence gate.")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--states", type=int, default=36)
    args = parser.parse_args()
    if not args.strict:
        raise SystemExit("Use --strict to run the integration gate.")
    _, _, dataset, out, refiner_model, selector, selector_stats, bc, policy = load_stack()
    summary = strict_equivalence(dataset, refiner_model, selector, selector_stats, policy, bc, out, args.batch_size, max(32, args.states))
    print(json.dumps(summary, indent=2))
    if not summary["passed"]:
        raise SystemExit("Strict batched equivalence gate failed; serial training remains required.")


if __name__ == "__main__":
    main()
