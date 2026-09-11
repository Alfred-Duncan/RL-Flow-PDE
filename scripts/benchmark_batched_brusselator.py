from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_solver_v5_brusselator import (
    cfg,
    data,
    feasible,
    future_return,
    models,
    refiner,
    stats,
    train_bc,
    train_coarse,
    train_local,
    train_selector,
    build_selector_data,
    beam_teacher,
    paths,
)
from src.solver_v5.models.macro_policy import MacroPolicy
from src.solver_v5_brusselator.batched_rollout import BatchedFrozenBundle, batched_future_return
from src.solver_v5_brusselator.env import FrozenBundle
from src.utils.seed import get_device


def load_policy(checkpoint_dir: Path, device: torch.device, fallback: MacroPolicy) -> MacroPolicy:
    path = checkpoint_dir / "rvpi.pt"
    if not path.exists():
        return fallback
    model = MacroPolicy().to(device)
    model.load_state_dict(torch.load(path, map_location=device)["model"])
    return model.eval()


def bundle_equivalence(refiner_model, dataset, device: torch.device) -> float:
    cases = [0, 1, 2, 3]
    previous = torch.stack([dataset.frame(case, 0) for case in cases]).to(device)
    current = previous.clone()
    force = torch.stack([dataset.forcing(case, 0) for case in cases]).to(device)
    batched = BatchedFrozenBundle.create(refiner_model, force, previous, current, torch.zeros(len(cases), device=device))
    maximum = 0.0
    for row, case in enumerate(cases):
        legacy = FrozenBundle.create(refiner_model, dataset.forcing(case, 0).reshape(1).to(device), previous[row : row + 1], current[row : row + 1], 0.0)
        maximum = max(maximum, float((legacy.provisional - batched.provisional[row : row + 1]).abs().max()))
        maximum = max(maximum, float((legacy.corrections - batched.corrections[row]).abs().max()))
        maximum = max(maximum, float((legacy.apply_set([0, 6, 24]) - batched.apply_sets([[0, 6, 24] if item == row else [] for item in range(len(cases))])[row : row + 1]).abs().max()))
    return maximum


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify and benchmark the isolated batched Brusselator rollout evaluator.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--immediate", action="store_true", help="Only benchmark one transition instead of a full continuation.")
    args = parser.parse_args()

    config = cfg()
    device = get_device(config["device"])
    dataset = data(config)
    checkpoint_dir = paths()[1]
    summary_stats = stats(dataset)
    coarse = train_coarse(dataset, config, summary_stats, device, checkpoint_dir)
    local = train_local(dataset, coarse, config, summary_stats, device, checkpoint_dir)
    refiner_model = refiner(coarse, local, config, summary_stats)
    selector, selector_stats = train_selector(build_selector_data(dataset, refiner_model, config, checkpoint_dir), config, device, checkpoint_dir)
    bc = train_bc(beam_teacher(dataset, refiner_model, selector, selector_stats, config, checkpoint_dir), config, device, checkpoint_dir)
    policy = load_policy(checkpoint_dir, device, bc)

    batch_size = min(args.batch_size, len(dataset.case_ids("train")))
    states = []
    for case in dataset.case_ids("train")[:batch_size]:
        state = dataset.frame(case, 0).unsqueeze(0)
        states.append({"case": case, "step": 0, "remaining": 76, "q": 0, "previous": state, "current": state})
    actions = [2] * batch_size

    difference = bundle_equivalence(refiner_model, dataset, device)
    torch.cuda.synchronize(device) if device.type == "cuda" else None
    start = time.perf_counter()
    serial = [future_return(dataset, refiner_model, selector, selector_stats, policy, state, action, args.immediate) for state, action in zip(states, actions)]
    torch.cuda.synchronize(device) if device.type == "cuda" else None
    serial_seconds = time.perf_counter() - start
    torch.cuda.synchronize(device) if device.type == "cuda" else None
    start = time.perf_counter()
    parallel = batched_future_return(dataset, refiner_model, selector, selector_stats, policy, states, actions, feasible, args.immediate)
    torch.cuda.synchronize(device) if device.type == "cuda" else None
    batched_seconds = time.perf_counter() - start
    return_difference = max(abs(left - right) for left, right in zip(serial, parallel))
    relative_return_difference = max(abs(left - right) / max(abs(left), 1.0) for left, right in zip(serial, parallel))
    print({
        "device": str(device),
        "batch_size": batch_size,
        "immediate": args.immediate,
        "bundle_max_abs_diff": difference,
        "bundle_diff_over_data_std": difference / float(dataset.train_states.std()),
        "return_max_abs_diff": return_difference,
        "return_max_relative_diff": relative_return_difference,
        "serial_returns": serial,
        "batched_returns": parallel,
        "serial_seconds": serial_seconds,
        "batched_seconds": batched_seconds,
        "speedup": serial_seconds / max(batched_seconds, 1e-12),
    })
    # Batched CUDA kernels need not be bit-identical to B=1 kernels.  The
    # rollout return is the behavioral contract; the field tolerance is far
    # below the data scale and catches indexing/order mistakes.
    if difference > 5e-4 or (args.immediate and return_difference > 2e-5) or (not args.immediate and relative_return_difference > 0.02):
        raise SystemExit("Batched evaluator diverges from the serial reference; do not integrate it.")


if __name__ == "__main__":
    main()
