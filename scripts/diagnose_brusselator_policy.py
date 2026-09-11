from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_solver_v5_brusselator import (
    ACTIONS, beam_teacher, build_selector_data, cfg, data, feasible, future_return,
    gains, paths, refiner, run_case, selector_features, selector_fields, stats, train_bc, train_coarse, train_local, train_selector,
)
from src.solver_v5_brusselator.env import FrozenBundle
from src.solver_v5.models.macro_policy import MacroPolicy
from src.utils.seed import get_device, set_seed


def finite(tensor: torch.Tensor) -> bool:
    return bool(torch.isfinite(tensor).all())


def first_invalid(observation: tuple[torch.Tensor, ...]) -> list[str]:
    names = ("fields", "selector_stats", "selector_embedding", "context")
    return [name for name, value in zip(names, observation) if not finite(value)]


@torch.no_grad()
def main() -> None:
    config = cfg()
    device = get_device(config["device"])
    set_seed(42)
    dataset = data(config)
    _, checkpoint_dir = paths()
    summary_stats = stats(dataset)
    coarse = train_coarse(dataset, config, summary_stats, device, checkpoint_dir)
    local = train_local(dataset, coarse, config, summary_stats, device, checkpoint_dir)
    refiner_model = refiner(coarse, local, config, summary_stats)
    selector, selector_stats = train_selector(build_selector_data(dataset, refiner_model, config, checkpoint_dir), config, device, checkpoint_dir)
    bc = train_bc(beam_teacher(dataset, refiner_model, selector, selector_stats, config, checkpoint_dir), config, device, checkpoint_dir)
    policy = MacroPolicy().to(device)
    rvpi_path = checkpoint_dir / "rvpi.pt"
    policy.load_state_dict(torch.load(rvpi_path, map_location=device)["model"] if rvpi_path.exists() else bc.state_dict())
    policy.eval()

    report: dict[str, object] = {
        "bc_parameters_finite": all(finite(value) for value in bc.state_dict().values()),
        "policy_parameters_finite": all(finite(value) for value in policy.state_dict().values()),
        "traces": [],
    }
    for source, source_policy, case in (("BeamBC", bc, dataset.case_ids("train")[0]), ("RVPI", policy, dataset.case_ids("train")[1])):
        _, _, trace = run_case(dataset, refiner_model, selector, selector_stats, source_policy, case, source, 42, keep=True)
        invalid = []
        for state in trace:
            obs = tuple(value.to(device) for value in state["obs"])
            logits = policy(*obs)
            broken = first_invalid(obs)
            if not finite(logits):
                broken.append("policy_logits")
            if broken:
                invalid.append({"step": state["step"], "invalid": broken})
        report["traces"].append({"source": source, "case": case, "steps": len(trace), "invalid_step_count": len(invalid), "first_invalid": invalid[:1]})

    state = run_case(dataset, refiner_model, selector, selector_stats, policy, dataset.case_ids("train")[0], "RVPI", 42, keep=True)[2][0]
    allowed = feasible(state["remaining"] + state["q"], 37 - state["step"])
    values = torch.tensor([future_return(dataset, refiner_model, selector, selector_stats, policy, state, action) for action in allowed], device=device)
    values = torch.nan_to_num(values, nan=-1e6, posinf=-1e6, neginf=-1e6)
    margin = max(float(values.std(unbiased=False)) * 0.1, 1e-4)
    target = torch.softmax((values - values.max()) / margin, 0)
    obs = tuple(value.to(device) for value in state["obs"])
    logits = policy(*obs)[0]
    mask = torch.tensor([action in allowed for action in ACTIONS], device=device)
    available = logits[mask]
    old_logits = policy(*obs)[0].detach()[mask]
    kl = F.kl_div(F.log_softmax(available, 0), target, reduction="batchmean")
    regularizer = F.mse_loss(F.log_softmax(available, 0), F.log_softmax(old_logits, 0))
    report["first_state"] = {
        "allowed": allowed,
        "returns": values.cpu().tolist(),
        "margin": margin,
        "target": target.cpu().tolist(),
        "logits": logits.cpu().tolist(),
        "finite": {"returns": finite(values), "target": finite(target), "logits": finite(logits), "kl": finite(kl), "regularizer": finite(regularizer)},
        "kl": float(kl.detach().cpu()) if finite(kl) else None,
        "regularizer": float(regularizer.detach().cpu()) if finite(regularizer) else None,
    }
    label_probe = {"fields_finite": True, "features_finite": True, "gains_finite": True, "field_abs_max": 0.0, "feature_abs_max": 0.0, "gain_abs_max": 0.0}
    for case in dataset.case_ids("train")[:4]:
        previous = current = dataset.frame(case, 0).unsqueeze(0).to(device)
        for step in range(38):
            target = dataset.frame(case, step + 1).unsqueeze(0).to(device)
            bundle = FrozenBundle.create(refiner_model, dataset.forcing(case, step).reshape(1).to(device), previous, current, step / 38)
            field = selector_fields(bundle)
            features = selector_features(bundle, [], step / 38)
            gain = gains(bundle, target, [])
            label_probe["fields_finite"] &= finite(field)
            label_probe["features_finite"] &= finite(features)
            label_probe["gains_finite"] &= finite(gain)
            label_probe["field_abs_max"] = max(label_probe["field_abs_max"], float(field.abs().max()))
            label_probe["feature_abs_max"] = max(label_probe["feature_abs_max"], float(features.abs().max()))
            label_probe["gain_abs_max"] = max(label_probe["gain_abs_max"], float(gain.abs().max()))
            previous, current = current, bundle.apply_set([])
    report["current_label_probe"] = label_probe
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
