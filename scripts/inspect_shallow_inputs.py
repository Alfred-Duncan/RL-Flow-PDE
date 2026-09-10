from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "data" / "official" / "shallow_water" / "shallow-water-256x256x72_1.npz"
OUTPUT = ROOT / "results" / "solver_v4" / "shallow_water" / "shallow_input_diagnostic.json"


def main() -> None:
    if not ARCHIVE.exists():
        raise FileNotFoundError(ARCHIVE)
    with np.load(ARCHIVE, allow_pickle=False) as archive:
        inputs, outputs = np.asarray(archive["inputs"]), np.asarray(archive["outputs"])
    initial = outputs[:, 0]
    flattened_inputs, flattened_initial = inputs.reshape(len(inputs), -1), initial.reshape(len(initial), -1)
    correlations = [float(np.corrcoef(flattened_inputs[index], flattened_initial[index])[0, 1]) for index in range(min(len(inputs), len(initial))) if flattened_inputs[index].std() > 0 and flattened_initial[index].std() > 0]
    payload = {
        "input_shape": list(inputs.shape), "output_shape": list(outputs.shape),
        "input_min": float(inputs.min()), "input_max": float(inputs.max()), "input_std": float(inputs.std()),
        "input_case_variation_mean_std": float(flattened_inputs.mean(1).std()),
        "input_case_variation_l2_std": float(np.linalg.norm(flattened_inputs, axis=1).std()),
        "input_matches_output_t0_shape": list(inputs.shape[1:]) == list(initial.shape[1:]),
        "input_output_t0_mean_correlation": float(np.mean(correlations)) if correlations else None,
        "interpretation": "Input arrays vary across cases. They are inspected only here; Solver V4 does not alter its Markov state with this diagnostic.",
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
