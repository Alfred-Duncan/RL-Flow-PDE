from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.solver_v5_brusselator.data import OfficialBrusselatorV5


def main() -> None:
    data = OfficialBrusselatorV5(ROOT / "data" / "official" / "brusselator" / "Brusselator_force_train.npz")
    payload = {**data.metadata, "train_cases": len(data.case_ids("train")), "validation_cases": len(data.case_ids("val")), "test_cases": len(data.case_ids("test")), "forcing_at_one_time_shape": tuple(data.forcing(0, 0).shape), "frame_shape": tuple(data.frame(0, 0).shape), "interpretation": "The 39-axis is physical time. Official inputs are time-varying scalar forcing values, not a separate static condition field."}
    result = ROOT / "results" / "solver_v5" / "brusselator"; result.mkdir(parents=True, exist_ok=True)
    (result / "official_data_diagnostic.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
