from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.solver_v3.data.official_shallow_water import OFFICIAL_DRIVE_URL, RAW_FILES, OfficialShallowWater

FOLDER_URL = "https://drive.google.com/drive/folders/14NQ3-i7RJJV9d9PuZT6CzK2If437e8Vy"


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare official LNO 3D_shallow raw data.")
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    target = ROOT / "data" / "official" / "shallow_water"
    if args.download:
        target.mkdir(parents=True, exist_ok=True)
        subprocess.run([sys.executable, "-m", "gdown", "--folder", FOLDER_URL, "-O", str(target)], check=True)
    missing = [str(target / name) for name in RAW_FILES if not (target / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing official files: {missing}. Source: {OFFICIAL_DRIVE_URL}")
    data = OfficialShallowWater(target)
    print(f"Official Shallow Water ready: {data.metadata}")


if __name__ == "__main__":
    main()
