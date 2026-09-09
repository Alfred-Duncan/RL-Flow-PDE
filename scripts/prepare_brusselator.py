from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.official_brusselator import EXPECTED_NAME, OFFICIAL_DRIVE_URL, expected_data_path, require_official_data

FILE_URL = "https://drive.google.com/file/d/1XHN1skIuZQwEFuyHmAPlthBi7NDEW9M5/view"


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare official LNO 3D_Brusselator data.")
    parser.add_argument("--download", action="store_true", help="Download the official shared NPZ through gdown.")
    args = parser.parse_args()
    path = expected_data_path(ROOT)
    if args.download:
        path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([sys.executable, "-m", "gdown", "--fuzzy", FILE_URL, "-O", str(path)], check=True)
    require_official_data(path)
    print(f"Official 3D_Brusselator data ready: {path}")
    print(f"Source: {OFFICIAL_DRIVE_URL}")


if __name__ == "__main__":
    main()
