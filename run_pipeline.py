from __future__ import annotations

import argparse

from scripts.experiment import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="paper", choices=["paper"])
    args = parser.parse_args()
    run_pipeline(args.mode)


if __name__ == "__main__":
    main()
