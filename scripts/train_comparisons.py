"""Sprint 2: Experiments E2 (YOLOv8s) and E3 (YOLO11n, optional comparison).

Usage:
    python scripts/train_comparisons.py [--only e2_stronger_v8s|e3_compare_11n]

E3 failures are tolerated (the architecture comparison is optional per the
project plan); E2 failures are fatal.
"""
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.train import run_experiment
from src.utils import setup_logging

log = setup_logging("flameguard.train-comparisons")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", choices=["e2_stronger_v8s", "e3_compare_11n"])
    args = parser.parse_args()

    experiments = [args.only] if args.only else ["e2_stronger_v8s", "e3_compare_11n"]
    for exp_id in experiments:
        try:
            run_experiment(exp_id)
        except Exception:
            if exp_id == "e3_compare_11n":
                log.warning("Optional experiment %s failed - continuing:\n%s",
                            exp_id, traceback.format_exc())
            else:
                raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
