"""Sprint 2: Experiment E1 - YOLOv8n baseline (transfer learning).

Usage:
    python scripts/train_baseline.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.train import run_experiment


def main() -> int:
    run_experiment("e1_baseline_v8n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
