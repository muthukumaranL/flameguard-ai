"""Run the remaining training experiments back-to-back so the GPU never idles.

Order: E2 (YOLOv8s) -> E4d (probe control) -> E4a/E4b/E4c (tuning probes).
E3 (YOLO11n) is attempted last and is allowed to fail without blocking.
E5 (final) is launched separately after the probe results are reviewed.

Usage:
    python scripts/run_training_chain.py
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.train import run_experiment
from src.utils import setup_logging

log = setup_logging("flameguard.chain")

# Cheapest-first: the probe study and the equal-cost architecture comparison run
# before the expensive VRAM-limited YOLOv8s capacity probe, so a time overrun
# degrades the least important experiment rather than the most important ones.
REQUIRED = ["e4d_probe_baseline", "e4a_probe_adamw", "e4b_probe_augment",
            "e4c_probe_loss", "e3_compare_11n"]
OPTIONAL = ["e2_stronger_v8s"]


def main() -> int:
    failures: list[str] = []
    for exp_id in REQUIRED:
        try:
            row = run_experiment(exp_id)
            log.info("OK %s | mAP50=%.4f mAP50-95=%.4f R=%.4f (%s)",
                     exp_id, row["map50"], row["map50_95"], row["recall"],
                     row["duration"])
        except Exception:
            failures.append(exp_id)
            log.error("FAILED %s:\n%s", exp_id, traceback.format_exc())

    for exp_id in OPTIONAL:
        try:
            run_experiment(exp_id)
        except Exception:
            log.warning("Optional experiment %s failed - continuing without it", exp_id)

    if failures:
        log.error("Chain finished with failures: %s", failures)
        return 1
    log.info("Training chain complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
