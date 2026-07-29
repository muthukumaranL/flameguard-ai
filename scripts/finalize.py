"""One-shot finalisation after a new model is trained.

Re-benchmarks every completed experiment, re-selects the final model, and - only
when that selection actually changes the deployed weights - re-runs the
single-shot test evaluation and the error analysis. It then regenerates the
report, slides and samples from the refreshed artefacts, verifies the project,
and rebuilds the submission ZIP.

The winner-change guard matters: the held-out test set must be evaluated exactly
once per final model, so we do NOT re-touch it when the winner is unchanged.

Usage:
    python scripts/finalize.py [--group 07] [--skip-samples]
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

from src import paths
from src.utils import setup_logging

log = setup_logging("flameguard.finalize")
PY = sys.executable
ROOT = paths.PROJECT_ROOT


def _winner() -> str | None:
    p = paths.FINAL_MODEL_METADATA_PATH
    if not p.exists():
        return None
    return yaml.safe_load(p.read_text(encoding="utf-8")).get("experiment_id")


def _run(script: str, *args: str) -> None:
    cmd = [PY, str(ROOT / "scripts" / script), *args]
    log.info("RUN %s", " ".join(cmd[1:]))
    result = subprocess.run(cmd, cwd=str(ROOT))
    if result.returncode != 0:
        raise SystemExit(f"{script} failed with exit code {result.returncode}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", default="##")
    parser.add_argument("--skip-samples", action="store_true",
                        help="do not regenerate sample predictions (slow)")
    args = parser.parse_args()

    before = _winner()
    log.info("Final model before re-benchmark: %s", before)

    log.info("=== 1/6 Benchmark + re-select final model ===")
    _run("benchmark.py")
    after = _winner()
    log.info("Final model after re-benchmark: %s (was %s)", after, before)

    if after != before:
        log.info("=== Winner CHANGED -> re-evaluating test set + error analysis ===")
        _run("evaluate_final.py")
        _run("error_analysis.py")
        if not args.skip_samples:
            _run("generate_samples.py")
    else:
        log.info("=== Winner unchanged -> test set NOT re-touched (evaluated once) ===")

    log.info("=== 4/6 Regenerate report ===")
    _run("generate_report.py")
    log.info("=== 5/6 Regenerate slides ===")
    _run("generate_slides.py")

    log.info("=== 6/6 Verify + package ===")
    _run("verify_project.py")
    _run("package_submission.py", "--group", args.group)
    log.info("Finalisation complete. Final model: %s", after)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
