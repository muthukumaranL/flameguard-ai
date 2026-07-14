"""Stage 1 runner: import raw dataset, validate it, repair leakage, re-split.

Usage:
    python scripts/validate_dataset.py [--source <path-to-roboflow-export>]

Steps:
  1. Copy the Roboflow v1 export into data/raw/fire_and_smoke_v1 (originals untouched).
  2. Full integrity audit of the raw dataset  -> outputs/dataset_validation/raw/
  3. Leakage-aware re-split                   -> data/processed/fire_smoke_resplit/
  4. Re-audit of the processed dataset        -> outputs/dataset_validation/processed/
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import paths
from src.dataset_validator import validate_dataset
from src.resplit import resplit_dataset
from src.utils import setup_logging

log = setup_logging("flameguard.stage1")

DEFAULT_SOURCE = (
    paths.PROJECT_ROOT.parent / "fire and smoke.v1i.yolov8"
)


def import_raw(source: Path) -> Path:
    """Copy the Roboflow export into data/raw (skip when already imported)."""
    dest = paths.RAW_DATASET_DIR
    if dest.exists() and (dest / "data.yaml").exists():
        log.info("Raw dataset already imported at %s", paths.rel_to_root(dest))
        return dest
    if not source.exists():
        raise FileNotFoundError(
            f"Source dataset not found: {source}\n"
            "Pass --source <path to the extracted Roboflow YOLOv8 export>."
        )
    log.info("Copying dataset %s -> %s (originals preserved)", source, dest)
    shutil.copytree(source, dest)
    provenance = {
        "source_path": str(source),
        "imported_at": datetime.now(timezone.utc).isoformat(),
        "roboflow_project": "fire-detector-cqdzi/fire-and-smoke-b5lli",
        "roboflow_version": 1,
        "license": "CC BY 4.0",
        "url": "https://universe.roboflow.com/fire-detector-cqdzi/fire-and-smoke-b5lli/dataset/1",
    }
    with (dest / "PROVENANCE.json").open("w", encoding="utf-8") as fh:
        json.dump(provenance, fh, indent=2)
    return dest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE,
                        help="Path to the extracted Roboflow YOLOv8 export")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-resplit", action="store_true",
                        help="Only run the raw audit")
    args = parser.parse_args()

    paths.ensure_output_dirs()
    raw_dir = import_raw(args.source)

    log.info("=== Auditing raw dataset ===")
    raw_report = validate_dataset(raw_dir, paths.VALIDATION_OUTPUT_DIR / "raw")
    if not raw_report["min_200_images_per_class"]:
        log.error("Dataset fails the >=200 images/class requirement: %s",
                  raw_report["images_per_class_total"])
        return 1

    if args.skip_resplit:
        return 0

    log.info("=== Re-splitting (leakage repair) ===")
    resplit_dataset(raw_dir, paths.PROCESSED_DATASET_DIR,
                    paths.VALIDATION_OUTPUT_DIR, seed=args.seed)

    log.info("=== Auditing processed dataset ===")
    processed_report = validate_dataset(paths.PROCESSED_DATASET_DIR,
                                        paths.VALIDATION_OUTPUT_DIR / "processed")
    if not processed_report["min_200_images_per_class"]:
        log.error("Processed split fails >=200 images/class - investigate before training")
        return 1
    log.info("Stage 1 complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
