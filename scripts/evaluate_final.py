"""Final evaluation pipeline for the selected model.

Order matters:
  1. Confidence-threshold analysis on the VALIDATION split -> choose threshold.
  2. One-time evaluation on the held-out TEST split (never touched before).
  3. Wall-clock inference speed on GPU (if present) and CPU.
  4. Update model metadata + app default threshold with measured values.

Usage:
    python scripts/evaluate_final.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

from src import paths
from src.evaluate import evaluate_split, measure_inference_speed, threshold_analysis
from src.utils import pick_device, setup_logging

log = setup_logging("flameguard.eval-final")


def choose_threshold(df) -> float:
    """Pick the operating threshold from the validation sweep.

    Rule: among thresholds whose F1 is within F1_EPS of the maximum, keep only
    those whose recall is within RECALL_EPS of the best of that near-optimal set
    (never trade away real recall for precision on a safety detector), then choose
    the one with the FEWEST false positives. This resolves the common case where
    two adjacent thresholds have essentially equal F1 and identical recall but very
    different false-alarm rates - the higher threshold is strictly preferable.
    """
    F1_EPS = 0.005
    RECALL_EPS = 0.005
    best_f1 = df["f1"].max()
    near = df[df["f1"] >= best_f1 - F1_EPS]
    best_recall = near["recall"].max()
    near = near[near["recall"] >= best_recall - RECALL_EPS]
    chosen = near.sort_values("false_positives", ascending=True).iloc[0]
    return float(chosen.confidence_threshold)


def main() -> int:
    weights = paths.FINAL_MODEL_PATH
    if not weights.exists():
        log.error("models/final/best.pt missing - run scripts/benchmark.py first")
        return 1
    data_yaml = paths.PROCESSED_DATA_YAML
    out = paths.EVALUATION_OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)

    log.info("=== 1. Threshold analysis (validation split) ===")
    thr_df = threshold_analysis(weights, data_yaml, out)
    conf = choose_threshold(thr_df)
    log.info("Selected confidence threshold: %.2f", conf)

    log.info("=== 2. One-time TEST evaluation ===")
    test_metrics = evaluate_split(weights, data_yaml, "test", out)

    log.info("=== 3. Inference speed (wall clock) ===")
    sample_dir = paths.PROCESSED_DATASET_DIR / "test" / "images"
    samples = sorted(sample_dir.glob("*.jpg"))[:40]
    speed = {}
    device = pick_device()
    if device.startswith("cuda"):
        speed["gpu"] = measure_inference_speed(weights, samples, device)
    speed["cpu"] = measure_inference_speed(weights, samples, "cpu", runs=15)
    with (out / "inference_speed.json").open("w", encoding="utf-8") as fh:
        json.dump(speed, fh, indent=2)
    for dev, s in speed.items():
        log.info("%s: %.1f ms/img (%.1f FPS)", dev, s["mean_ms"], s["fps"])

    log.info("=== 4. Persist chosen threshold ===")
    meta_path = paths.FINAL_MODEL_METADATA_PATH
    meta = yaml.safe_load(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    meta["confidence_threshold"] = conf
    meta["test_metrics"] = {
        "precision": test_metrics["precision"], "recall": test_metrics["recall"],
        "f1": test_metrics["f1"], "map50": test_metrics["map50"],
        "map50_95": test_metrics["map50_95"], "per_class": test_metrics["per_class"],
    }
    meta["measured_speed"] = speed
    meta_path.write_text(yaml.safe_dump(meta, sort_keys=False), encoding="utf-8")

    app_cfg_path = paths.CONFIG_DIR / "app_config.yaml"
    app_cfg = yaml.safe_load(app_cfg_path.read_text(encoding="utf-8"))
    app_cfg["defaults"]["confidence_threshold"] = conf
    app_cfg_path.write_text(yaml.safe_dump(app_cfg, sort_keys=False, allow_unicode=True),
                            encoding="utf-8")
    log.info("Updated app default threshold to %.2f", conf)
    log.info("Final evaluation complete -> %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
