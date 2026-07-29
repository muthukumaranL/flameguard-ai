"""Benchmark all trained models on the SAME validation split and select the final model.

Selection is recall-weighted because the project is safety-oriented: missing a
real fire (false negative) is worse than an extra false alarm.

    score = 0.35*mAP50-95 + 0.25*recall + 0.25*smoke_recall + 0.15*speed_score

The winner's best.pt is copied to models/final/best.pt with a metadata file.
Test-set numbers are NOT produced here - the test set is evaluated exactly once
later by scripts/evaluate_final.py.

Usage:
    python scripts/benchmark.py
"""
from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import yaml

from src import paths
from src.evaluate import evaluate_split
from src.utils import file_size_mb, setup_logging

log = setup_logging("flameguard.benchmark")

CANDIDATES = {
    "e1_baseline_v8n": "YOLOv8n (baseline, 40ep)",
    "e2_stronger_v8s": "YOLOv8s (batch2, 18ep)",
    "e3_compare_11n": "YOLO11n (comparison, 12ep)",
    "e5_final": "YOLOv8n tuned (continuation)",
    "e6_final_11n": "YOLO11n tuned (80ep, cls=1.0)",
    # e7_final_11n_recall was planned but not completed (training stopped to meet
    # the deadline); it is intentionally excluded so no partial model is benchmarked.
}
DATA_YAML = paths.PROCESSED_DATA_YAML


def _speed_score(latency_ms: float, latencies: list[float]) -> float:
    """1.0 for the fastest candidate, scaled down linearly to 0 for slowest."""
    lo, hi = min(latencies), max(latencies)
    if hi == lo:
        return 1.0
    return (hi - latency_ms) / (hi - lo)


def main() -> int:
    rows = []
    details = {}
    for exp_id, label in CANDIDATES.items():
        weights = paths.TRAINING_OUTPUT_DIR / exp_id / "weights" / "best.pt"
        if not weights.exists():
            log.warning("skipping %s (no weights at %s)", exp_id, weights)
            continue
        log.info("Benchmarking %s ...", exp_id)
        metrics = evaluate_split(weights, DATA_YAML, "val",
                                 paths.BENCHMARK_OUTPUT_DIR / exp_id)
        details[exp_id] = metrics
        smoke = metrics["per_class"].get("Smoke", {})
        fire = metrics["per_class"].get("Fire", {})
        rows.append({
            "experiment": exp_id,
            "model": label,
            "weights": str(weights.relative_to(paths.PROJECT_ROOT)),
            "model_size_mb": metrics["model_size_mb"],
            "input_size": 640,
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "map50": metrics["map50"],
            "map50_95": metrics["map50_95"],
            "fire_recall": fire.get("recall"),
            "smoke_recall": smoke.get("recall"),
            "fire_ap50_95": fire.get("ap50_95"),
            "smoke_ap50_95": smoke.get("ap50_95"),
            "latency_ms": metrics["total_ms_per_image"],
            "fps": metrics["fps_estimate"],
        })

    if not rows:
        log.error("No trained candidates found - train models first")
        return 1

    df = pd.DataFrame(rows)
    latencies = df["latency_ms"].tolist()
    df["speed_score"] = df["latency_ms"].map(lambda v: _speed_score(v, latencies))
    df["selection_score"] = (0.35 * df["map50_95"] + 0.25 * df["recall"]
                             + 0.25 * df["smoke_recall"].fillna(0.0)
                             + 0.15 * df["speed_score"])
    df = df.sort_values("selection_score", ascending=False).reset_index(drop=True)
    paths.BENCHMARK_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.round(4).to_csv(paths.BENCHMARK_OUTPUT_DIR / "benchmark_table.csv", index=False)
    _chart(df)

    winner = df.iloc[0]
    log.info("Selected final model: %s (score=%.4f)", winner.experiment,
             winner.selection_score)

    # Preserve the finalised threshold / test metrics when the winner has NOT
    # changed: benchmarking re-selecting the same model must not wipe the values
    # that scripts/evaluate_final.py wrote (and must not force a needless
    # re-evaluation of the held-out test set, which is evaluated exactly once).
    prior = {}
    if paths.FINAL_MODEL_METADATA_PATH.exists():
        prior = yaml.safe_load(
            paths.FINAL_MODEL_METADATA_PATH.read_text(encoding="utf-8")) or {}
    winner_unchanged = prior.get("experiment_id") == winner.experiment

    src_weights = paths.PROJECT_ROOT / winner.weights
    paths.FINAL_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_weights, paths.FINAL_MODEL_PATH)

    with (paths.PROCESSED_DATASET_DIR / "data.yaml").open(encoding="utf-8") as fh:
        names = yaml.safe_load(fh)["names"]
    metadata = {
        "model_name": winner.model,
        "experiment_id": winner.experiment,
        "selected_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "fire and smoke v1 (Roboflow, CC BY 4.0) - leakage-repaired resplit",
        "dataset_yaml": "data/processed/fire_smoke_resplit/data.yaml",
        "class_names": names,
        "selection_rule": "0.35*mAP50-95 + 0.25*recall + 0.25*smoke_recall + 0.15*speed",
        "validation_metrics": {
            "precision": float(winner.precision), "recall": float(winner.recall),
            "f1": float(winner.f1), "map50": float(winner.map50),
            "map50_95": float(winner.map50_95),
            "fire_recall": float(winner.fire_recall),
            "smoke_recall": float(winner.smoke_recall),
            "latency_ms": float(winner.latency_ms),
        },
        "model_size_mb": float(winner.model_size_mb),
        # thresholds/test metrics are finalised by scripts/evaluate_final.py; carry
        # them forward when the winner is unchanged, reset to None when it changed
        # (a new final model must be re-evaluated on the test set).
        "confidence_threshold": prior.get("confidence_threshold") if winner_unchanged else None,
        "iou_threshold": 0.50,
    }
    if winner_unchanged and "test_metrics" in prior:
        metadata["test_metrics"] = prior["test_metrics"]
    if winner_unchanged and "measured_speed" in prior:
        metadata["measured_speed"] = prior["measured_speed"]
    log.info("Winner %s vs prior %s -> %s",
             winner.experiment, prior.get("experiment_id"),
             "preserved threshold/test metrics" if winner_unchanged
             else "reset for re-evaluation")
    with paths.FINAL_MODEL_METADATA_PATH.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(metadata, fh, sort_keys=False)
    with (paths.BENCHMARK_OUTPUT_DIR / "selection_report.json").open("w", encoding="utf-8") as fh:
        json.dump({"winner": winner.experiment,
                   "ranking": df[["experiment", "selection_score"]].to_dict("records"),
                   "criteria": metadata["selection_rule"]}, fh, indent=2)
    log.info("Final model copied to %s", paths.FINAL_MODEL_PATH)
    return 0


def _chart(df: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    x = df["model"]
    axes[0].bar(x, df["map50_95"], color="#4c72b0")
    axes[0].set_title("mAP@0.5:0.95 (validation)")
    axes[0].tick_params(axis="x", rotation=20)
    axes[1].bar(x, df["recall"], color="#c1121f", label="overall")
    axes[1].bar(x, df["smoke_recall"], color="#5a9fc8", alpha=0.7, width=0.5,
                label="smoke")
    axes[1].set_title("Recall (validation)")
    axes[1].legend(); axes[1].tick_params(axis="x", rotation=20)
    axes[2].scatter(df["latency_ms"], df["map50_95"], s=80, color="#2a9d8f")
    for _, r in df.iterrows():
        axes[2].annotate(r["model"], (r["latency_ms"], r["map50_95"]),
                         fontsize=8, xytext=(4, 4), textcoords="offset points")
    axes[2].set_xlabel("latency ms/image"); axes[2].set_title("Accuracy vs speed")
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.savefig(paths.BENCHMARK_OUTPUT_DIR / "benchmark_chart.png",
                dpi=130, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
