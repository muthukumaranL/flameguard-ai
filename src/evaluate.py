"""Model evaluation: split metrics, threshold analysis, speed measurement.

All numbers are produced by running the model - never hand-entered.
"""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.metrics import confusion_counts, f1_score
from src.paths import PROJECT_ROOT
from src.utils import file_size_mb, pick_device, setup_logging

log = setup_logging("flameguard.evaluate")

# Plots Ultralytics writes into its val run directory that we keep for the report
VAL_PLOTS = (
    "confusion_matrix.png", "confusion_matrix_normalized.png",
    "BoxPR_curve.png", "BoxF1_curve.png", "BoxP_curve.png", "BoxR_curve.png",
)


def count_split_images(data_yaml: Path, split: str) -> int:
    """Number of images in a split, resolved through data.yaml.

    Ultralytics' metrics object does not expose how many images it saw, and the
    folder name does not always match the split name ('val' -> 'valid/images'),
    so the count is resolved from the yaml rather than guessed.
    """
    import yaml as _yaml

    cfg = _yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    rel = cfg.get(split) or cfg.get({"val": "valid", "valid": "val"}.get(split, split))
    if not rel:
        return 0
    img_dir = (data_yaml.parent / rel).resolve()
    return sum(1 for p in img_dir.iterdir() if p.is_file()) if img_dir.exists() else 0


def evaluate_split(weights: Path, data_yaml: Path, split: str,
                   out_dir: Path, device: str | None = None) -> dict[str, Any]:
    """Run Ultralytics validation on one split and export a metric package."""
    from ultralytics import YOLO

    from src.train import ultralytics_data_yaml

    out_dir.mkdir(parents=True, exist_ok=True)
    device = device or pick_device()
    model = YOLO(str(weights))
    results = model.val(data=str(ultralytics_data_yaml(data_yaml)), split=split, device=device,
                        plots=True, verbose=False,
                        project=str(out_dir), name=f"val_{split}", exist_ok=True)

    names = [results.names[k] for k in sorted(results.names)]
    per_class: dict[str, dict[str, float]] = {}
    for idx, cls_idx in enumerate(results.box.ap_class_index):
        p, r, ap50, ap = results.box.class_result(idx)
        per_class[names[int(cls_idx)]] = {
            "precision": float(p), "recall": float(r), "f1": f1_score(float(p), float(r)),
            "ap50": float(ap50), "ap50_95": float(ap),
        }

    speed_ms = {k: float(v) for k, v in results.speed.items()}
    total_ms = sum(speed_ms.values())
    metrics: dict[str, Any] = {
        "weights": str(weights),
        "split": split,
        "device": device,
        "images": count_split_images(data_yaml, split),
        "precision": float(results.box.mp),
        "recall": float(results.box.mr),
        "f1": f1_score(float(results.box.mp), float(results.box.mr)),
        "map50": float(results.box.map50),
        "map50_95": float(results.box.map),
        "per_class": per_class,
        "speed_ms_per_image": speed_ms,
        "total_ms_per_image": total_ms,
        "fps_estimate": 1000.0 / total_ms if total_ms > 0 else None,
        "model_size_mb": round(file_size_mb(weights), 2),
        "confusion_counts": confusion_counts(results.confusion_matrix.matrix),
    }
    with (out_dir / f"metrics_{split}.json").open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)

    run_dir = Path(results.save_dir)
    for plot in VAL_PLOTS:
        src = run_dir / plot
        if src.exists():
            shutil.copy2(src, out_dir / f"{split}_{plot}")
    log.info("%s split: P=%.4f R=%.4f mAP50=%.4f mAP50-95=%.4f",
             split, metrics["precision"], metrics["recall"],
             metrics["map50"], metrics["map50_95"])
    return metrics


def threshold_analysis(weights: Path, data_yaml: Path, out_dir: Path,
                       thresholds: tuple[float, ...] = (0.10, 0.15, 0.20, 0.30,
                                                        0.40, 0.50, 0.60),
                       split: str = "val") -> pd.DataFrame:
    """Measure P/R/F1 and FP/FN on the validation split at candidate thresholds."""
    from ultralytics import YOLO

    from src.train import ultralytics_data_yaml

    out_dir.mkdir(parents=True, exist_ok=True)
    resolved_yaml = ultralytics_data_yaml(data_yaml)
    device = pick_device()
    rows = []
    for conf in thresholds:
        model = YOLO(str(weights))
        # plots=True is REQUIRED: Ultralytics only populates
        # results.confusion_matrix.matrix when it also renders the plot. With
        # plots=False the matrix stays all-zeros and FP/FN/TP would read as 0.
        results = model.val(data=str(resolved_yaml), split=split, device=device,
                            conf=conf, plots=True, verbose=False,
                            project=str(out_dir), name=f"thr_{conf:.2f}", exist_ok=True)
        counts = confusion_counts(results.confusion_matrix.matrix)
        p, r = float(results.box.mp), float(results.box.mr)
        rows.append({
            "confidence_threshold": conf,
            "precision": p, "recall": r, "f1": f1_score(p, r),
            "false_positives": counts["false_positives_background"],
            "false_negatives": counts["false_negatives_missed"],
            "true_positives": counts["true_positives"],
        })
        log.info("conf=%.2f -> P=%.4f R=%.4f F1=%.4f FP=%d FN=%d",
                 conf, p, r, rows[-1]["f1"], rows[-1]["false_positives"],
                 rows[-1]["false_negatives"])
        # clean the per-threshold val run dir; we only keep the table
        shutil.rmtree(out_dir / f"thr_{conf:.2f}", ignore_errors=True)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "threshold_analysis.csv", index=False)
    _plot_threshold_analysis(df, out_dir / "threshold_analysis.png")
    return df


def _plot_threshold_analysis(df: pd.DataFrame, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    x = df.confidence_threshold
    axes[0].plot(x, df.precision, "o-", label="precision", color="#4c72b0")
    axes[0].plot(x, df.recall, "s-", label="recall", color="#c1121f")
    axes[0].plot(x, df.f1, "^-", label="F1", color="#2a9d8f")
    axes[0].set_xlabel("confidence threshold"); axes[0].set_title("P / R / F1 vs threshold")
    axes[0].legend()
    axes[1].plot(x, df.false_positives, "o-", label="false positives", color="#fb8500")
    axes[1].plot(x, df.false_negatives, "s-", label="false negatives (missed)", color="#6a040f")
    axes[1].set_xlabel("confidence threshold"); axes[1].set_title("Error counts vs threshold")
    axes[1].legend()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def measure_inference_speed(weights: Path, sample_images: list[Path],
                            device: str, warmup: int = 5, runs: int = 30) -> dict[str, float]:
    """Wall-clock single-image predict() latency on a specific device."""
    from ultralytics import YOLO

    model = YOLO(str(weights))
    imgs = [str(p) for p in sample_images]
    for i in range(warmup):
        model.predict(imgs[i % len(imgs)], device=device, verbose=False)
    times = []
    for i in range(runs):
        start = time.perf_counter()
        model.predict(imgs[i % len(imgs)], device=device, verbose=False)
        times.append(time.perf_counter() - start)
    arr = np.array(times) * 1000
    return {
        "device": device,
        "mean_ms": float(arr.mean()),
        "median_ms": float(np.median(arr)),
        "p95_ms": float(np.percentile(arr, 95)),
        "fps": float(1000.0 / arr.mean()),
        "runs": runs,
    }


def resolve_data_yaml(rel: str = "data/processed/fire_smoke_resplit/data.yaml") -> Path:
    return (PROJECT_ROOT / rel).resolve()
