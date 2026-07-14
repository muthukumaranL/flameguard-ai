"""Training driver for all FlameGuard AI experiments.

Every experiment is transfer learning: COCO-pretrained Ultralytics weights are
fine-tuned on the leakage-repaired fire/smoke dataset.  Experiments are defined
declaratively in config/training_config.yaml and executed with fixed seeds.

Each run appends one row to outputs/training/experiment_log.csv so the report
and benchmark tables are generated from measured values only.
"""
from __future__ import annotations

import csv
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from src.config import load_training_config
from src.paths import PROJECT_ROOT, TRAINING_OUTPUT_DIR
from src.utils import file_size_mb, human_duration, pick_device, set_seeds, setup_logging

log = setup_logging("flameguard.train")

EXPERIMENT_LOG = TRAINING_OUTPUT_DIR / "experiment_log.csv"
LOG_COLUMNS = [
    "experiment_id", "date_utc", "model", "starting_weights", "imgsz", "epochs_requested",
    "epochs_run", "batch", "optimizer", "lr0", "weight_decay", "augmentation_notes",
    "seed", "hardware", "duration", "best_epoch", "precision", "recall", "f1",
    "map50", "map50_95", "model_size_mb", "notes",
]

# Fire/smoke rise upward - vertical flips create unrealistic scenes, so they
# are disabled for every experiment (documented in the report).
FIXED_AUG = {"flipud": 0.0}

# Keys in an experiment block that are passed straight to Ultralytics train()
PASSTHROUGH_KEYS = {
    "optimizer", "lr0", "lrf", "momentum", "weight_decay", "warmup_epochs",
    "hsv_h", "hsv_s", "hsv_v", "degrees", "translate", "scale", "shear",
    "mosaic", "mixup", "close_mosaic", "fliplr", "box", "cls", "dfl", "cos_lr",
}


def ultralytics_data_yaml(data_yaml: Path) -> Path:
    """Return a runtime copy of data.yaml with an absolute ``path`` entry.

    The committed data.yaml keeps ``path: .`` so the repository stays
    machine-independent, but Ultralytics resolves relative paths against its
    global datasets directory rather than the yaml location.  This writes a
    resolved copy (a runtime artefact, not committed) and returns its path.
    """
    cfg = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    cfg["path"] = str(data_yaml.parent.resolve())
    resolved = TRAINING_OUTPUT_DIR / "_resolved_data.yaml"
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return resolved


def _fitness(row: pd.Series) -> float:
    """Ultralytics default fitness: 0.1*mAP50 + 0.9*mAP50-95."""
    return 0.1 * row.get("metrics/mAP50(B)", 0.0) + 0.9 * row.get("metrics/mAP50-95(B)", 0.0)


def _best_epoch_stats(results_csv: Path) -> dict[str, float]:
    df = pd.read_csv(results_csv)
    df.columns = [c.strip() for c in df.columns]
    fit = df.apply(_fitness, axis=1)
    best = df.iloc[int(fit.idxmax())]
    p = float(best.get("metrics/precision(B)", float("nan")))
    r = float(best.get("metrics/recall(B)", float("nan")))
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return {
        "epochs_run": int(df["epoch"].max()) + (0 if df["epoch"].min() == 1 else 1),
        "best_epoch": int(best["epoch"]),
        "precision": p,
        "recall": r,
        "f1": f1,
        "map50": float(best.get("metrics/mAP50(B)", float("nan"))),
        "map50_95": float(best.get("metrics/mAP50-95(B)", float("nan"))),
    }


def _append_log_row(row: dict[str, Any]) -> None:
    EXPERIMENT_LOG.parent.mkdir(parents=True, exist_ok=True)
    exists = EXPERIMENT_LOG.exists()
    with EXPERIMENT_LOG.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=LOG_COLUMNS)
        if not exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in LOG_COLUMNS})


def run_experiment(exp_id: str, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Train one configured experiment; returns its summary row."""
    from ultralytics import YOLO

    cfg = load_training_config()
    if exp_id not in cfg["experiments"]:
        raise KeyError(f"Unknown experiment '{exp_id}'")
    exp = dict(cfg["experiments"][exp_id])
    if overrides:
        exp.update(overrides)

    seed = int(cfg.get("seed", 42))
    set_seeds(seed)
    device = pick_device()
    data_yaml = (PROJECT_ROOT / cfg["dataset_yaml"]).resolve()
    if not data_yaml.exists():
        raise FileNotFoundError(f"dataset yaml missing: {data_yaml}")

    save_dir = (PROJECT_ROOT / exp["save_dir"]).resolve()
    weights = exp["model"]
    train_kwargs: dict[str, Any] = {
        "data": str(ultralytics_data_yaml(data_yaml)),
        "epochs": int(exp["epochs"]),
        "batch": int(exp["batch"]),
        "imgsz": int(cfg.get("imgsz", 640)),
        "patience": int(cfg.get("patience", 10)),
        "workers": int(cfg.get("workers", 2)),
        "seed": seed,
        "device": device,
        "project": str(save_dir.parent),
        "name": save_dir.name,
        "exist_ok": True,
        "plots": True,
        "val": True,
        **FIXED_AUG,
    }
    for key in PASSTHROUGH_KEYS & exp.keys():
        train_kwargs[key] = exp[key]
    if str(train_kwargs.get("optimizer", "auto")).lower() == "auto":
        train_kwargs.pop("optimizer", None)

    log.info("=== %s | model=%s device=%s epochs=%s batch=%s ===",
             exp_id, weights, device, exp["epochs"], exp["batch"])
    started = time.perf_counter()
    model = YOLO(weights)          # downloads COCO-pretrained weights on first use
    results = model.train(**train_kwargs)
    duration = time.perf_counter() - started

    run_dir = Path(results.save_dir)
    stats = _best_epoch_stats(run_dir / "results.csv")
    args_used = yaml.safe_load((run_dir / "args.yaml").read_text(encoding="utf-8"))

    row: dict[str, Any] = {
        "experiment_id": exp_id,
        "date_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
        "model": Path(weights).stem,
        "starting_weights": f"{weights} (COCO-pretrained)",
        "imgsz": train_kwargs["imgsz"],
        "epochs_requested": train_kwargs["epochs"],
        "batch": args_used.get("batch"),
        "optimizer": args_used.get("optimizer"),
        "lr0": args_used.get("lr0"),
        "weight_decay": args_used.get("weight_decay"),
        "augmentation_notes": _augmentation_summary(args_used),
        "seed": seed,
        "hardware": f"{platform.node()} | {device}",
        "duration": human_duration(duration),
        "model_size_mb": round(file_size_mb(run_dir / "weights" / "best.pt"), 2),
        "notes": exp.get("notes", ""),
        **stats,
    }
    _append_log_row(row)
    log.info("%s done in %s | best_epoch=%s mAP50=%.4f mAP50-95=%.4f R=%.4f",
             exp_id, row["duration"], row["best_epoch"], row["map50"],
             row["map50_95"], row["recall"])
    return row


def _augmentation_summary(args: dict[str, Any]) -> str:
    keys = ("fliplr", "flipud", "degrees", "translate", "scale", "mosaic",
            "mixup", "hsv_h", "hsv_s", "hsv_v", "close_mosaic")
    return "; ".join(f"{k}={args.get(k)}" for k in keys if k in args)
