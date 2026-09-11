"""Sweep detection confidence thresholds and report precision/recall trade-offs.

Expected CSV columns: confidence, matched. Each row represents one predicted
box after class-aware IoU matching. A matched value of 1/true/yes is a true
positive; unmatched predictions are false positives. Optional ground_truth_count
sets the total number of labeled objects so recall can include missed objects.
"""
from __future__ import annotations

import argparse
import csv


def _truthy(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def sweep(rows, ground_truth_count: int | None = None, thresholds=None):
    thresholds = thresholds or [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    if ground_truth_count is not None and ground_truth_count < 1:
        raise ValueError("ground_truth_count must be positive")
    result = []
    for threshold in thresholds:
        if not 0 <= threshold <= 1:
            raise ValueError("thresholds must be in [0, 1]")
        kept = [row for row in rows if float(row["confidence"]) >= threshold]
        tp = sum(_truthy(row["matched"]) for row in kept)
        fp = len(kept) - tp
        denominator = ground_truth_count if ground_truth_count is not None else max(tp, 1)
        precision = tp / len(kept) if kept else 0.0
        recall = tp / denominator
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        result.append(
            {
                "threshold": float(threshold),
                "predictions": len(kept),
                "tp": tp,
                "fp": fp,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path")
    parser.add_argument("--ground-truth-count", type=int)
    args = parser.parse_args()
    with open(args.csv_path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in sweep(rows, args.ground_truth_count):
        print(
            f"threshold={row['threshold']:.2f} precision={row['precision']:.3f} "
            f"recall={row['recall']:.3f} f1={row['f1']:.3f} predictions={row['predictions']}"
        )


if __name__ == "__main__":
    main()
