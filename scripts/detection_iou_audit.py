"""Class-aware object-detection IoU audit.

CSV schema for ground truth and predictions:
    image_id,class_name,x1,y1,x2,y2

Predictions may also include a confidence column. The evaluator greedily
matches predictions to unmatched ground-truth boxes of the same class and
reports precision, recall and F1 at a configurable IoU threshold.

Example:
    python scripts/detection_iou_audit.py --ground-truth gt.csv --predictions pred.csv
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Box:
    image_id: str
    class_name: str
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float = 1.0


def iou(a: Box, b: Box) -> float:
    left = max(a.x1, b.x1)
    top = max(a.y1, b.y1)
    right = min(a.x2, b.x2)
    bottom = min(a.y2, b.y2)
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    area_a = max(0.0, a.x2 - a.x1) * max(0.0, a.y2 - a.y1)
    area_b = max(0.0, b.x2 - b.x1) * max(0.0, b.y2 - b.y1)
    union = area_a + area_b - intersection
    return 0.0 if union <= 0.0 else intersection / union


def load_boxes(path: Path) -> list[Box]:
    boxes: list[Box] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            boxes.append(
                Box(
                    image_id=row["image_id"],
                    class_name=row["class_name"],
                    x1=float(row["x1"]),
                    y1=float(row["y1"]),
                    x2=float(row["x2"]),
                    y2=float(row["y2"]),
                    confidence=float(row.get("confidence") or 1.0),
                )
            )
    return boxes


def evaluate(ground_truth: list[Box], predictions: list[Box], threshold: float) -> dict[str, float | int]:
    matched_gt: set[int] = set()
    true_positive = 0

    for pred in sorted(predictions, key=lambda box: box.confidence, reverse=True):
        candidates = [
            (index, iou(pred, gt))
            for index, gt in enumerate(ground_truth)
            if index not in matched_gt
            and gt.image_id == pred.image_id
            and gt.class_name == pred.class_name
        ]
        if not candidates:
            continue
        best_index, best_iou = max(candidates, key=lambda item: item[1])
        if best_iou >= threshold:
            matched_gt.add(best_index)
            true_positive += 1

    false_positive = len(predictions) - true_positive
    false_negative = len(ground_truth) - true_positive
    precision = true_positive / (true_positive + false_positive) if predictions else 0.0
    recall = true_positive / (true_positive + false_negative) if ground_truth else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": true_positive,
        "fp": false_positive,
        "fn": false_negative,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--iou", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 < args.iou <= 1.0:
        raise SystemExit("--iou must be in (0, 1]")
    metrics = evaluate(load_boxes(args.ground_truth), load_boxes(args.predictions), args.iou)
    print(f"IoU threshold: {args.iou:.2f}")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")


if __name__ == "__main__":
    main()
