"""Compare fire/smoke detector operating points across confidence thresholds."""

from __future__ import annotations

import numpy as np


def threshold_operating_points(
    y_true: np.ndarray,
    confidence: np.ndarray,
    thresholds: tuple[float, ...] = (0.3, 0.5, 0.7, 0.85),
) -> list[dict[str, float]]:
    """Return precision, recall, false-positive rate, and F1 at each threshold."""
    y = np.asarray(y_true, dtype=int)
    scores = np.asarray(confidence, dtype=float)
    if y.ndim != 1 or scores.ndim != 1 or y.shape != scores.shape or y.size == 0:
        raise ValueError("y_true and confidence must be equal non-empty 1D arrays")
    if not np.all(np.isin(y, [0, 1])):
        raise ValueError("y_true must contain only 0 and 1")
    if not np.all(np.isfinite(scores)) or np.any((scores < 0) | (scores > 1)):
        raise ValueError("confidence must be finite and in [0, 1]")

    rows: list[dict[str, float]] = []
    for threshold in thresholds:
        if not 0 <= threshold <= 1:
            raise ValueError("thresholds must be in [0, 1]")
        pred = scores >= threshold
        tp = int(np.sum((y == 1) & pred))
        fp = int(np.sum((y == 0) & pred))
        fn = int(np.sum((y == 1) & ~pred))
        tn = int(np.sum((y == 0) & ~pred))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        fpr = fp / (fp + tn) if fp + tn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append({
            "threshold": float(threshold),
            "precision": float(precision),
            "recall": float(recall),
            "false_positive_rate": float(fpr),
            "f1": float(f1),
        })
    return rows


if __name__ == "__main__":
    labels = np.array([0, 0, 1, 1, 0, 1, 0, 1, 0, 1])
    scores = np.array([0.1, 0.45, 0.62, 0.91, 0.55, 0.78, 0.2, 0.48, 0.72, 0.88])
    for row in threshold_operating_points(labels, scores):
        print(row)