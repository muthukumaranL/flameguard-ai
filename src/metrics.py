"""Metric helpers shared by evaluation, benchmarking and reporting."""
from __future__ import annotations

import numpy as np


def f1_score(precision: float, recall: float) -> float:
    """Harmonic mean of precision and recall (0 when both are 0)."""
    return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0


def confusion_counts(matrix: np.ndarray) -> dict[str, int]:
    """Summarise an Ultralytics detection confusion matrix.

    The matrix is (nc+1, nc+1) with rows = predicted, columns = ground truth,
    and the final index representing background.
    """
    nc = matrix.shape[0] - 1
    tp = int(np.trace(matrix[:nc, :nc]))
    misclassified = int(matrix[:nc, :nc].sum() - tp)
    false_positives = int(matrix[:nc, nc].sum())   # predicted object, true background
    false_negatives = int(matrix[nc, :nc].sum())   # true object, predicted nothing
    return {
        "true_positives": tp,
        "cross_class_confusions": misclassified,
        "false_positives_background": false_positives,
        "false_negatives_missed": false_negatives,
    }


METRIC_EXPLANATIONS: dict[str, str] = {
    "IoU": (
        "Intersection over Union - the overlap area between a predicted box and a "
        "ground-truth box divided by the area of their union. A prediction counts "
        "as correct when its IoU with a ground-truth object of the same class "
        "exceeds a threshold (0.5 for mAP@0.5)."
    ),
    "Precision": (
        "Of all boxes the model predicted, the fraction that were correct "
        "(TP / (TP + FP)). High precision means few false alarms."
    ),
    "Recall": (
        "Of all real objects, the fraction the model found (TP / (TP + FN)). "
        "High recall means few missed fires or smoke plumes - the safety-critical "
        "direction for this project."
    ),
    "F1-score": (
        "Harmonic mean of precision and recall; balances false alarms against "
        "missed detections at a single confidence threshold."
    ),
    "Average Precision (AP)": (
        "Area under the precision-recall curve for one class, sweeping the "
        "confidence threshold. Summarises the trade-off in a single number."
    ),
    "mAP@0.5": (
        "Mean AP across classes with a match counted at IoU >= 0.5. Rewards "
        "finding objects with reasonably placed boxes."
    ),
    "mAP@0.5:0.95": (
        "Mean AP averaged over ten IoU thresholds from 0.5 to 0.95. Stricter - "
        "rewards precise localisation as well as detection."
    ),
}
