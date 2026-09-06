"""Audit exported object-detection confidence scores by class.

FlameGuard exports flattened detection records with fields such as
``source_file``, ``class_name`` and ``confidence``. This utility reads one of
those CSV exports and summarizes confidence quality by class so weak or
borderline predictions are visible before deployment decisions are made.

The audit is descriptive rather than a substitute for mAP/precision/recall:
confidence tells us how certain the model is, not whether a prediction is
correct. Use it alongside labelled evaluation data.

Run:
    python scripts/audit_detection_confidence.py path/to/detections.csv
    python scripts/audit_detection_confidence.py path/to/detections.csv --low 0.35 --high 0.75
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median


@dataclass(frozen=True)
class ConfidenceSummary:
    class_name: str
    count: int
    mean_confidence: float
    median_confidence: float
    minimum: float
    maximum: float
    low_confidence_count: int
    high_confidence_count: int

    def low_share(self) -> float:
        return self.low_confidence_count / self.count if self.count else 0.0

    def high_share(self) -> float:
        return self.high_confidence_count / self.count if self.count else 0.0


def load_confidences(path: Path) -> dict[str, list[float]]:
    """Load class/confidence pairs from a FlameGuard detection CSV export."""
    grouped: dict[str, list[float]] = defaultdict(list)

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"class_name", "confidence"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"CSV is missing required column(s): {', '.join(sorted(missing))}"
            )

        for row_number, row in enumerate(reader, start=2):
            class_name = (row.get("class_name") or "").strip()
            confidence_text = (row.get("confidence") or "").strip()
            if not class_name or not confidence_text:
                continue

            try:
                confidence = float(confidence_text)
            except ValueError as exc:
                raise ValueError(
                    f"row {row_number} has invalid confidence: {confidence_text!r}"
                ) from exc

            if not 0.0 <= confidence <= 1.0:
                raise ValueError(
                    f"row {row_number} confidence must be between 0 and 1"
                )

            grouped[class_name].append(confidence)

    if not grouped:
        raise ValueError("CSV contains no usable detection rows")

    return dict(grouped)


def summarize(
    grouped: dict[str, list[float]],
    *,
    low_threshold: float,
    high_threshold: float,
) -> list[ConfidenceSummary]:
    """Compute per-class confidence summaries."""
    if not 0.0 <= low_threshold < high_threshold <= 1.0:
        raise ValueError("thresholds must satisfy 0 <= low < high <= 1")

    summaries: list[ConfidenceSummary] = []
    for class_name, values in sorted(grouped.items()):
        summaries.append(
            ConfidenceSummary(
                class_name=class_name,
                count=len(values),
                mean_confidence=mean(values),
                median_confidence=median(values),
                minimum=min(values),
                maximum=max(values),
                low_confidence_count=sum(value < low_threshold for value in values),
                high_confidence_count=sum(value >= high_threshold for value in values),
            )
        )
    return summaries


def print_report(
    summaries: list[ConfidenceSummary],
    *,
    low_threshold: float,
    high_threshold: float,
) -> None:
    total = sum(summary.count for summary in summaries)
    print(f"Detections audited: {total}")
    print(
        f"Bands: low < {low_threshold:.2f}, "
        f"middle {low_threshold:.2f}-{high_threshold:.2f}, "
        f"high >= {high_threshold:.2f}\n"
    )
    print(
        f"{'class':<14} {'n':>6} {'mean':>8} {'median':>8} "
        f"{'min':>8} {'max':>8} {'low%':>8} {'high%':>8}"
    )
    print("-" * 76)

    for summary in summaries:
        print(
            f"{summary.class_name:<14} {summary.count:>6d} "
            f"{summary.mean_confidence:>8.3f} "
            f"{summary.median_confidence:>8.3f} "
            f"{summary.minimum:>8.3f} {summary.maximum:>8.3f} "
            f"{100 * summary.low_share():>7.1f}% "
            f"{100 * summary.high_share():>7.1f}%"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize confidence scores from a FlameGuard detection CSV."
    )
    parser.add_argument("csv_path", type=Path, help="Detection CSV export")
    parser.add_argument(
        "--low", type=float, default=0.35, help="Upper bound for low-confidence detections"
    )
    parser.add_argument(
        "--high", type=float, default=0.75, help="Lower bound for high-confidence detections"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    grouped = load_confidences(args.csv_path)
    summaries = summarize(
        grouped,
        low_threshold=args.low,
        high_threshold=args.high,
    )
    print_report(
        summaries,
        low_threshold=args.low,
        high_threshold=args.high,
    )


if __name__ == "__main__":
    main()
