"""Audit object-detection recall by ground-truth object size.

Small fire/smoke regions are often harder to detect than large ones. This
script groups labeled objects by normalized bounding-box area and reports
recall for each slice from a simple CSV export.

Expected CSV columns: matched, box_area_ratio
"""
from __future__ import annotations

import argparse
import csv

BINS = (("small", 0.0, 0.02), ("medium", 0.02, 0.15), ("large", 0.15, 1.01))


def audit(rows):
    result = {}
    for name, low, high in BINS:
        group = [r for r in rows if low <= float(r["box_area_ratio"]) < high]
        matched = sum(str(r["matched"]).lower() in {"1", "true", "yes"} for r in group)
        result[name] = {
            "objects": len(group),
            "matched": matched,
            "recall": matched / len(group) if group else None,
        }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path")
    args = parser.parse_args()
    with open(args.csv_path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for size, metrics in audit(rows).items():
        recall = "n/a" if metrics["recall"] is None else f"{metrics['recall']:.3f}"
        print(f"{size:>6}: objects={metrics['objects']:4d} recall={recall}")


if __name__ == "__main__":
    main()
