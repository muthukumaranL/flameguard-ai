"""Dataset integrity validation for the FlameGuard AI fire/smoke dataset.

Checks performed per split (train/valid/test):
  * image <-> label file matching (both directions)
  * corrupt / unreadable images (PIL verify + reopen)
  * unsupported image formats
  * empty label files (valid negatives - counted, not flagged)
  * invalid class ids, malformed lines, out-of-range coordinates
  * degenerate (zero/near-zero size) boxes, boxes outside bounds
  * extremely small / extremely large boxes (reported for EDA review)
  * duplicate filenames across splits
  * exact duplicate images (MD5) within and across splits
  * per-class image and annotation counts (>=200 originals requirement)

Results are written to outputs/dataset_validation/ as JSON + CSV + Markdown.
"""
from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from PIL import Image

from src.annotation_parser import label_path_for_image, parse_label_file
from src.utils import setup_logging

log = setup_logging("flameguard.validate")

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
TINY_BOX_AREA = 0.0004   # < 0.04% of image area (~13x13px at 640) - review flag
HUGE_BOX_AREA = 0.90     # > 90% of image area - review flag


@dataclass
class SplitStats:
    """Aggregated statistics for one dataset split."""

    images: int = 0
    label_files: int = 0
    images_without_labels: list[str] = field(default_factory=list)
    labels_without_images: list[str] = field(default_factory=list)
    corrupt_images: list[str] = field(default_factory=list)
    unsupported_files: list[str] = field(default_factory=list)
    empty_label_files: int = 0
    label_issues: list[dict[str, str]] = field(default_factory=list)
    tiny_boxes: int = 0
    huge_boxes: int = 0
    boxes_per_class: dict[int, int] = field(default_factory=lambda: defaultdict(int))
    images_per_class: dict[int, int] = field(default_factory=lambda: defaultdict(int))
    images_with_both: int = 0
    background_images: int = 0


def _md5(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with path.open("rb") as fh:
        while blob := fh.read(chunk):
            h.update(blob)
    return h.hexdigest()


def _check_image(path: Path) -> str | None:
    """Return an issue string when the image is corrupt/unreadable, else None."""
    try:
        with Image.open(path) as im:
            im.verify()
        # verify() leaves the file unusable; reopen to force a full decode
        with Image.open(path) as im:
            im.load()
        return None
    except Exception as exc:  # any decoder failure means a corrupt image
        return f"{type(exc).__name__}: {exc}"


def validate_dataset(dataset_dir: Path, output_dir: Path) -> dict[str, Any]:
    """Run the full audit over a YOLO dataset directory; write reports.

    Returns the report dict (also saved as validation_report.json).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    data_yaml_path = dataset_dir / "data.yaml"
    with data_yaml_path.open("r", encoding="utf-8") as fh:
        data_yaml = yaml.safe_load(fh)
    class_names = data_yaml["names"]
    if isinstance(class_names, dict):
        class_names = [class_names[k] for k in sorted(class_names)]
    valid_ids = set(range(len(class_names)))
    log.info("Validating %s | classes=%s", dataset_dir.name, class_names)

    split_stats: dict[str, SplitStats] = {}
    hash_index: dict[str, list[str]] = defaultdict(list)   # md5 -> ["split/name", ...]
    filename_index: dict[str, list[str]] = defaultdict(list)

    for split in ("train", "valid", "test"):
        stats = SplitStats()
        img_dir = dataset_dir / split / "images"
        lbl_dir = dataset_dir / split / "labels"
        images = sorted(p for p in img_dir.iterdir() if p.is_file()) if img_dir.exists() else []
        labels = {p.stem for p in lbl_dir.iterdir() if p.suffix == ".txt"} if lbl_dir.exists() else set()
        stats.images = len(images)
        stats.label_files = len(labels)

        for img in images:
            filename_index[img.name].append(f"{split}")
            if img.suffix.lower() not in SUPPORTED_EXTS:
                stats.unsupported_files.append(img.name)
                continue
            if issue := _check_image(img):
                stats.corrupt_images.append(f"{img.name}: {issue}")
                continue
            hash_index[_md5(img)].append(f"{split}/{img.name}")

            lbl = label_path_for_image(img)
            if not lbl.exists():
                stats.images_without_labels.append(img.name)
                continue
            parsed = parse_label_file(lbl, valid_ids)
            for iss in parsed.issues:
                stats.label_issues.append({"file": lbl.name, "issue": iss})
            if not parsed.boxes and not parsed.issues:
                stats.empty_label_files += 1
            classes_here = set()
            for box in parsed.boxes:
                stats.boxes_per_class[box.class_id] += 1
                classes_here.add(box.class_id)
                if box.area < TINY_BOX_AREA:
                    stats.tiny_boxes += 1
                elif box.area > HUGE_BOX_AREA:
                    stats.huge_boxes += 1
            for cid in classes_here:
                stats.images_per_class[cid] += 1
            if classes_here == valid_ids:
                stats.images_with_both += 1
            if not classes_here:
                stats.background_images += 1

        img_stems = {p.stem for p in images}
        stats.labels_without_images = sorted(labels - img_stems)
        split_stats[split] = stats
        log.info("%s: %d images, %d labels, %d corrupt, %d label issues",
                 split, stats.images, stats.label_files,
                 len(stats.corrupt_images), len(stats.label_issues))

    exact_dupes = {h: locs for h, locs in hash_index.items() if len(locs) > 1}
    cross_split_dupes = {
        h: locs for h, locs in exact_dupes.items()
        if len({loc.split("/", 1)[0] for loc in locs}) > 1
    }
    dupe_filenames = {n: s for n, s in filename_index.items() if len(s) > 1}

    per_class_totals = {
        name: sum(split_stats[s].images_per_class.get(cid, 0) for s in split_stats)
        for cid, name in enumerate(class_names)
    }
    min_images_ok = all(v >= 200 for v in per_class_totals.values())

    report: dict[str, Any] = {
        "dataset": str(dataset_dir),
        "classes": class_names,
        "splits": {
            s: {
                "images": st.images,
                "label_files": st.label_files,
                "images_without_labels": st.images_without_labels,
                "labels_without_images": st.labels_without_images,
                "corrupt_images": st.corrupt_images,
                "unsupported_files": st.unsupported_files,
                "empty_label_files_background": st.empty_label_files,
                "background_images": st.background_images,
                "label_issue_count": len(st.label_issues),
                "tiny_boxes_flagged": st.tiny_boxes,
                "huge_boxes_flagged": st.huge_boxes,
                "boxes_per_class": {class_names[c]: n for c, n in sorted(st.boxes_per_class.items())},
                "images_per_class": {class_names[c]: n for c, n in sorted(st.images_per_class.items())},
                "images_with_both_classes": st.images_with_both,
            }
            for s, st in split_stats.items()
        },
        "exact_duplicate_groups": len(exact_dupes),
        "exact_duplicates_cross_split": len(cross_split_dupes),
        "duplicate_filenames_across_splits": len(dupe_filenames),
        "images_per_class_total": per_class_totals,
        "min_200_images_per_class": min_images_ok,
    }

    with (output_dir / "validation_report.json").open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    with (output_dir / "label_issues.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["split", "file", "issue"])
        for s, st in split_stats.items():
            for row in st.label_issues:
                writer.writerow([s, row["file"], row["issue"]])

    with (output_dir / "duplicate_report.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["md5", "locations", "cross_split"])
        for h, locs in sorted(exact_dupes.items()):
            writer.writerow([h, ";".join(locs), h in cross_split_dupes])

    _write_summary_md(output_dir / "validation_summary.md", report)
    log.info("Validation reports written to %s", output_dir)
    return report


def _write_summary_md(path: Path, report: dict[str, Any]) -> None:
    lines = ["# Dataset Validation Summary", ""]
    lines.append(f"Dataset: `{Path(report['dataset']).name}` | Classes: {report['classes']}")
    lines.append("")
    lines.append("| Split | Images | Labels | Corrupt | Label issues | Background | Both classes |")
    lines.append("|---|---|---|---|---|---|---|")
    for s, st in report["splits"].items():
        lines.append(
            f"| {s} | {st['images']} | {st['label_files']} | {len(st['corrupt_images'])} "
            f"| {st['label_issue_count']} | {st['background_images']} | {st['images_with_both_classes']} |"
        )
    lines.append("")
    lines.append(f"- Exact duplicate groups: **{report['exact_duplicate_groups']}** "
                 f"(cross-split: **{report['exact_duplicates_cross_split']}**)")
    lines.append(f"- Images per class (total): {report['images_per_class_total']}")
    lines.append(f"- >=200 original images per class: **{'PASS' if report['min_200_images_per_class'] else 'FAIL'}**")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
