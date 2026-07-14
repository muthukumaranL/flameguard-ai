"""Structured error analysis of the final model on the test split.

Every test image is compared against ground truth (IoU-matched, class-aware):
  TP   - correct detection (IoU >= 0.5, same class)
  FP   - predicted box with no matching ground truth
  FN   - ground-truth object with no matching prediction
  LOC  - right class, wrong box (0.1 <= IoU < 0.5)

Outputs (outputs/error_analysis/):
  error_summary.json          counts per category and per class
  per_image_errors.csv        one row per test image
  gallery_<category>/         annotated example images (GT green, pred class colour)
  error_gallery.png           report figure with representative success/failure cases

Usage:
    python scripts/error_analysis.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import paths
from src.annotation_parser import label_path_for_image, parse_label_file
from src.inference import DetectionEngine
from src.utils import setup_logging
from src.visualizations import CLASS_COLORS_RGB, image_grid

log = setup_logging("flameguard.errors")

IOU_TP = 0.50
IOU_LOC = 0.10
CONF = 0.30          # operating threshold for error analysis (report notes this)
GT_COLOR = (60, 180, 75)


def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def analyse_image(engine: DetectionEngine, img_path: Path,
                  class_names: dict[int, str]) -> dict:
    parsed = parse_label_file(label_path_for_image(img_path), set(class_names))
    import cv2

    bgr = cv2.imread(str(img_path))
    result = engine.predict(bgr, conf=CONF, iou=0.50, draw=False)
    h, w = bgr.shape[:2]
    gts = [(b.class_id, b.to_xyxy(w, h)) for b in parsed.boxes]
    preds = [(d.class_id, (d.x1, d.y1, d.x2, d.y2), d.confidence)
             for d in result.detections]

    matched_gt: set[int] = set()
    matched_pred: set[int] = set()
    loc_pairs: list[tuple[int, int]] = []
    # greedy match: best IoU first
    pairs = sorted(
        ((gi, pi, _iou(g[1], p[1]))
         for gi, g in enumerate(gts) for pi, p in enumerate(preds)
         if g[0] == p[0]),
        key=lambda t: t[2], reverse=True)
    for gi, pi, iou in pairs:
        if gi in matched_gt or pi in matched_pred:
            continue
        if iou >= IOU_TP:
            matched_gt.add(gi); matched_pred.add(pi)
        elif iou >= IOU_LOC:
            matched_gt.add(gi); matched_pred.add(pi)
            loc_pairs.append((gi, pi))

    fn = [gts[i] for i in range(len(gts)) if i not in matched_gt]
    fp = [preds[i] for i in range(len(preds)) if i not in matched_pred]
    tp_count = len(matched_gt) - len(loc_pairs)
    return {
        "filename": img_path.name,
        "n_gt": len(gts), "n_pred": len(preds),
        "tp": tp_count, "fp": len(fp), "fn": len(fn), "loc": len(loc_pairs),
        "fn_classes": Counter(class_names[c] for c, _ in fn),
        "fp_classes": Counter(class_names[c] for c, _, _ in fp),
        "gts": gts, "preds": preds,
        "category": _categorise(tp_count, len(fp), len(fn), len(loc_pairs), len(gts)),
    }


def _categorise(tp: int, fp: int, fn: int, loc: int, n_gt: int) -> str:
    if fp and not fn and not loc:
        return "false_positive"
    if fn and not fp and not loc:
        return "false_negative"
    if loc:
        return "localization"
    if fp and fn:
        return "mixed_error"
    if n_gt and tp == n_gt and not fp:
        return "true_positive"
    if not n_gt and not fp:
        return "true_negative"
    return "other"


def draw_case(img_path: Path, case: dict, class_names: dict[int, str]) -> Image.Image:
    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    for cid, box in case["gts"]:
        draw.rectangle(box, outline=GT_COLOR, width=3)
    for cid, box, conf in case["preds"]:
        color = CLASS_COLORS_RGB.get(cid, (255, 255, 0))
        draw.rectangle(box, outline=color, width=2)
        draw.text((box[0] + 3, box[1] + 3),
                  f"{class_names[cid]} {conf:.2f}", fill=color)
    return img


def colour_prior_probe(engine: DetectionEngine, out: Path) -> dict:
    """Measure how much the model relies on colour alone, with no structure.

    Feeds the detector synthetic images that contain no fire and no texture:
    flat colour fields and random noise. Anything detected here is, by
    construction, a false positive driven purely by colour statistics.
    """
    import numpy as np

    probes = {
        "flat_orange": np.full((320, 320, 3), (30, 120, 240), np.uint8),   # BGR
        "flat_red": np.full((320, 320, 3), (40, 40, 200), np.uint8),
        "flat_grey": np.full((320, 320, 3), 128, np.uint8),
        "flat_blue": np.full((320, 320, 3), (200, 120, 40), np.uint8),
        "random_noise": np.random.default_rng(42).integers(
            0, 255, (320, 320, 3), dtype=np.uint8),
    }
    findings = {}
    for name, img in probes.items():
        result = engine.predict(img, conf=CONF, iou=0.5, draw=False)
        findings[name] = {
            "detections": result.counts["total"],
            "fire": result.counts["fire"],
            "smoke": result.counts["smoke"],
            "max_confidence": round(
                max((d.confidence for d in result.detections), default=0.0), 3),
            "classes": sorted({d.class_name for d in result.detections}),
        }
        log.info("colour probe %-13s -> %d detection(s), max conf %.3f",
                 name, findings[name]["detections"], findings[name]["max_confidence"])
    payload = {"confidence_threshold": CONF, "probes": findings}
    with (out / "colour_prior_probe.json").open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    return payload


def main() -> int:
    if not paths.FINAL_MODEL_PATH.exists():
        log.error("final model missing - run scripts/benchmark.py first")
        return 1
    engine = DetectionEngine(paths.FINAL_MODEL_PATH)
    class_names = engine.class_names
    test_dir = paths.PROCESSED_DATASET_DIR / "test" / "images"
    out = paths.ERROR_ANALYSIS_OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)

    cases = []
    for i, img_path in enumerate(sorted(test_dir.iterdir())):
        cases.append(analyse_image(engine, img_path, class_names))
        if (i + 1) % 100 == 0:
            log.info("analysed %d test images", i + 1)

    df = pd.DataFrame([{k: v for k, v in c.items()
                        if k in ("filename", "n_gt", "n_pred", "tp", "fp",
                                 "fn", "loc", "category")} for c in cases])
    df.to_csv(out / "per_image_errors.csv", index=False)

    summary = {
        "operating_confidence": CONF,
        "images": len(cases),
        "category_counts": df.category.value_counts().to_dict(),
        "total": {k: int(df[k].sum()) for k in ("tp", "fp", "fn", "loc")},
        "missed_by_class": dict(sum((Counter(c["fn_classes"]) for c in cases), Counter())),
        "false_positives_by_class": dict(sum((Counter(c["fp_classes"]) for c in cases), Counter())),
    }
    with (out / "error_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    log.info("summary: %s", summary)

    # galleries: up to 9 representative cases per category
    rng = np.random.default_rng(42)
    gallery_tiles, gallery_titles = [], []
    for category in ("true_positive", "false_positive", "false_negative", "localization"):
        subset = [c for c in cases if c["category"] == category]
        if not subset:
            continue
        picks = list(rng.choice(len(subset), min(9, len(subset)), replace=False))
        gal_dir = out / f"gallery_{category}"
        gal_dir.mkdir(exist_ok=True)
        for j, idx in enumerate(picks):
            case = subset[int(idx)]
            img = draw_case(test_dir / case["filename"], case, class_names)
            img.save(gal_dir / case["filename"])
            if j < 3:      # 3 per category for the combined report figure
                gallery_tiles.append(img)
                gallery_titles.append(f"{category}: {case['filename'][:28]}")
        log.info("gallery_%s: %d examples", category, len(picks))

    if gallery_tiles:
        image_grid(gallery_tiles, cols=3, titles=gallery_titles).save(
            out / "error_gallery.png")

    log.info("--- colour-prior probe (synthetic, structure-free inputs) ---")
    colour_prior_probe(engine, out)
    log.info("error analysis complete -> %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
