"""Produce the 10 sample predictions required for submission.

Samples are chosen from the TEST split using the error analysis, so the set
deliberately contains failures as well as successes:
    2 fire-only, 2 smoke-only, 2 both-classes, 2 difficult/failed, 2 negatives.

For each sample we save: the original image, a ground-truth overlay, the model
prediction, a detection CSV, and a one-line interpretation with a correctness
assessment.  A combined grid figure is written for the report and slides.

Usage:
    python scripts/generate_samples.py
"""
from __future__ import annotations

import csv
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import pandas as pd

from src import paths
from src.annotation_parser import label_path_for_image, parse_label_file
from src.image_inference import save_detection_outputs
from src.inference import DetectionEngine
from src.utils import setup_logging
from src.visualizations import draw_ground_truth, image_grid

log = setup_logging("flameguard.samples")


def _content(classes: set[int]) -> str:
    if classes == {0, 1}:
        return "both"
    if classes == {0}:
        return "fire_only"
    if classes == {1}:
        return "smoke_only"
    return "negative"


def pick_samples(errors: pd.DataFrame, test_dir: Path,
                 class_ids: set[int]) -> list[dict]:
    """Select 10 samples spanning content types and outcomes (2 must be failures)."""
    content_of: dict[str, str] = {}
    for img in sorted(test_dir.iterdir()):
        parsed = parse_label_file(label_path_for_image(img), class_ids)
        content_of[img.name] = _content({b.class_id for b in parsed.boxes})
    errors = errors.copy()
    errors["content"] = errors.filename.map(content_of)

    picks: list[dict] = []
    used: set[str] = set()

    def take(mask, n, group):
        subset = errors[mask & ~errors.filename.isin(used)]
        for _, row in subset.head(n).iterrows():
            used.add(row.filename)
            picks.append({"filename": row.filename, "group": group,
                          "content": row.content, "category": row.category,
                          "n_gt": int(row.n_gt), "tp": int(row.tp),
                          "fp": int(row.fp), "fn": int(row.fn), "loc": int(row.loc)})

    ok = errors.category == "true_positive"
    take(ok & (errors.content == "fire_only"), 2, "fire-only (success)")
    take(ok & (errors.content == "smoke_only"), 2, "smoke-only (success)")
    take(ok & (errors.content == "both"), 2, "fire + smoke (success)")
    # difficult / failure cases - the required unsuccessful examples
    take(errors.category.isin(["false_negative", "mixed_error"]) & (errors.n_gt > 0),
         1, "difficult (missed detection)")
    take(errors.category.isin(["localization", "false_positive"]) & (errors.n_gt > 0),
         1, "difficult (localisation / false positive)")
    # negatives: background images (no ground truth)
    take((errors.content == "negative") & (errors.category == "true_negative"),
         1, "negative (correctly empty)")
    take((errors.content == "negative") & (errors.category == "false_positive"),
         1, "negative (false alarm)")
    # top up with any remaining true positives if a bucket was empty
    if len(picks) < 10:
        take(ok, 10 - len(picks), "additional success")
    return picks[:10]


def interpret(s: dict) -> tuple[str, str]:
    """Return (assessment, interpretation) for a sample."""
    if s["fn"] == 0 and s["fp"] == 0 and s["loc"] == 0 and s["n_gt"] > 0:
        return ("Correct",
                f"All {s['n_gt']} annotated object(s) detected with correct class and "
                f"well-placed boxes.")
    if s["n_gt"] == 0 and s["fp"] == 0:
        return ("Correct",
                "Background image with no fire or smoke; the model correctly reported "
                "no detections.")
    if s["n_gt"] == 0 and s["fp"] > 0:
        return ("Incorrect",
                f"False alarm: {s['fp']} detection(s) on a background image - typically "
                f"cloud, haze, sunset glow or warm artificial light mimicking smoke or "
                f"flame.")
    if s["fn"] > 0 and s["fp"] == 0:
        return ("Incorrect",
                f"Missed {s['fn']} of {s['n_gt']} annotated object(s). Characteristic of "
                f"thin/transparent smoke, small distant flames, or low-light scenes.")
    if s["loc"] > 0:
        return ("Partially correct",
                f"Object found but localised poorly ({s['loc']} box(es) with IoU below "
                f"0.5) - common for diffuse smoke plumes that have no crisp boundary.")
    return ("Partially correct",
            f"Mixed outcome: {s['tp']} correct, {s['fp']} false positive(s), "
            f"{s['fn']} missed.")


def main() -> int:
    if not paths.FINAL_MODEL_PATH.exists():
        log.error("models/final/best.pt missing - run scripts/benchmark.py first")
        return 1
    err_csv = paths.ERROR_ANALYSIS_OUTPUT_DIR / "per_image_errors.csv"
    if not err_csv.exists():
        log.error("run scripts/error_analysis.py first")
        return 1

    engine = DetectionEngine(paths.FINAL_MODEL_PATH)
    conf = 0.30
    meta = paths.FINAL_MODEL_METADATA_PATH
    if meta.exists():
        import yaml

        conf = float(yaml.safe_load(meta.read_text(encoding="utf-8"))
                     .get("confidence_threshold") or conf)

    test_dir = paths.PROCESSED_DATASET_DIR / "test" / "images"
    picks = pick_samples(pd.read_csv(err_csv), test_dir, set(engine.class_names))
    log.info("selected %d samples", len(picks))

    paths.SAMPLE_INPUTS_DIR.mkdir(parents=True, exist_ok=True)
    paths.SAMPLE_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    rows, tiles, titles = [], [], []

    for i, s in enumerate(picks, start=1):
        src = test_dir / s["filename"]
        stem = f"sample_{i:02d}"
        shutil.copy2(src, paths.SAMPLE_INPUTS_DIR / f"{stem}{src.suffix}")

        parsed = parse_label_file(label_path_for_image(src), set(engine.class_names))
        draw_ground_truth(src, parsed.boxes, engine.class_names).save(
            paths.SAMPLE_OUTPUTS_DIR / f"{stem}_ground_truth.png")

        result = engine.predict(cv2.imread(str(src)), conf=conf, iou=0.5, draw=True)
        save_detection_outputs(result, f"{stem}.jpg", paths.SAMPLE_OUTPUTS_DIR)

        assessment, interpretation = interpret(s)
        confs = "; ".join(f"{d.class_name} {d.confidence:.2f}" for d in result.detections) or "none"
        rows.append({
            "sample": stem,
            "source_file": s["filename"],
            "group": s["group"],
            "ground_truth_objects": s["n_gt"],
            "detections": result.counts["total"],
            "confidences": confs,
            "assessment": assessment,
            "interpretation": interpretation,
        })
        from PIL import Image as PILImage

        tiles.append(PILImage.fromarray(result.annotated_bgr[:, :, ::-1]))
        titles.append(f"{stem}: {s['group']} - {assessment}")

    with (paths.SAMPLE_OUTPUTS_DIR / "sample_summary.csv").open(
            "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    image_grid(tiles, cols=4, cell=300, titles=titles).save(
        paths.SAMPLE_OUTPUTS_DIR / "sample_grid.png")
    log.info("samples written to %s (%d correct, %d incorrect/partial)",
             paths.SAMPLE_OUTPUTS_DIR,
             sum(r["assessment"] == "Correct" for r in rows),
             sum(r["assessment"] != "Correct" for r in rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
