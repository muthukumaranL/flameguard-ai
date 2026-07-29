"""Image inference: happy path, no-detection path, corrupt inputs, records."""
from __future__ import annotations

import numpy as np
import pytest

from src.image_inference import (InvalidImageError, annotated_png_bytes,
                                 detect_image, load_image_bytes, load_image_file)
from src.inference import detections_to_records


def test_cpu_inference_on_dataset_image(engine, sample_image_path):
    image = load_image_file(sample_image_path)
    result = detect_image(engine, image, conf=0.25, iou=0.5)
    assert result.inference_ms > 0
    assert result.annotated_bgr is not None
    assert result.annotated_bgr.shape == image.shape
    assert result.counts["total"] == result.counts["fire"] + result.counts["smoke"]
    assert result.status in {"No Hazard Detected", "Fire Detected",
                             "Smoke Detected", "Fire and Smoke Detected"}


def test_no_detection_path(engine, blank_image):
    """The empty-result path must be correct: zero counts, honest status."""
    result = detect_image(engine, blank_image, conf=0.99, iou=0.5)
    assert result.counts["total"] == 0
    assert result.status == "No Hazard Detected"
    assert result.max_confidence(0) is None
    assert result.annotated_bgr is not None      # still returns a drawable frame


def test_known_limitation_flat_colour_false_positive(engine):
    """Characterisation test for a REAL, measured failure mode.

    The detector retains some reliance on colour statistics: at least one flat
    colour field - no fire, no texture, no structure - is still reported as an
    object. The deployed YOLO11n is markedly more robust here than the earlier
    YOLOv8n (it no longer fires on flat orange), but a uniform red field still
    triggers a false positive. Structured-but-meaningless input (random noise)
    is correctly ignored.

    The exact colours that trigger are model-dependent, so rather than hard-code
    one, we sweep a small palette and assert the *property*: some flat colour
    false-positives (the documented limitation persists in reduced form) while
    random noise does not. The report's colour-prior narrative is generated from
    the same measurement (outputs/error_analysis/colour_prior_probe.json).
    """
    palette = {
        "red":    (40, 40, 200),
        "orange": (30, 120, 240),
        "grey":   (128, 128, 128),
        "blue":   (200, 120, 40),
    }
    flat_hits = 0
    for bgr in palette.values():
        field = np.full((320, 320, 3), bgr, dtype=np.uint8)
        if detect_image(engine, field, conf=0.3, iou=0.5).counts["total"] >= 1:
            flat_hits += 1
    assert flat_hits >= 1, (
        "Model no longer false-positives on ANY flat colour field - excellent. "
        "Update the error-analysis section of the report and this test."
    )

    noise = np.random.default_rng(0).integers(0, 255, (320, 320, 3), dtype=np.uint8)
    assert detect_image(engine, noise, conf=0.3, iou=0.5).counts["total"] == 0


def test_corrupt_bytes_raise_invalid_image():
    with pytest.raises(InvalidImageError):
        load_image_bytes(b"this is definitely not an image")


def test_unsupported_suffix_rejected(tmp_path):
    weird = tmp_path / "file.tiff"
    weird.write_bytes(b"anything")
    with pytest.raises(InvalidImageError):
        load_image_file(weird)


def test_detection_records_schema(engine, sample_image_path):
    image = load_image_file(sample_image_path)
    result = detect_image(engine, image, conf=0.05, iou=0.5)
    records = detections_to_records(result, sample_image_path.name)
    expected_keys = {"source_file", "class_id", "class_name", "confidence",
                     "x1", "y1", "x2", "y2", "box_width", "box_height",
                     "image_width", "image_height", "inference_ms",
                     "confidence_threshold", "iou_threshold"}
    for row in records:
        assert set(row.keys()) == expected_keys
        # class_name follows the engine mapping (Fire/Smoke for the fine-tuned
        # model; raw id string when running the COCO fallback in early tests)
        assert row["class_name"] == engine.class_names.get(row["class_id"],
                                                           str(row["class_id"]))
        assert 0 <= row["confidence"] <= 1


def test_annotated_png_roundtrip(engine, blank_image):
    result = detect_image(engine, blank_image, conf=0.5, iou=0.5)
    png = annotated_png_bytes(result)
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    restored = load_image_bytes(png)
    assert restored.shape == blank_image.shape
