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


def test_no_detection_on_blank_image(engine, blank_image):
    result = detect_image(engine, blank_image, conf=0.5, iou=0.5)
    assert result.counts["total"] == 0
    assert result.status == "No Hazard Detected"
    assert result.max_confidence(0) is None


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
        assert row["class_name"] in {"Fire", "Smoke"}
        assert 0 <= row["confidence"] <= 1


def test_annotated_png_roundtrip(engine, blank_image):
    result = detect_image(engine, blank_image, conf=0.5, iou=0.5)
    png = annotated_png_bytes(result)
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    restored = load_image_bytes(png)
    assert restored.shape == blank_image.shape
