"""Output generation: files written, formats valid, temp hygiene."""
from __future__ import annotations

import csv
import io
import json

from src.image_inference import (load_image_file, records_csv_bytes,
                                 records_json_bytes, save_detection_outputs)
from src.inference import detections_to_records


def test_save_detection_outputs_creates_three_files(engine, sample_image_path, tmp_path):
    image = load_image_file(sample_image_path)
    result = engine.predict(image, conf=0.25, iou=0.5)
    paths = save_detection_outputs(result, sample_image_path.name, tmp_path)
    assert paths["image"].exists() and paths["image"].stat().st_size > 0
    assert paths["csv"].exists()
    assert paths["json"].exists()
    data = json.loads(paths["json"].read_text(encoding="utf-8"))
    assert isinstance(data, list)


def test_records_csv_parses_back(engine, sample_image_path):
    image = load_image_file(sample_image_path)
    result = engine.predict(image, conf=0.05, iou=0.5)
    records = detections_to_records(result, "x.jpg")
    blob = records_csv_bytes(records)
    if records:
        rows = list(csv.DictReader(io.StringIO(blob.decode("utf-8"))))
        assert len(rows) == len(records)
        assert rows[0]["source_file"] == "x.jpg"
    else:
        assert blob == b""


def test_records_json_valid_when_empty(engine, blank_image):
    """A no-detection result must serialise to an empty JSON array, not to null."""
    result = engine.predict(blank_image, conf=0.99, iou=0.5)
    blob = records_json_bytes(detections_to_records(result, "blank.png"))
    assert json.loads(blob) == []


def test_no_stray_files_outside_tmp(engine, sample_image_path, tmp_path):
    before = set(tmp_path.iterdir())
    image = load_image_file(sample_image_path)
    result = engine.predict(image, conf=0.25, iou=0.5)
    save_detection_outputs(result, "probe.jpg", tmp_path / "sub")
    after = set(tmp_path.iterdir())
    assert after - before == {tmp_path / "sub"}
