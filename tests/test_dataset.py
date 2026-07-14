"""Dataset structure and annotation-parser tests."""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.annotation_parser import (MIN_BOX_SIZE, label_path_for_image,
                                   parse_label_file)
from src.paths import PROCESSED_DATASET_DIR
from src.resplit import canonical_stem

processed = pytest.mark.skipif(
    not (PROCESSED_DATASET_DIR / "data.yaml").exists(),
    reason="processed dataset not built yet",
)


@processed
def test_data_yaml_parses_with_two_classes():
    with (PROCESSED_DATASET_DIR / "data.yaml").open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    assert data["nc"] == 2
    names = data["names"]
    names = list(names.values()) if isinstance(names, dict) else names
    assert [n.lower() for n in names] == ["fire", "smoke"]


@processed
@pytest.mark.parametrize("split", ["train", "valid", "test"])
def test_images_and_labels_match(split):
    img_dir = PROCESSED_DATASET_DIR / split / "images"
    lbl_dir = PROCESSED_DATASET_DIR / split / "labels"
    img_stems = {p.stem for p in img_dir.iterdir()}
    lbl_stems = {p.stem for p in lbl_dir.iterdir()}
    assert img_stems == lbl_stems
    assert len(img_stems) > 0


def test_parse_valid_label(tmp_path: Path):
    lbl = tmp_path / "ok.txt"
    lbl.write_text("0 0.5 0.5 0.2 0.3\n1 0.25 0.75 0.1 0.1\n", encoding="utf-8")
    result = parse_label_file(lbl, {0, 1})
    assert len(result.boxes) == 2 and not result.issues
    assert result.boxes[0].class_id == 0
    x1, y1, x2, y2 = result.boxes[0].to_xyxy(640, 640)
    assert (round(x1), round(y1), round(x2), round(y2)) == (256, 224, 384, 416)


@pytest.mark.parametrize("content,expected_issue", [
    ("2 0.5 0.5 0.2 0.2", "invalid class id"),
    ("0 1.5 0.5 0.2 0.2", "outside [0,1]"),
    ("0 0.5 0.5", "expected 5 fields"),
    ("0 a b c d", "non-numeric"),
    (f"0 0.5 0.5 {MIN_BOX_SIZE / 2} 0.2", "degenerate box"),
    ("0 0.05 0.5 0.2 0.2", "outside image bounds"),
])
def test_parse_invalid_labels(tmp_path: Path, content: str, expected_issue: str):
    lbl = tmp_path / "bad.txt"
    lbl.write_text(content + "\n", encoding="utf-8")
    result = parse_label_file(lbl, {0, 1})
    assert not result.boxes
    assert any(expected_issue in issue for issue in result.issues)


def test_empty_label_file_is_valid_negative(tmp_path: Path):
    lbl = tmp_path / "empty.txt"
    lbl.write_text("", encoding="utf-8")
    result = parse_label_file(lbl, {0, 1})
    assert not result.boxes and not result.issues


def test_label_path_for_image():
    img = Path("data/x/train/images/pic.jpg")
    assert label_path_for_image(img) == Path("data/x/train/labels/pic.txt")


@pytest.mark.parametrize("filename,expected", [
    ("WEBSmoke390_jpg.rf.abc123def.jpg", "websmoke390"),
    ("MirrorWEBSmoke390_jpg.rf.0d72f8f.jpg", "websmoke390"),
    ("NoiseWEBFire12_jpg.rf.99aa.jpg", "webfire12"),
    ("Img_1063_jpg.rf.ecd12a3d.jpg", "img_1063"),
])
def test_canonical_stem_strips_augmentation_markers(filename, expected):
    assert canonical_stem(filename) == expected
