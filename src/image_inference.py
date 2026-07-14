"""Image-file detection helpers used by the Streamlit image tab and tests."""
from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from src.inference import DetectionEngine, InferenceResult, detections_to_records
from src.utils import setup_logging

log = setup_logging("flameguard.image")

SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


class InvalidImageError(ValueError):
    """Raised when an upload cannot be decoded as a supported image."""


def load_image_bytes(data: bytes) -> np.ndarray:
    """Decode raw upload bytes to a BGR numpy array (EXIF orientation applied)."""
    try:
        with Image.open(io.BytesIO(data)) as im:
            im = ImageOps.exif_transpose(im)
            rgb = np.asarray(im.convert("RGB"))
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise InvalidImageError(f"File could not be decoded as an image: {exc}") from exc
    return rgb[:, :, ::-1].copy()   # RGB -> BGR


def load_image_file(path: Path) -> np.ndarray:
    """Decode an image file to BGR; validates suffix and content."""
    if path.suffix.lower() not in SUPPORTED_SUFFIXES:
        raise InvalidImageError(f"Unsupported image type: {path.suffix}")
    return load_image_bytes(path.read_bytes())


def detect_image(engine: DetectionEngine, image_bgr: np.ndarray, *,
                 conf: float, iou: float, show_labels: bool = True,
                 show_conf: bool = True, line_width: int = 2) -> InferenceResult:
    """Single shared entry point for still-image detection."""
    return engine.predict(image_bgr, conf=conf, iou=iou, draw=True,
                          show_labels=show_labels, show_conf=show_conf,
                          line_width=line_width)


def annotated_png_bytes(result: InferenceResult) -> bytes:
    """Encode the annotated frame as PNG for display/download."""
    if result.annotated_bgr is None:
        raise ValueError("result has no annotated frame")
    rgb = result.annotated_bgr[:, :, ::-1]
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    return buf.getvalue()


def records_csv_bytes(records: list[dict[str, Any]]) -> bytes:
    """Detection records as UTF-8 CSV bytes."""
    import csv
    import io as _io

    if not records:
        return b""
    buf = _io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(records[0].keys()))
    writer.writeheader()
    writer.writerows(records)
    return buf.getvalue().encode("utf-8")


def records_json_bytes(records: list[dict[str, Any]]) -> bytes:
    return json.dumps(records, indent=2).encode("utf-8")


def save_detection_outputs(result: InferenceResult, source_name: str,
                           out_dir: Path) -> dict[str, Path]:
    """Persist annotated image + CSV + JSON for a processed image."""
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(source_name).stem
    records = detections_to_records(result, source_name)
    paths = {
        "image": out_dir / f"{stem}_pred.png",
        "csv": out_dir / f"{stem}_detections.csv",
        "json": out_dir / f"{stem}_detections.json",
    }
    paths["image"].write_bytes(annotated_png_bytes(result))
    paths["csv"].write_bytes(records_csv_bytes(records))
    paths["json"].write_bytes(records_json_bytes(records))
    log.info("saved outputs for %s (%d detections)", source_name, len(records))
    return paths
