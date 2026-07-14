"""Shared detection engine for FlameGuard AI.

One cached Ultralytics model serves image upload, video processing, browser
webcam and the OpenCV fallback - the prediction path exists exactly once.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from src.config import load_class_names
from src.paths import FINAL_MODEL_PATH
from src.utils import device_label, pick_device, setup_logging

log = setup_logging("flameguard.inference")

FIRE_CLASS_ID = 0
SMOKE_CLASS_ID = 1


@dataclass(frozen=True)
class Detection:
    """One detected object in pixel coordinates."""

    class_id: int
    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def box_width(self) -> float:
        return self.x2 - self.x1

    @property
    def box_height(self) -> float:
        return self.y2 - self.y1


@dataclass
class InferenceResult:
    """Detections plus the annotated frame and timing for one image/frame."""

    detections: list[Detection]
    annotated_bgr: np.ndarray | None
    inference_ms: float
    image_width: int
    image_height: int
    confidence_threshold: float
    iou_threshold: float
    counts: dict[str, int] = field(init=False)

    def __post_init__(self) -> None:
        self.counts = {
            "fire": sum(1 for d in self.detections if d.class_id == FIRE_CLASS_ID),
            "smoke": sum(1 for d in self.detections if d.class_id == SMOKE_CLASS_ID),
            "total": len(self.detections),
        }

    def max_confidence(self, class_id: int) -> float | None:
        scores = [d.confidence for d in self.detections if d.class_id == class_id]
        return max(scores) if scores else None

    @property
    def status(self) -> str:
        fire, smoke = self.counts["fire"] > 0, self.counts["smoke"] > 0
        if fire and smoke:
            return "Fire and Smoke Detected"
        if fire:
            return "Fire Detected"
        if smoke:
            return "Smoke Detected"
        return "No Hazard Detected"


class MissingModelError(FileNotFoundError):
    """Raised when the trained model file is absent."""


class DetectionEngine:
    """Loads the fine-tuned model once and exposes a single predict path."""

    def __init__(self, weights: Path | str = FINAL_MODEL_PATH,
                 device: str | None = None) -> None:
        from ultralytics import YOLO

        weights = Path(weights)
        if not weights.exists():
            raise MissingModelError(
                f"Model weights not found at '{weights}'. "
                "Train the model first (see README: training pipeline) or place "
                "a YOLO .pt file at models/final/best.pt."
            )
        self.weights_path = weights
        self.device = device or pick_device()
        self.model = YOLO(str(weights))
        self.class_names = load_class_names()
        log.info("DetectionEngine ready: %s on %s", weights.name, self.device)

    @property
    def device_description(self) -> str:
        return device_label()

    def predict(self, image_bgr: np.ndarray, *, conf: float = 0.30, iou: float = 0.50,
                draw: bool = True, show_labels: bool = True, show_conf: bool = True,
                line_width: int = 2, max_size: int | None = None) -> InferenceResult:
        """Run detection on a BGR numpy image (OpenCV convention).

        ``max_size`` optionally downscales the longer image side before
        inference (used by the live webcam path on slow CPUs).
        """
        h, w = image_bgr.shape[:2]
        working = image_bgr
        scale = 1.0
        if max_size and max(h, w) > max_size:
            scale = max_size / max(h, w)
            import cv2

            working = cv2.resize(image_bgr, (int(w * scale), int(h * scale)))

        start = time.perf_counter()
        results = self.model.predict(working, conf=conf, iou=iou,
                                     device=self.device, verbose=False)
        elapsed_ms = (time.perf_counter() - start) * 1000
        res = results[0]

        detections: list[Detection] = []
        if res.boxes is not None and len(res.boxes) > 0:
            xyxy = res.boxes.xyxy.cpu().numpy() / scale
            confs = res.boxes.conf.cpu().numpy()
            classes = res.boxes.cls.cpu().numpy().astype(int)
            for (x1, y1, x2, y2), score, cid in zip(xyxy, confs, classes):
                detections.append(Detection(
                    class_id=int(cid),
                    class_name=self.class_names.get(int(cid), str(cid)),
                    confidence=float(score),
                    x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2),
                ))

        annotated = None
        if draw:
            annotated = res.plot(labels=show_labels, conf=show_conf,
                                 line_width=line_width)
            if scale != 1.0:
                import cv2

                annotated = cv2.resize(annotated, (w, h))
        return InferenceResult(
            detections=detections, annotated_bgr=annotated,
            inference_ms=elapsed_ms, image_width=w, image_height=h,
            confidence_threshold=conf, iou_threshold=iou,
        )


def detections_to_records(result: InferenceResult, source_name: str) -> list[dict[str, Any]]:
    """Flatten a result into CSV/JSON rows matching the project schema."""
    rows = []
    for det in result.detections:
        row = asdict(det)
        rows.append({
            "source_file": source_name,
            "class_id": row["class_id"],
            "class_name": row["class_name"],
            "confidence": round(row["confidence"], 4),
            "x1": round(row["x1"], 1), "y1": round(row["y1"], 1),
            "x2": round(row["x2"], 1), "y2": round(row["y2"], 1),
            "box_width": round(det.box_width, 1),
            "box_height": round(det.box_height, 1),
            "image_width": result.image_width,
            "image_height": result.image_height,
            "inference_ms": round(result.inference_ms, 2),
            "confidence_threshold": result.confidence_threshold,
            "iou_threshold": result.iou_threshold,
        })
    return rows


class StatusSmoother:
    """Temporal smoothing of the live status banner to reduce flicker.

    Tracks the last ``window`` frame statuses; a class is reported present when
    it appears in at least ``min_hits`` of them. Frame-level detections remain
    visible on the video itself - only the banner is smoothed.
    """

    def __init__(self, window: int = 5, min_hits: int = 2) -> None:
        self.window = window
        self.min_hits = min_hits
        self._fire: list[bool] = []
        self._smoke: list[bool] = []

    def update(self, result: InferenceResult) -> str:
        self._fire.append(result.counts["fire"] > 0)
        self._smoke.append(result.counts["smoke"] > 0)
        self._fire = self._fire[-self.window:]
        self._smoke = self._smoke[-self.window:]
        fire = sum(self._fire) >= self.min_hits
        smoke = sum(self._smoke) >= self.min_hits
        if fire and smoke:
            return "Fire and Smoke Detected"
        if fire:
            return "Fire Detected"
        if smoke:
            return "Smoke Detected"
        return "No Hazard Detected"


EngineFactory = Callable[[], DetectionEngine]
