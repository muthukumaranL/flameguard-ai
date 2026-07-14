"""Shared pytest fixtures for FlameGuard AI."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.paths import FINAL_MODEL_PATH, PROCESSED_DATASET_DIR  # noqa: E402


def _fallback_weights() -> Path | None:
    """Best available weights: the trained final model, else COCO yolov8n."""
    if FINAL_MODEL_PATH.exists():
        return FINAL_MODEL_PATH
    coco = PROJECT_ROOT / "yolov8n.pt"
    return coco if coco.exists() else None


@pytest.fixture(scope="session")
def weights_path() -> Path:
    w = _fallback_weights()
    if w is None:
        pytest.skip("no model weights available (train first or provide yolov8n.pt)")
    return w


@pytest.fixture(scope="session")
def engine(weights_path):
    from src.inference import DetectionEngine

    return DetectionEngine(weights_path, device="cpu")


@pytest.fixture(scope="session")
def sample_image_path() -> Path:
    img_dir = PROCESSED_DATASET_DIR / "test" / "images"
    if not img_dir.exists():
        pytest.skip("processed dataset not built yet")
    images = sorted(img_dir.glob("*.jpg"))
    if not images:
        pytest.skip("processed test split has no images")
    return images[0]


@pytest.fixture()
def blank_image() -> np.ndarray:
    """A uniform grey image - should yield zero detections."""
    return np.full((320, 320, 3), 128, dtype=np.uint8)
