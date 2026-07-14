"""Webcam building blocks that can be tested without a physical camera."""
from __future__ import annotations

from dataclasses import dataclass, field

from src.inference import StatusSmoother


@dataclass
class _FakeResult:
    counts: dict = field(default_factory=lambda: {"fire": 0, "smoke": 0, "total": 0})


def _result(fire: int = 0, smoke: int = 0) -> _FakeResult:
    return _FakeResult(counts={"fire": fire, "smoke": smoke, "total": fire + smoke})


def test_smoother_requires_min_hits():
    s = StatusSmoother(window=5, min_hits=2)
    assert s.update(_result(fire=1)) == "No Hazard Detected"   # 1 hit < 2
    assert s.update(_result(fire=1)) == "Fire Detected"        # 2 hits


def test_smoother_suppresses_single_frame_flicker():
    s = StatusSmoother(window=5, min_hits=2)
    for _ in range(5):
        s.update(_result())
    assert s.update(_result(smoke=1)) == "No Hazard Detected"  # lone flicker
    assert s.update(_result()) == "No Hazard Detected"


def test_smoother_both_classes():
    s = StatusSmoother(window=3, min_hits=2)
    s.update(_result(fire=1, smoke=1))
    assert s.update(_result(fire=1, smoke=1)) == "Fire and Smoke Detected"


def test_smoother_recovers_after_window():
    s = StatusSmoother(window=3, min_hits=2)
    s.update(_result(fire=2))
    s.update(_result(fire=2))
    assert s.update(_result(fire=1)) == "Fire Detected"
    for _ in range(3):
        status = s.update(_result())
    assert status == "No Hazard Detected"


def test_open_camera_returns_capture_object():
    """open_camera must not raise even when no camera exists."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from src.webcam_inference import open_camera

    cap = open_camera(99)          # index that will not exist
    try:
        assert not cap.isOpened()  # graceful: no exception, just not opened
    finally:
        cap.release()
