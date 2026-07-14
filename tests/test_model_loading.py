"""Model-loading behaviour, including the missing-model error path."""
from __future__ import annotations

from pathlib import Path

import pytest

from src.inference import DetectionEngine, MissingModelError


def test_missing_model_raises_helpful_error(tmp_path: Path):
    ghost = tmp_path / "not_there.pt"
    with pytest.raises(MissingModelError) as excinfo:
        DetectionEngine(ghost)
    assert "not found" in str(excinfo.value)
    assert "best.pt" in str(excinfo.value)


def test_engine_loads_on_cpu(engine):
    assert engine.device == "cpu"
    assert engine.model is not None
    assert engine.class_names[0] == "Fire"
    assert engine.class_names[1] == "Smoke"


def test_engine_device_description(engine):
    assert isinstance(engine.device_description, str)
    assert engine.device_description
