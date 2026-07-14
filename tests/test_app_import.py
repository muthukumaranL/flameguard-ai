"""The application and every src module must at least parse and import."""
from __future__ import annotations

import importlib
import py_compile
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SRC_MODULES = [
    "src.paths", "src.config", "src.utils", "src.annotation_parser",
    "src.dataset_validator", "src.resplit", "src.eda", "src.metrics",
    "src.evaluate", "src.train", "src.inference", "src.image_inference",
    "src.video_inference", "src.visualizations",
]


@pytest.mark.parametrize("module", SRC_MODULES)
def test_src_module_imports(module):
    importlib.import_module(module)


def test_app_compiles():
    py_compile.compile(str(PROJECT_ROOT / "app.py"), doraise=True)


def test_webcam_script_compiles():
    py_compile.compile(str(PROJECT_ROOT / "src" / "webcam_inference.py"), doraise=True)


def test_metrics_json_loadable_when_present():
    """Model-performance tab reads this file; if present it must be valid."""
    import json

    metrics = PROJECT_ROOT / "outputs" / "evaluation" / "metrics_test.json"
    if not metrics.exists():
        pytest.skip("evaluation not run yet")
    data = json.loads(metrics.read_text(encoding="utf-8"))
    for key in ("precision", "recall", "map50", "map50_95", "per_class"):
        assert key in data
