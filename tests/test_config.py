"""Configuration and path-layer tests."""
from __future__ import annotations

from pathlib import Path

from src import paths
from src.config import (load_app_config, load_class_colors, load_class_names,
                        load_training_config)


def test_project_root_is_this_repo():
    assert (paths.PROJECT_ROOT / "app.py").exists()
    assert (paths.PROJECT_ROOT / "config" / "app_config.yaml").exists()


def test_ensure_output_dirs(tmp_path, monkeypatch):
    paths.ensure_output_dirs()
    assert paths.EDA_OUTPUT_DIR.exists()
    assert paths.VALIDATION_OUTPUT_DIR.exists()


def test_class_names_mapping():
    names = load_class_names()
    assert names == {0: "Fire", 1: "Smoke"}


def test_class_colors_are_bgr_triples():
    colors = load_class_colors()
    assert set(colors) == {0, 1}
    for c in colors.values():
        assert len(c) == 3 and all(0 <= v <= 255 for v in c)


def test_app_config_required_keys():
    cfg = load_app_config()
    for key in ("app_title", "model_path", "defaults", "disclaimer",
                "supported_image_types", "supported_video_types", "limits"):
        assert key in cfg
    assert 0 < cfg["defaults"]["confidence_threshold"] < 1
    assert 0 < cfg["defaults"]["iou_threshold"] < 1
    assert "educational" in cfg["disclaimer"]


def test_training_config_experiments():
    cfg = load_training_config()
    exps = cfg["experiments"]
    for required in ("e1_baseline_v8n", "e2_stronger_v8s", "e5_final"):
        assert required in exps
        assert exps[required]["epochs"] > 0
        assert Path(exps[required]["save_dir"]).parts[0] == "outputs"


def test_rel_to_root():
    assert paths.rel_to_root(paths.EDA_OUTPUT_DIR) == str(Path("outputs") / "eda")
