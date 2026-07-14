"""Configuration loading for FlameGuard AI.

Thin YAML loaders with caching; every config file lives in ``config/``.
"""
from __future__ import annotations

import functools
from typing import Any

import yaml

from src.paths import CONFIG_DIR, PROJECT_ROOT


@functools.lru_cache(maxsize=None)
def _load_yaml(name: str) -> dict[str, Any]:
    path = CONFIG_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Missing config file: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"Config {path} did not parse to a mapping")
    return data


def load_app_config() -> dict[str, Any]:
    """Streamlit application settings (config/app_config.yaml)."""
    return _load_yaml("app_config.yaml")


def load_training_config() -> dict[str, Any]:
    """Training experiment definitions (config/training_config.yaml)."""
    return _load_yaml("training_config.yaml")


def load_class_names() -> dict[int, str]:
    """Class-id -> name mapping (config/class_names.yaml)."""
    raw = _load_yaml("class_names.yaml")["names"]
    return {int(k): str(v) for k, v in raw.items()}


def load_class_colors() -> dict[int, tuple[int, int, int]]:
    """Class-id -> BGR colour tuple for OpenCV drawing."""
    raw = _load_yaml("class_names.yaml").get("colors", {})
    return {int(k): tuple(int(c) for c in v) for k, v in raw.items()}


def resolve_model_path() -> "Path":  # noqa: F821 - forward ref for docs
    """Absolute path to the final trained model as configured for the app."""
    from pathlib import Path

    rel = load_app_config()["model_path"]
    return (PROJECT_ROOT / rel).resolve()
