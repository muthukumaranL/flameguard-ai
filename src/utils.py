"""Shared utilities: logging, seeding, device selection, small helpers."""
from __future__ import annotations

import logging
import os
import random
import sys
from pathlib import Path

import numpy as np


def setup_logging(name: str = "flameguard", level: int = logging.INFO) -> logging.Logger:
    """Configure and return a namespaced logger (console handler, one-time)."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
                              datefmt="%H:%M:%S")
        )
        logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger


def set_seeds(seed: int = 42) -> None:
    """Seed python, numpy and (when available) torch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:  # torch not installed yet (e.g. doc-only tooling)
        pass


def _cuda_usable() -> bool:
    """True only when CUDA is available AND at least one device is visible.

    ``torch.cuda.is_available()`` alone is not enough: with
    ``CUDA_VISIBLE_DEVICES=""`` (or a driver/runtime mismatch) it can report
    True while ``device_count()`` is 0, and touching device 0 then raises.
    """
    try:
        import torch

        return torch.cuda.is_available() and torch.cuda.device_count() > 0
    except Exception:
        return False


def pick_device() -> str:
    """Return 'cuda:0' when a CUDA GPU is usable, else 'cpu'."""
    return "cuda:0" if _cuda_usable() else "cpu"


def device_label() -> str:
    """Human-readable device description for the UI/report."""
    if _cuda_usable():
        try:
            import torch

            return f"GPU - {torch.cuda.get_device_name(0)}"
        except Exception:  # visible but unusable - fall back rather than crash
            pass
    return "CPU"


def file_size_mb(path: Path) -> float:
    """File size in megabytes (0.0 when missing)."""
    return path.stat().st_size / (1024 * 1024) if path.exists() else 0.0


def human_duration(seconds: float) -> str:
    """Format seconds as e.g. '1h 23m 45s'."""
    s = int(round(seconds))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h:
        return f"{h}h {m}m {sec}s"
    if m:
        return f"{m}m {sec}s"
    return f"{sec}s"
