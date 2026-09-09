"""Summarize object-detection latency against a real-time FPS budget.

Run:
    python scripts/detection_latency_budget_audit.py

Use this for measured per-frame inference times. It reports median, p95, p99,
worst-case latency, achieved FPS, and the share of frames that miss a target
latency budget. Tail latency matters for live fire/smoke detection because a
fast average can still hide visible stalls.
"""
from __future__ import annotations

import numpy as np


def audit_latency(latency_ms: np.ndarray, target_fps: float = 30.0) -> dict[str, float]:
    if latency_ms.size == 0 or np.any(latency_ms <= 0):
        raise ValueError("latency_ms must contain positive measurements")
    budget_ms = 1000.0 / target_fps
    return {
        "median_ms": float(np.median(latency_ms)),
        "p95_ms": float(np.percentile(latency_ms, 95)),
        "p99_ms": float(np.percentile(latency_ms, 99)),
        "worst_ms": float(np.max(latency_ms)),
        "mean_fps": float(1000.0 / np.mean(latency_ms)),
        "budget_ms": float(budget_ms),
        "budget_miss_rate": float(np.mean(latency_ms > budget_ms)),
    }


def demo() -> None:
    rng = np.random.default_rng(7)
    normal = rng.normal(18.0, 2.5, size=490)
    stalls = np.array([39.0, 45.0, 51.0, 61.0, 34.0, 42.0, 56.0, 37.0, 48.0, 70.0])
    values = np.clip(np.concatenate([normal, stalls]), 1.0, None)
    for key, value in audit_latency(values).items():
        print(f"{key}: {value:.3f}")


if __name__ == "__main__":
    demo()
