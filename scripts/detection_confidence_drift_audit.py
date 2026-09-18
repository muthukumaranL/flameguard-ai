"""Audit detector confidence drift between reference and live inference windows."""

from __future__ import annotations

import numpy as np


def confidence_drift_audit(
    reference: list[float],
    candidate: list[float],
    *,
    alert_threshold: float = 0.5,
) -> dict[str, float]:
    """Track score-location, tail, and threshold-crossing shifts."""
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(candidate, dtype=float)
    if ref.size == 0 or cur.size == 0:
        raise ValueError("confidence windows must be non-empty")
    if not np.all(np.isfinite(ref)) or not np.all(np.isfinite(cur)):
        raise ValueError("confidence values must be finite")
    if np.any((ref < 0) | (ref > 1)) or np.any((cur < 0) | (cur > 1)):
        raise ValueError("confidence values must lie in [0, 1]")
    if not 0 <= alert_threshold <= 1:
        raise ValueError("alert_threshold must lie in [0, 1]")

    ref_rate = float(np.mean(ref >= alert_threshold))
    cur_rate = float(np.mean(cur >= alert_threshold))
    return {
        "mean_shift": float(np.mean(cur) - np.mean(ref)),
        "median_shift": float(np.median(cur) - np.median(ref)),
        "p10_shift": float(np.quantile(cur, 0.10) - np.quantile(ref, 0.10)),
        "p90_shift": float(np.quantile(cur, 0.90) - np.quantile(ref, 0.90)),
        "reference_alert_rate": ref_rate,
        "candidate_alert_rate": cur_rate,
        "alert_rate_shift": cur_rate - ref_rate,
    }


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    ref = np.clip(rng.normal(0.72, 0.12, 1000), 0, 1)
    live = np.clip(rng.normal(0.63, 0.16, 700), 0, 1)
    print(confidence_drift_audit(ref.tolist(), live.tolist()))
