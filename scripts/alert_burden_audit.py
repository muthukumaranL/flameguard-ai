"""Quantify how detector alerts translate into operator workload."""

from __future__ import annotations

import numpy as np


def alert_burden_audit(
    alert_times_seconds: list[float],
    *,
    observation_hours: float,
    burst_gap_seconds: float = 60.0,
) -> dict[str, float]:
    """Report alert rate and burst concentration for an observation window."""
    if observation_hours <= 0 or burst_gap_seconds < 0:
        raise ValueError("observation_hours must be positive and burst gap non-negative")
    times = np.asarray(alert_times_seconds, dtype=float)
    if times.size and (not np.all(np.isfinite(times)) or np.any(times < 0)):
        raise ValueError("alert times must be finite and non-negative")
    if times.size == 0:
        return {"alerts": 0.0, "alerts_per_hour": 0.0, "bursts": 0.0, "largest_burst": 0.0}

    times = np.sort(times)
    burst_sizes = [1]
    for gap in np.diff(times):
        if gap <= burst_gap_seconds:
            burst_sizes[-1] += 1
        else:
            burst_sizes.append(1)
    return {
        "alerts": float(times.size),
        "alerts_per_hour": float(times.size / observation_hours),
        "bursts": float(len(burst_sizes)),
        "largest_burst": float(max(burst_sizes)),
        "burst_alert_share": float(sum(size for size in burst_sizes if size > 1) / times.size),
    }


if __name__ == "__main__":
    alerts = [15, 40, 58, 900, 2500, 2510, 2520, 5100]
    print(alert_burden_audit(alerts, observation_hours=2.0))
