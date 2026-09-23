"""Measure temporal instability in frame-level fire/smoke detections."""

from __future__ import annotations

import numpy as np


def detection_flicker_audit(
    detections: np.ndarray,
    fps: float,
    min_run_seconds: float = 0.5,
) -> dict[str, float | int]:
    """Report state transitions and short positive runs that create visual alert flicker."""
    states = np.asarray(detections, dtype=int)
    if states.ndim != 1 or states.size == 0 or not np.all(np.isin(states, [0, 1])):
        raise ValueError("detections must be a non-empty 1D binary array")
    if fps <= 0 or min_run_seconds <= 0:
        raise ValueError("fps and min_run_seconds must be positive")

    transitions = int(np.sum(states[1:] != states[:-1])) if states.size > 1 else 0
    positive_runs: list[int] = []
    run = 0
    for state in states:
        if state:
            run += 1
        elif run:
            positive_runs.append(run)
            run = 0
    if run:
        positive_runs.append(run)

    min_frames = max(1, int(np.ceil(min_run_seconds * fps)))
    short_runs = [length for length in positive_runs if length < min_frames]
    duration_minutes = states.size / fps / 60.0
    return {
        "frames": int(states.size),
        "state_transitions": transitions,
        "transitions_per_minute": float(transitions / duration_minutes) if duration_minutes else 0.0,
        "positive_runs": len(positive_runs),
        "short_positive_runs": len(short_runs),
        "flicker_run_rate": float(len(short_runs) / len(positive_runs)) if positive_runs else 0.0,
        "median_positive_run_seconds": float(np.median(positive_runs) / fps) if positive_runs else 0.0,
    }


if __name__ == "__main__":
    stream = np.array([0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0])
    print(detection_flicker_audit(stream, fps=4.0, min_run_seconds=0.75))
