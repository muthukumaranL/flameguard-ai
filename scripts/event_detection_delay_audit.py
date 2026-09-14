"""Measure event-level detection recall and time-to-detect for video alerts.

Frame metrics can look healthy while an incident is detected too late to be
useful. This utility evaluates whether each labelled event was detected and how
long the first alert took after event onset.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence


def event_detection_delay_audit(
    events: Sequence[tuple[int, int]],
    detected_frames: Iterable[int],
    *,
    fps: float = 30.0,
) -> dict[str, float]:
    if fps <= 0:
        raise ValueError("fps must be positive")
    if not events:
        raise ValueError("events cannot be empty")

    detections = sorted({int(frame) for frame in detected_frames})
    delays: list[int] = []
    missed = 0

    for start, end in events:
        if start < 0 or end < start:
            raise ValueError("each event must satisfy 0 <= start <= end")
        first = next((frame for frame in detections if start <= frame <= end), None)
        if first is None:
            missed += 1
        else:
            delays.append(first - start)

    detected_events = len(events) - missed
    delay_seconds = [delay / fps for delay in delays]

    return {
        "events": float(len(events)),
        "detected_events": float(detected_events),
        "missed_events": float(missed),
        "event_recall": detected_events / len(events),
        "mean_detection_delay_seconds": sum(delay_seconds) / len(delay_seconds) if delay_seconds else float("nan"),
        "max_detection_delay_seconds": max(delay_seconds) if delay_seconds else float("nan"),
    }


if __name__ == "__main__":
    labelled_events = [(90, 180), (360, 450), (700, 760)]
    alerts = [104, 120, 365, 370, 800]

    print("Event detection delay audit")
    for metric, value in event_detection_delay_audit(labelled_events, alerts, fps=30.0).items():
        print(f"{metric:>30}: {value:.4f}")
