"""Simulate cooldown-based alert suppression for real-time detectors."""

from __future__ import annotations


def simulate_alert_cooldown(
    detections: list[bool],
    *,
    fps: float,
    cooldown_seconds: float = 30.0,
) -> dict[str, float | int | list[int]]:
    """Suppress repeated alerts during a cooldown window and report reduction."""
    if fps <= 0 or cooldown_seconds < 0:
        raise ValueError("fps must be positive and cooldown_seconds non-negative")
    cooldown_frames = int(round(fps * cooldown_seconds))
    emitted: list[int] = []
    next_allowed = 0
    raw_alerts = 0

    for frame_idx, detected in enumerate(detections):
        if not detected:
            continue
        raw_alerts += 1
        if frame_idx >= next_allowed:
            emitted.append(frame_idx)
            next_allowed = frame_idx + cooldown_frames + 1

    suppressed = raw_alerts - len(emitted)
    return {
        "raw_positive_frames": raw_alerts,
        "emitted_alerts": len(emitted),
        "suppressed_alerts": suppressed,
        "suppression_rate": suppressed / raw_alerts if raw_alerts else 0.0,
        "alert_frames": emitted,
    }


if __name__ == "__main__":
    stream = [False] * 10 + [True] * 8 + [False] * 30 + [True] * 5
    print(simulate_alert_cooldown(stream, fps=2.0, cooldown_seconds=10.0))
