"""Stabilize frame-level fire/smoke detections before raising an alert.

A single noisy frame should not trigger an incident. This stateful filter enters
alert state only after enough positive frames in a rolling window and releases
only after a configurable run of negative frames, reducing flicker in video.
"""
from __future__ import annotations

from collections import deque


class TemporalAlertFilter:
    def __init__(self, window_size: int = 5, positives_required: int = 3, release_after: int = 3):
        if window_size < 1:
            raise ValueError("window_size must be positive")
        if not 1 <= positives_required <= window_size:
            raise ValueError("positives_required must be within the window")
        if release_after < 1:
            raise ValueError("release_after must be positive")
        self.window = deque(maxlen=window_size)
        self.positives_required = positives_required
        self.release_after = release_after
        self.negative_streak = 0
        self.alerting = False

    def update(self, detected: bool) -> dict:
        detected = bool(detected)
        self.window.append(detected)
        positive_count = sum(self.window)

        if detected:
            self.negative_streak = 0
        else:
            self.negative_streak += 1

        if not self.alerting and positive_count >= self.positives_required:
            self.alerting = True
        elif self.alerting and self.negative_streak >= self.release_after:
            self.alerting = False
            self.window.clear()

        return {
            "alerting": self.alerting,
            "positive_frames": int(positive_count),
            "frames_observed": len(self.window),
            "negative_streak": self.negative_streak,
        }


if __name__ == "__main__":
    filter_ = TemporalAlertFilter(window_size=5, positives_required=3, release_after=2)
    for frame_detection in [False, True, False, True, True, False, True, False, False]:
        print(frame_detection, filter_.update(frame_detection))
