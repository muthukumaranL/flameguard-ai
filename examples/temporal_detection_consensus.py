"""Require short-term temporal consensus before escalating visual detections."""

from __future__ import annotations

from collections import deque


class TemporalConsensus:
    """Confirm an alert only when enough recent frames independently support it."""

    def __init__(self, window: int = 5, required: int = 3) -> None:
        if window < 1 or required < 1 or required > window:
            raise ValueError("require 1 <= required <= window")
        self.window = window
        self.required = required
        self._history: deque[bool] = deque(maxlen=window)

    def update(self, detected: bool) -> dict[str, int | float | bool]:
        self._history.append(bool(detected))
        positives = sum(self._history)
        observed = len(self._history)
        return {
            "confirmed": positives >= self.required,
            "positive_frames": positives,
            "observed_frames": observed,
            "support_ratio": positives / observed,
        }

    def reset(self) -> None:
        self._history.clear()


if __name__ == "__main__":
    gate = TemporalConsensus(window=5, required=3)
    for frame_detection in [False, True, False, True, True, False]:
        print(gate.update(frame_detection))
