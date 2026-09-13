"""Audit temporal bursts of false-positive detections in video streams.

Frame-level precision can hide whether false alarms arrive as isolated blips or
as long bursts. This utility summarizes consecutive false-positive runs so a
real-time alert system can tune persistence and cooldown rules.
"""

from __future__ import annotations

from collections.abc import Iterable


def false_alarm_bursts(false_positive_frames: Iterable[bool], *, fps: float = 30.0) -> dict[str, float]:
    if fps <= 0:
        raise ValueError("fps must be positive")

    flags = [bool(value) for value in false_positive_frames]
    if not flags:
        raise ValueError("false_positive_frames cannot be empty")

    bursts: list[int] = []
    current = 0
    for flag in flags:
        if flag:
            current += 1
        elif current:
            bursts.append(current)
            current = 0
    if current:
        bursts.append(current)

    false_frames = sum(flags)
    if not bursts:
        return {
            "frames": float(len(flags)),
            "false_positive_frames": 0.0,
            "false_positive_rate": 0.0,
            "burst_count": 0.0,
            "mean_burst_frames": 0.0,
            "max_burst_frames": 0.0,
            "max_burst_seconds": 0.0,
        }

    return {
        "frames": float(len(flags)),
        "false_positive_frames": float(false_frames),
        "false_positive_rate": false_frames / len(flags),
        "burst_count": float(len(bursts)),
        "mean_burst_frames": sum(bursts) / len(bursts),
        "max_burst_frames": float(max(bursts)),
        "max_burst_seconds": max(bursts) / fps,
    }


if __name__ == "__main__":
    # Example: three short blips plus one persistent six-frame false alarm.
    stream = [False, True, False, False, True, True, False, True, False] + [True] * 6 + [False] * 8
    result = false_alarm_bursts(stream, fps=30.0)
    print("False-alarm burst audit")
    for metric, value in result.items():
        print(f"{metric:>24}: {value:.4f}")
