"""Slice false-alarm rates by scene/environment for fire detection systems."""

from __future__ import annotations

from collections import defaultdict


def scene_false_alarm_audit(records: list[dict]) -> list[dict]:
    """Summarize false alarms per scene on negative frames.

    Each record requires: scene, ground_truth (bool), detected (bool).
    Positive frames are excluded because this audit targets nuisance alarms.
    """
    totals = defaultdict(lambda: {"negative_frames": 0, "false_alarms": 0})
    for row in records:
        scene = str(row["scene"])
        truth = bool(row["ground_truth"])
        detected = bool(row["detected"])
        if truth:
            continue
        totals[scene]["negative_frames"] += 1
        totals[scene]["false_alarms"] += int(detected)

    result = []
    for scene, counts in totals.items():
        n = counts["negative_frames"]
        result.append(
            {
                "scene": scene,
                **counts,
                "false_alarm_rate": counts["false_alarms"] / n if n else 0.0,
            }
        )
    return sorted(result, key=lambda row: row["false_alarm_rate"], reverse=True)


if __name__ == "__main__":
    sample = [
        {"scene": "kitchen", "ground_truth": False, "detected": True},
        {"scene": "kitchen", "ground_truth": False, "detected": False},
        {"scene": "outdoor", "ground_truth": False, "detected": False},
        {"scene": "warehouse", "ground_truth": True, "detected": True},
    ]
    for row in scene_false_alarm_audit(sample):
        print(row)
