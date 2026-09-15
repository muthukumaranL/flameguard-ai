"""Convert frame detections into incidents and evaluate alert precision.

Frame precision can over-penalize or hide repeated alarms. Incident-level metrics
better reflect what an operator experiences in a deployed safety system.
"""
from __future__ import annotations


def _events(frames: set[int], merge_gap: int) -> list[tuple[int, int]]:
    if not frames:
        return []
    ordered = sorted(frames)
    events = []
    start = end = ordered[0]
    for frame in ordered[1:]:
        if frame - end <= merge_gap + 1:
            end = frame
        else:
            events.append((start, end))
            start = end = frame
    events.append((start, end))
    return events


def incident_precision_audit(predicted_frames, true_frames, merge_gap: int = 2) -> dict[str, float]:
    if merge_gap < 0:
        raise ValueError("merge_gap must be non-negative")
    pred_events = _events(set(map(int, predicted_frames)), merge_gap)
    true_events = _events(set(map(int, true_frames)), merge_gap)

    matched_true = set()
    true_positive = 0
    for ps, pe in pred_events:
        matches = [i for i, (ts, te) in enumerate(true_events) if i not in matched_true and ps <= te and pe >= ts]
        if matches:
            matched_true.add(matches[0])
            true_positive += 1

    false_positive = len(pred_events) - true_positive
    missed = len(true_events) - len(matched_true)
    precision = true_positive / len(pred_events) if pred_events else 0.0
    recall = len(matched_true) / len(true_events) if true_events else 0.0
    return {
        "predicted_incidents": len(pred_events),
        "true_incidents": len(true_events),
        "true_positive_incidents": true_positive,
        "false_alarm_incidents": false_positive,
        "missed_incidents": missed,
        "incident_precision": precision,
        "incident_recall": recall,
    }


if __name__ == "__main__":
    print(incident_precision_audit([10, 11, 12, 40, 41, 90], [9, 10, 11, 42, 43]))
