"""Surface cameras that generate disproportionate alert volume."""

from __future__ import annotations

from collections import defaultdict


def camera_alert_rate_audit(events: list[dict[str, object]]) -> list[dict[str, float | str]]:
    """Calculate alert rate and positive-frame share for every camera."""
    if not events:
        raise ValueError("events cannot be empty")

    totals: dict[str, int] = defaultdict(int)
    alerts: dict[str, int] = defaultdict(int)
    for event in events:
        camera = str(event.get("camera_id", "")).strip()
        if not camera:
            raise ValueError("every event needs a camera_id")
        totals[camera] += 1
        alerts[camera] += int(bool(event.get("alert", False)))

    total_alerts = sum(alerts.values())
    rows: list[dict[str, float | str]] = []
    for camera in sorted(totals):
        count = totals[camera]
        alert_count = alerts[camera]
        rows.append({
            "camera_id": camera,
            "frames": float(count),
            "alerts": float(alert_count),
            "alert_rate": float(alert_count / count),
            "share_of_all_alerts": float(alert_count / total_alerts) if total_alerts else 0.0,
        })
    return sorted(rows, key=lambda item: float(item["alert_rate"]), reverse=True)


if __name__ == "__main__":
    sample = [
        {"camera_id": "warehouse", "alert": True},
        {"camera_id": "warehouse", "alert": False},
        {"camera_id": "lobby", "alert": False},
        {"camera_id": "lobby", "alert": False},
    ]
    print(camera_alert_rate_audit(sample))