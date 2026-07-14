"""Standalone OpenCV webcam detector - the offline fallback for live demos.

Usage:
    python src/webcam_inference.py [--camera 0] [--conf 0.30] [--iou 0.50]

Opens the local webcam, overlays fire/smoke detections and FPS, and quits
cleanly when Q is pressed.  Works without a browser or network.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2

from src.inference import DetectionEngine, MissingModelError, StatusSmoother
from src.utils import setup_logging

log = setup_logging("flameguard.webcam")

WINDOW = "FlameGuard AI - press Q to quit"


def open_camera(index: int) -> cv2.VideoCapture:
    """Open a webcam; on Windows the DirectShow backend starts much faster."""
    if sys.platform == "win32":
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(index)
    return cap


def run(camera: int, conf: float, iou: float, max_size: int = 640) -> int:
    try:
        engine = DetectionEngine()
    except MissingModelError as exc:
        log.error("%s", exc)
        return 2

    cap = open_camera(camera)
    if not cap.isOpened():
        log.error("Could not open webcam index %d. Is a camera connected and free?",
                  camera)
        return 3

    smoother = StatusSmoother(window=5, min_hits=2)
    fps = 0.0
    alpha = 0.1          # exponential moving average for the FPS readout
    log.info("Live detection running on %s - press Q to quit", engine.device)
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                log.warning("Empty frame from camera - stopping")
                break
            start = time.perf_counter()
            result = engine.predict(frame, conf=conf, iou=iou, draw=True,
                                    max_size=max_size)
            frame_time = time.perf_counter() - start
            fps = (1 - alpha) * fps + alpha * (1.0 / frame_time) if fps else 1.0 / frame_time

            display = result.annotated_bgr
            status = smoother.update(result)
            color = (0, 200, 0) if status == "No Hazard Detected" else (0, 0, 255)
            cv2.putText(display, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, color, 2, cv2.LINE_AA)
            cv2.putText(display,
                        f"FPS {fps:4.1f} | fire {result.counts['fire']} | "
                        f"smoke {result.counts['smoke']} | {engine.device}",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow(WINDOW, display)
            if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
    log.info("Camera released - goodbye")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0, help="webcam index")
    parser.add_argument("--conf", type=float, default=0.30)
    parser.add_argument("--iou", type=float, default=0.50)
    parser.add_argument("--max-size", type=int, default=640,
                        help="downscale longer side before inference")
    args = parser.parse_args()
    return run(args.camera, args.conf, args.iou, args.max_size)


if __name__ == "__main__":
    raise SystemExit(main())
