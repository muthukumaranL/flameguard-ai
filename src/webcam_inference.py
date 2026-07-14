"""Standalone OpenCV detector - the offline fallback for live demos.

Opens the local webcam, overlays fire/smoke detections, the status banner and a
measured FPS readout, quits cleanly on Q and always releases the camera. It needs
no browser, no network and no WebRTC handshake, which is precisely why it exists:
browser camera access fails in exactly the situations where a demo must not.

Usage:
    python src/webcam_inference.py                       # default webcam
    python src/webcam_inference.py --camera 1 --conf 0.4 # a different camera
    python src/webcam_inference.py --camera clip.mp4     # a video file instead
    python src/webcam_inference.py --selftest 5          # headless check, no window

Exit codes: 0 ok · 2 model missing · 3 capture source unavailable · 4 no frames.
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


def open_camera(source: int | str) -> cv2.VideoCapture:
    """Open a capture source.

    ``source`` is a webcam index (int) or a path to a video file - the latter
    lets the fallback be exercised on a machine with no camera, and doubles as a
    headless command-line video detector.
    """
    if isinstance(source, str) and not source.isdigit():
        return cv2.VideoCapture(source)
    index = int(source)
    if sys.platform == "win32":       # DirectShow starts much faster on Windows
        return cv2.VideoCapture(index, cv2.CAP_DSHOW)
    return cv2.VideoCapture(index)


def run(camera: int | str, conf: float, iou: float, max_size: int = 640,
        selftest_frames: int = 0) -> int:
    """Live loop. ``selftest_frames`` > 0 runs headlessly for N frames and exits.

    The self-test mode exists so the fallback can be verified in CI and in the
    project's verification loop, where no GUI window can be opened.
    """
    try:
        engine = DetectionEngine()
    except MissingModelError as exc:
        log.error("%s", exc)
        return 2

    cap = open_camera(camera)
    if not cap.isOpened():
        log.error("Could not open capture source %r. Is a camera connected and free?",
                  camera)
        return 3

    if selftest_frames > 0:
        grabbed = 0
        try:
            for _ in range(selftest_frames):
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                result = engine.predict(frame, conf=conf, iou=iou, draw=True,
                                        max_size=max_size)
                grabbed += 1
                log.info("frame %d: %dx%d | %s | fire=%d smoke=%d | %.0f ms",
                         grabbed, frame.shape[1], frame.shape[0], result.status,
                         result.counts["fire"], result.counts["smoke"],
                         result.inference_ms)
        finally:
            cap.release()
            cv2.destroyAllWindows()
        if grabbed == 0:
            log.error("camera opened but returned no frames")
            return 4
        log.info("self-test OK: %d frames captured and processed on %s, camera released",
                 grabbed, engine.device)
        return 0

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
    parser.add_argument("--camera", default=0,
                        help="webcam index (e.g. 0) or a path to a video file")
    parser.add_argument("--conf", type=float, default=0.30)
    parser.add_argument("--iou", type=float, default=0.50)
    parser.add_argument("--max-size", type=int, default=640,
                        help="downscale longer side before inference")
    parser.add_argument("--selftest", type=int, default=0, metavar="N",
                        help="headless check: grab N frames, print results, exit")
    args = parser.parse_args()
    return run(args.camera, args.conf, args.iou, args.max_size, args.selftest)


if __name__ == "__main__":
    raise SystemExit(main())
