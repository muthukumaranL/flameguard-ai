"""Video-file detection: frame-by-frame processing with progress reporting.

Memory-safe by construction - one frame is held at a time.  Output is written
with OpenCV (mp4v) and converted to browser-friendly H.264 via the ffmpeg
binary bundled with imageio-ffmpeg when available.
"""
from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import cv2

from src.inference import DetectionEngine, detections_to_records
from src.utils import setup_logging

log = setup_logging("flameguard.video")

FALLBACK_FPS = 25.0     # used when container metadata reports 0/NaN fps


class InvalidVideoError(ValueError):
    """Raised when a video cannot be opened or contains no frames."""


@dataclass
class VideoStats:
    """Aggregated results of one processed video."""

    source_name: str
    width: int = 0
    height: int = 0
    source_fps: float = 0.0
    total_frames: int = 0
    frames_processed: int = 0
    frame_skip: int = 1
    processing_seconds: float = 0.0
    frames_with_fire: int = 0
    frames_with_smoke: int = 0
    max_fire_confidence: float | None = None
    max_smoke_confidence: float | None = None
    total_detections: int = 0
    records: list[dict[str, Any]] = field(default_factory=list)

    @property
    def processing_fps(self) -> float:
        return self.frames_processed / self.processing_seconds if self.processing_seconds else 0.0


def probe_video(path: Path) -> dict[str, Any]:
    """Read container metadata; raises InvalidVideoError when unusable."""
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise InvalidVideoError(f"Could not open video: {path.name}")
        meta = {
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": float(cap.get(cv2.CAP_PROP_FPS)),
            "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        }
        ok, _ = cap.read()
        if not ok:
            raise InvalidVideoError(f"Video contains no readable frames: {path.name}")
        return meta
    finally:
        cap.release()


def find_ffmpeg() -> str | None:
    """Locate an ffmpeg binary (bundled wheel first, then PATH)."""
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        import shutil as _shutil

        return _shutil.which("ffmpeg")


def convert_to_h264(src: Path, dst: Path) -> bool:
    """Re-encode to H.264 yuv420p for browser playback. True on success."""
    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        log.warning("ffmpeg unavailable - keeping mp4v output (may not play in browser)")
        return False
    cmd = [ffmpeg, "-y", "-i", str(src), "-c:v", "libx264",
           "-preset", "fast", "-pix_fmt", "yuv420p", "-an", str(dst)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        log.error("ffmpeg conversion failed: %s", proc.stderr[-400:])
        return False
    return True


def process_video(engine: DetectionEngine, input_path: Path, output_path: Path, *,
                  conf: float, iou: float, frame_skip: int = 1,
                  show_labels: bool = True, show_conf: bool = True,
                  line_width: int = 2,
                  progress_cb: Callable[[float], None] | None = None,
                  cancel_cb: Callable[[], bool] | None = None) -> VideoStats:
    """Run detection over a video file, writing an annotated copy.

    ``frame_skip=n`` runs the detector on every n-th frame; skipped frames are
    written with the most recent detection overlay so playback stays smooth.
    Only actually-processed frames contribute rows to ``stats.records``.
    """
    meta = probe_video(input_path)
    fps = meta["fps"] if meta["fps"] and meta["fps"] > 0 else FALLBACK_FPS
    stats = VideoStats(source_name=input_path.name, width=meta["width"],
                       height=meta["height"], source_fps=fps,
                       total_frames=meta["frames"], frame_skip=frame_skip)

    cap = cv2.VideoCapture(str(input_path))
    tmp_out = output_path.with_suffix(".raw.mp4")
    writer = cv2.VideoWriter(str(tmp_out), cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (meta["width"], meta["height"]))
    if not writer.isOpened():
        cap.release()
        raise InvalidVideoError("Could not open video writer (codec unavailable)")

    started = time.perf_counter()
    frame_idx = 0
    last_annotated = None
    try:
        while True:
            if cancel_cb and cancel_cb():
                log.info("processing cancelled at frame %d", frame_idx)
                break
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx % frame_skip == 0:
                result = engine.predict(frame, conf=conf, iou=iou, draw=True,
                                        show_labels=show_labels, show_conf=show_conf,
                                        line_width=line_width)
                stats.frames_processed += 1
                if result.counts["fire"]:
                    stats.frames_with_fire += 1
                    mfc = result.max_confidence(0)
                    stats.max_fire_confidence = max(stats.max_fire_confidence or 0.0, mfc)
                if result.counts["smoke"]:
                    stats.frames_with_smoke += 1
                    msc = result.max_confidence(1)
                    stats.max_smoke_confidence = max(stats.max_smoke_confidence or 0.0, msc)
                stats.total_detections += result.counts["total"]
                timestamp = frame_idx / fps
                for row in detections_to_records(result, input_path.name):
                    row.update({"frame_number": frame_idx,
                                "timestamp_s": round(timestamp, 3),
                                "fire_count_in_frame": result.counts["fire"],
                                "smoke_count_in_frame": result.counts["smoke"]})
                    stats.records.append(row)
                last_annotated = result.annotated_bgr
                writer.write(last_annotated)
            else:
                writer.write(last_annotated if last_annotated is not None else frame)
            frame_idx += 1
            if progress_cb and stats.total_frames > 0 and frame_idx % 10 == 0:
                progress_cb(min(frame_idx / stats.total_frames, 1.0))
    finally:
        cap.release()
        writer.release()
    stats.processing_seconds = time.perf_counter() - started
    if progress_cb:
        progress_cb(1.0)

    if convert_to_h264(tmp_out, output_path):
        tmp_out.unlink(missing_ok=True)
    else:
        tmp_out.replace(output_path)
    log.info("video done: %d/%d frames processed in %.1fs (%.1f fps), %d detections",
             stats.frames_processed, stats.total_frames, stats.processing_seconds,
             stats.processing_fps, stats.total_detections)
    return stats
