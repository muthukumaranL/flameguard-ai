"""Video inference: synthetic-video end-to-end run and failure paths."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from src.video_inference import (InvalidVideoError, VideoStats, find_ffmpeg,
                                 probe_video, process_video)


def _write_synthetic_video(path: Path, frames: int = 12, size: int = 160,
                           fps: float = 10.0) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (size, size))
    assert writer.isOpened()
    for i in range(frames):
        frame = np.full((size, size, 3), (i * 17) % 255, dtype=np.uint8)
        writer.write(frame)
    writer.release()


def test_probe_valid_video(tmp_path: Path):
    vid = tmp_path / "clip.mp4"
    _write_synthetic_video(vid)
    meta = probe_video(vid)
    assert meta["width"] == 160 and meta["height"] == 160
    assert meta["frames"] == 12
    assert meta["fps"] == pytest.approx(10.0, abs=0.5)


def test_probe_invalid_video(tmp_path: Path):
    fake = tmp_path / "fake.mp4"
    fake.write_bytes(b"not a real video container")
    with pytest.raises(InvalidVideoError):
        probe_video(fake)


def test_probe_missing_video(tmp_path: Path):
    with pytest.raises(InvalidVideoError):
        probe_video(tmp_path / "missing.mp4")


def test_process_video_end_to_end(engine, tmp_path: Path):
    vid = tmp_path / "clip.mp4"
    _write_synthetic_video(vid, frames=10)
    out = tmp_path / "clip_pred.mp4"
    progress: list[float] = []
    stats = process_video(engine, vid, out, conf=0.4, iou=0.5,
                          frame_skip=2, progress_cb=progress.append)
    assert isinstance(stats, VideoStats)
    assert out.exists() and out.stat().st_size > 0
    assert stats.frames_processed == 5          # every 2nd of 10 frames
    assert stats.total_frames == 10
    assert progress and progress[-1] == 1.0
    # grey synthetic frames must not contain fire/smoke
    assert stats.total_detections == 0
    meta = probe_video(out)
    assert meta["frames"] == 10                 # skipped frames still written


def test_ffmpeg_is_locatable():
    assert find_ffmpeg() is not None, "imageio-ffmpeg should bundle a binary"
