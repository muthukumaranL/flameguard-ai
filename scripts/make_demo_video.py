"""Build a demo video clip from the dataset's own sequential frames.

The Roboflow export contains families of consecutive frames extracted from source
videos (e.g. `fire1-0001-`, `fire1-0002-`, ...). This script reassembles a run of
consecutive frames from the **test** split back into a short clip, so the video
feature and the classroom demo have real motion footage that the model has never
trained on.

Outputs:
    outputs/sample_inputs/demo_clip.mp4          - the raw clip (input for the demo)
    presentation/backup_demo/demo_clip_pred.mp4  - a pre-processed annotated copy
                                                   (the fallback if the live demo fails)

Usage:
    python scripts/make_demo_video.py
"""
from __future__ import annotations

import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2

from src import paths
from src.inference import DetectionEngine
from src.utils import setup_logging
from src.video_inference import convert_to_h264, process_video

log = setup_logging("flameguard.demo-video")

FPS = 8.0            # the frames are sampled from video, not contiguous at 30fps
MIN_FRAMES = 24


def frame_families(image_dir: Path) -> dict[str, list[tuple[int, Path]]]:
    """Group test images into (family -> ordered frames) using their frame index."""
    families: dict[str, list[tuple[int, Path]]] = defaultdict(list)
    for p in image_dir.iterdir():
        m = re.match(r"^([A-Za-z_]+?)[-_]?(\d+)[-_]", p.name)
        if not m:
            continue
        family, idx = m.group(1).lower(), int(m.group(2))
        families[family].append((idx, p))
    for fam in families:
        families[fam].sort()
    return families


def _positive_rate(frames: list[tuple[int, Path]]) -> float:
    """Fraction of frames whose ground-truth label file is non-empty."""
    from src.annotation_parser import label_path_for_image

    positive = 0
    for _, p in frames:
        lbl = label_path_for_image(p)
        if lbl.exists() and lbl.read_text(encoding="utf-8").strip():
            positive += 1
    return positive / len(frames) if frames else 0.0


def main() -> int:
    test_dir = paths.PROCESSED_DATASET_DIR / "test" / "images"
    if not test_dir.exists():
        log.error("processed test split missing - run scripts/validate_dataset.py")
        return 1

    families = frame_families(test_dir)
    # A demo clip must actually contain fire or smoke: rank by how many frames are
    # annotated, and only then by length. (The largest family is the background set.)
    scored = [(fam, frames, _positive_rate(frames))
              for fam, frames in families.items() if len(frames) >= MIN_FRAMES]
    scored.sort(key=lambda t: (t[2], len(t[1])), reverse=True)
    if not scored or scored[0][2] < 0.5:
        log.error("no frame family with >=%d frames and mostly-annotated content found",
                  MIN_FRAMES)
        return 1
    family, frames, rate = scored[0]
    frames = frames[:120]
    log.info("using family '%s': %d frames, %.0f%% contain annotated fire/smoke",
             family, len(frames), rate * 100)

    first = cv2.imread(str(frames[0][1]))
    h, w = first.shape[:2]
    paths.SAMPLE_INPUTS_DIR.mkdir(parents=True, exist_ok=True)
    raw = paths.SAMPLE_INPUTS_DIR / "demo_clip.raw.mp4"
    out = paths.SAMPLE_INPUTS_DIR / "demo_clip.mp4"

    writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (w, h))
    for _, p in frames:
        img = cv2.imread(str(p))
        if img is None:
            continue
        if img.shape[:2] != (h, w):
            img = cv2.resize(img, (w, h))
        writer.write(img)
    writer.release()

    if convert_to_h264(raw, out):
        raw.unlink(missing_ok=True)
    else:
        raw.replace(out)
    log.info("demo clip -> %s (%d frames, %dx%d @ %.0f fps)",
             paths.rel_to_root(out), len(frames), w, h, FPS)

    if not paths.FINAL_MODEL_PATH.exists():
        log.warning("no trained model yet - skipping the annotated backup copy")
        return 0

    backup_dir = paths.PRESENTATION_DIR / "backup_demo"
    backup_dir.mkdir(parents=True, exist_ok=True)
    engine = DetectionEngine(paths.FINAL_MODEL_PATH)
    conf = 0.30
    if paths.FINAL_MODEL_METADATA_PATH.exists():
        import yaml

        conf = float(yaml.safe_load(
            paths.FINAL_MODEL_METADATA_PATH.read_text(encoding="utf-8")
        ).get("confidence_threshold") or conf)

    stats = process_video(engine, out, backup_dir / "demo_clip_pred.mp4",
                          conf=conf, iou=0.5, frame_skip=1)
    log.info("backup annotated clip -> %s | %d/%d frames with detections, "
             "%d total detections",
             paths.rel_to_root(backup_dir / "demo_clip_pred.mp4"),
             max(stats.frames_with_fire, stats.frames_with_smoke),
             stats.frames_processed, stats.total_detections)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
