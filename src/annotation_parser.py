"""Parsing and validation of YOLO-format annotation files.

A YOLO label file holds one line per object:
    <class_id> <x_center> <y_center> <width> <height>
with all coordinates normalised to [0, 1] relative to the image size.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class YoloBox:
    """A single normalised YOLO annotation."""

    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def aspect_ratio(self) -> float:
        return self.width / self.height if self.height > 0 else float("inf")

    def to_xyxy(self, img_w: int, img_h: int) -> tuple[float, float, float, float]:
        """Convert to absolute (x1, y1, x2, y2) pixel coordinates."""
        x1 = (self.x_center - self.width / 2) * img_w
        y1 = (self.y_center - self.height / 2) * img_h
        x2 = (self.x_center + self.width / 2) * img_w
        y2 = (self.y_center + self.height / 2) * img_h
        return x1, y1, x2, y2


@dataclass
class LabelFileResult:
    """Parsed label file plus any validation issues found."""

    path: Path
    boxes: list[YoloBox]
    issues: list[str]

    @property
    def is_empty(self) -> bool:
        return not self.boxes and not self.issues


# Degenerate-box tolerance: boxes narrower/shorter than this normalised size
# cannot survive 640px rasterisation meaningfully.
MIN_BOX_SIZE = 1e-4


def parse_label_file(path: Path, valid_class_ids: set[int]) -> LabelFileResult:
    """Parse one YOLO label file, collecting boxes and validation issues."""
    boxes: list[YoloBox] = []
    issues: list[str] = []
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        return LabelFileResult(path, [], [f"unreadable label file: {exc}"])

    for lineno, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            issues.append(f"line {lineno}: expected 5 fields, got {len(parts)}")
            continue
        try:
            cls = int(parts[0])
            coords = [float(v) for v in parts[1:]]
        except ValueError:
            issues.append(f"line {lineno}: non-numeric value")
            continue
        x, y, w, h = coords
        if cls not in valid_class_ids:
            issues.append(f"line {lineno}: invalid class id {cls}")
            continue
        if not all(0.0 <= v <= 1.0 for v in (x, y, w, h)):
            issues.append(f"line {lineno}: coordinate outside [0,1]: {coords}")
            continue
        if w <= MIN_BOX_SIZE or h <= MIN_BOX_SIZE:
            issues.append(f"line {lineno}: degenerate box w={w:.6f} h={h:.6f}")
            continue
        if x - w / 2 < -0.001 or x + w / 2 > 1.001 or y - h / 2 < -0.001 or y + h / 2 > 1.001:
            issues.append(f"line {lineno}: box extends outside image bounds")
            continue
        boxes.append(YoloBox(cls, x, y, w, h))
    return LabelFileResult(path, boxes, issues)


def label_path_for_image(image_path: Path) -> Path:
    """Map .../split/images/name.jpg -> .../split/labels/name.txt."""
    return image_path.parent.parent / "labels" / (image_path.stem + ".txt")
