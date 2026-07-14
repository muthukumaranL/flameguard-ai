"""Drawing helpers shared by EDA, error analysis and the report figures."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.annotation_parser import YoloBox

# RGB colours (PIL) per class id - keep in sync with config/class_names.yaml
CLASS_COLORS_RGB: dict[int, tuple[int, int, int]] = {
    0: (255, 64, 40),    # Fire - red/orange
    1: (90, 160, 200),   # Smoke - steel blue
}


def _font(size: int = 14) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def draw_ground_truth(image_path: Path, boxes: list[YoloBox],
                      class_names: dict[int, str], width: int = 3) -> Image.Image:
    """Return a copy of the image with ground-truth boxes drawn on it."""
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    font = _font()
    for box in boxes:
        x1, y1, x2, y2 = box.to_xyxy(img.width, img.height)
        color = CLASS_COLORS_RGB.get(box.class_id, (255, 255, 0))
        draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
        label = class_names.get(box.class_id, str(box.class_id))
        tb = draw.textbbox((x1, y1), label, font=font)
        draw.rectangle([tb[0] - 2, tb[1] - 2, tb[2] + 2, tb[3] + 2], fill=color)
        draw.text((x1, y1), label, fill=(255, 255, 255), font=font)
    return img


def image_grid(images: list[Image.Image], cols: int, cell: int = 320,
               titles: list[str] | None = None) -> Image.Image:
    """Compose thumbnails into a labelled grid image."""
    if not images:
        raise ValueError("image_grid called with no images")
    rows = (len(images) + cols - 1) // cols
    title_h = 22 if titles else 0
    canvas = Image.new("RGB", (cols * cell, rows * (cell + title_h)), (250, 250, 250))
    draw = ImageDraw.Draw(canvas)
    font = _font(12)
    for i, img in enumerate(images):
        thumb = img.copy()
        thumb.thumbnail((cell, cell))
        r, c = divmod(i, cols)
        x = c * cell + (cell - thumb.width) // 2
        y = r * (cell + title_h) + title_h + (cell - thumb.height) // 2
        canvas.paste(thumb, (x, y))
        if titles and i < len(titles):
            draw.text((c * cell + 6, r * (cell + title_h) + 4),
                      titles[i][:46], fill=(30, 30, 30), font=font)
    return canvas


def mean_brightness(image_path: Path) -> float:
    """Mean grayscale intensity in [0, 255] (downsampled for speed)."""
    with Image.open(image_path) as im:
        g = im.convert("L")
        g.thumbnail((64, 64))
        return float(np.asarray(g, dtype=np.float32).mean())
