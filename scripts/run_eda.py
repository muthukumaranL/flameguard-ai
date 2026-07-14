"""Stage 1b runner: full EDA package + augmentation preview figure.

Usage:
    python scripts/run_eda.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image, ImageEnhance

from src import paths
from src.eda import run_eda
from src.utils import setup_logging
from src.visualizations import image_grid

log = setup_logging("flameguard.eda-runner")


def augmentation_preview(dataset_dir: Path, out_path: Path, seed: int = 42) -> None:
    """Show what the on-the-fly training augmentations do to one image.

    Vertical flip is deliberately absent: flames and smoke plumes rise, so a
    vertically inverted scene is physically unrealistic and was disabled in
    training (flipud=0).
    """
    rng = np.random.default_rng(seed)
    img_dir = dataset_dir / "train" / "images"
    candidates = sorted(img_dir.iterdir())
    src = Image.open(candidates[int(rng.integers(len(candidates)))]).convert("RGB")

    variants: list[Image.Image] = [src]
    titles = ["original"]

    variants.append(src.transpose(Image.FLIP_LEFT_RIGHT)); titles.append("horizontal flip (fliplr=0.5)")
    variants.append(ImageEnhance.Brightness(src).enhance(1.4)); titles.append("brightness +40% (hsv_v)")
    variants.append(ImageEnhance.Contrast(src).enhance(1.5)); titles.append("contrast +50%")
    w, h = src.size
    crop = src.crop((int(w * 0.15), int(h * 0.15), int(w * 0.85), int(h * 0.85))).resize((w, h))
    variants.append(crop); titles.append("scale/zoom (scale=0.5)")
    variants.append(src.rotate(8, expand=False, fillcolor=(114, 114, 114))); titles.append("rotation ±10° (degrees)")

    # simple 2x2 mosaic composition from four random train images
    tiles = [Image.open(candidates[int(i)]).convert("RGB").resize((w // 2, h // 2))
             for i in rng.integers(0, len(candidates), 4)]
    mosaic = Image.new("RGB", (w, h))
    mosaic.paste(tiles[0], (0, 0)); mosaic.paste(tiles[1], (w // 2, 0))
    mosaic.paste(tiles[2], (0, h // 2)); mosaic.paste(tiles[3], (w // 2, h // 2))
    variants.append(mosaic); titles.append("mosaic (4-image composite)")

    hsv = np.array(src.convert("HSV"), dtype=np.int16)
    hsv[..., 0] = (hsv[..., 0] + 8) % 256
    hsv[..., 1] = np.clip(hsv[..., 1] * 1.3, 0, 255)
    variants.append(Image.fromarray(hsv.astype(np.uint8), "HSV").convert("RGB"))
    titles.append("HSV shift (hsv_h/hsv_s)")

    image_grid(variants, cols=4, titles=titles).save(out_path)
    log.info("augmentation preview -> %s", out_path)


def main() -> int:
    paths.ensure_output_dirs()
    dataset = paths.PROCESSED_DATASET_DIR
    if not (dataset / "data.yaml").exists():
        log.error("Processed dataset missing - run scripts/validate_dataset.py first")
        return 1
    run_eda(dataset)
    augmentation_preview(dataset, paths.EDA_OUTPUT_DIR / "16_augmentation_preview.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
