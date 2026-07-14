"""Exploratory data analysis for the FlameGuard AI dataset.

Builds a per-box and per-image feature table from the processed (re-split)
dataset, then renders the full chart package into outputs/eda/ and writes
data/dataset_summary.csv.  Every figure is produced from measured values.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from src.annotation_parser import label_path_for_image, parse_label_file
from src.paths import DATA_DIR, EDA_OUTPUT_DIR, SPLITS
from src.utils import setup_logging
from src.visualizations import draw_ground_truth, image_grid, mean_brightness

log = setup_logging("flameguard.eda")

plt.rcParams.update({
    "figure.dpi": 130, "savefig.bbox": "tight",
    "axes.grid": True, "grid.alpha": 0.3, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 10,
})
FIRE_COLOR, SMOKE_COLOR, NEUTRAL = "#e4572e", "#5a9fc8", "#6c757d"
SPLIT_COLORS = {"train": "#4c72b0", "valid": "#dd8452", "test": "#55a868"}

# COCO-equivalent size buckets on normalised areas for 640x640 inputs
SMALL_AREA = (32 / 640) ** 2
MEDIUM_AREA = (96 / 640) ** 2


def build_feature_tables(dataset_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (boxes_df, images_df) with derived numeric features."""
    with (dataset_dir / "data.yaml").open("r", encoding="utf-8") as fh:
        names = yaml.safe_load(fh)["names"]
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names)]
    valid_ids = set(range(len(names)))

    box_rows, img_rows = [], []
    for split in SPLITS:
        img_dir = dataset_dir / split / "images"
        for img_path in sorted(img_dir.iterdir()):
            parsed = parse_label_file(label_path_for_image(img_path), valid_ids)
            from PIL import Image as _Image

            with _Image.open(img_path) as im:
                w, h = im.size
            img_rows.append({
                "filename": img_path.name, "split": split,
                "width": w, "height": h, "aspect_ratio": w / h,
                "n_objects": len(parsed.boxes),
                "n_fire": sum(1 for b in parsed.boxes if b.class_id == 0),
                "n_smoke": sum(1 for b in parsed.boxes if b.class_id == 1),
                "brightness": mean_brightness(img_path),
            })
            for b in parsed.boxes:
                box_rows.append({
                    "filename": img_path.name, "split": split,
                    "class_id": b.class_id, "class_name": names[b.class_id],
                    "x_center": b.x_center, "y_center": b.y_center,
                    "box_w": b.width, "box_h": b.height, "box_area": b.area,
                    "box_aspect": b.aspect_ratio,
                    "img_w": w, "img_h": h,
                    "n_objects_in_image": len(parsed.boxes),
                })
    boxes = pd.DataFrame(box_rows)
    images = pd.DataFrame(img_rows)
    images["content"] = np.select(
        [(images.n_fire > 0) & (images.n_smoke > 0), images.n_fire > 0, images.n_smoke > 0],
        ["both", "fire_only", "smoke_only"], default="background")
    boxes["size_category"] = np.select(
        [boxes.box_area < SMALL_AREA, boxes.box_area < MEDIUM_AREA],
        ["small", "medium"], default="large")
    log.info("Feature tables: %d boxes, %d images", len(boxes), len(images))
    return boxes, images


def _save(fig: plt.Figure, name: str) -> None:
    out = EDA_OUTPUT_DIR / name
    fig.savefig(out)
    plt.close(fig)
    log.info("wrote %s", name)


def chart_dataset_composition(images: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    counts = images.groupby("split").size().reindex(list(SPLITS))
    axes[0].bar(counts.index, counts.values, color=[SPLIT_COLORS[s] for s in counts.index])
    for i, v in enumerate(counts.values):
        axes[0].text(i, v, f"{v:,}\n({v / counts.sum():.0%})", ha="center", va="bottom")
    axes[0].set_title("Images per split (re-split dataset)")
    axes[0].set_ylabel("images")

    comp = images.groupby(["split", "content"]).size().unstack(fill_value=0).reindex(list(SPLITS))
    comp = comp[["fire_only", "smoke_only", "both", "background"]]
    bottom = np.zeros(len(comp))
    palette = {"fire_only": FIRE_COLOR, "smoke_only": SMOKE_COLOR,
               "both": "#9d4edd", "background": NEUTRAL}
    for col in comp.columns:
        axes[1].bar(comp.index, comp[col], bottom=bottom, label=col, color=palette[col])
        bottom += comp[col].values
    axes[1].set_title("Image content by split")
    axes[1].legend(fontsize=8)
    _save(fig, "01_dataset_composition.png")


def chart_class_balance(boxes: pd.DataFrame, images: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ann = boxes.groupby("class_name").size()
    img_counts = {
        "Fire": int((images.n_fire > 0).sum()),
        "Smoke": int((images.n_smoke > 0).sum()),
    }
    axes[0].bar(img_counts.keys(), img_counts.values(), color=[FIRE_COLOR, SMOKE_COLOR])
    for i, (k, v) in enumerate(img_counts.items()):
        axes[0].text(i, v, f"{v:,}", ha="center", va="bottom")
    axes[0].set_title("Images containing each class")
    axes[1].bar(ann.index, ann.values, color=[FIRE_COLOR, SMOKE_COLOR])
    for i, v in enumerate(ann.values):
        axes[1].text(i, v, f"{v:,}", ha="center", va="bottom")
    ratio = ann.max() / ann.min()
    axes[1].set_title(f"Annotations per class (imbalance {ratio:.2f}:1)")
    _save(fig, "02_class_balance.png")


def chart_box_geometry(boxes: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for cname, color in (("Fire", FIRE_COLOR), ("Smoke", SMOKE_COLOR)):
        sub = boxes[boxes.class_name.str.lower() == cname.lower()]
        axes[0, 0].hist(sub.box_w, bins=40, alpha=0.6, label=cname, color=color)
        axes[0, 1].hist(sub.box_h, bins=40, alpha=0.6, label=cname, color=color)
        axes[1, 0].hist(sub.box_area, bins=np.logspace(-4.2, 0, 40), alpha=0.6,
                        label=cname, color=color)
        axes[1, 1].scatter(sub.box_w, sub.box_h, s=4, alpha=0.25, label=cname, color=color)
    axes[0, 0].set_title("Box width (normalised)")
    axes[0, 1].set_title("Box height (normalised)")
    axes[1, 0].set_xscale("log")
    axes[1, 0].set_title("Box area (normalised, log scale)")
    axes[1, 1].set_title("Box width vs height")
    axes[1, 1].set_xlabel("width"), axes[1, 1].set_ylabel("height")
    for ax in axes.flat:
        ax.legend(fontsize=8)
    _save(fig, "03_box_geometry.png")


def chart_center_heatmap(boxes: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, cname, cmap in ((axes[0], "Fire", "Reds"), (axes[1], "Smoke", "Blues")):
        sub = boxes[boxes.class_name.str.lower() == cname.lower()]
        hm, _, _ = np.histogram2d(sub.x_center, sub.y_center, bins=40,
                                  range=[[0, 1], [0, 1]])
        ax.imshow(hm.T, origin="upper", cmap=cmap, extent=[0, 1, 1, 0], aspect="auto")
        ax.set_title(f"{cname} box-centre density")
        ax.grid(False)
    _save(fig, "04_center_heatmap.png")


def chart_size_categories(boxes: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    tab = boxes.groupby(["class_name", "size_category"]).size().unstack(fill_value=0)
    tab = tab[["small", "medium", "large"]]
    tab.plot.bar(ax=ax, color=["#c1121f", "#fb8500", "#2a9d8f"], rot=0)
    ax.set_title("Object size categories (COCO-equivalent buckets)")
    ax.set_ylabel("boxes")
    _save(fig, "05_size_categories.png")


def chart_objects_per_image(images: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    m = images.n_objects.max()
    axes[0].hist(images.n_objects, bins=np.arange(-0.5, m + 1.5), color="#4c72b0")
    axes[0].set_title("Objects per image (annotation density)")
    axes[0].set_xlabel("objects")
    for split in SPLITS:
        axes[1].hist(images.loc[images.split == split, "n_objects"],
                     bins=np.arange(-0.5, m + 1.5), histtype="step",
                     label=split, color=SPLIT_COLORS[split], density=True)
    axes[1].set_title("Annotation density by split (normalised)")
    axes[1].legend()
    _save(fig, "06_objects_per_image.png")


def chart_correlation_matrix(boxes: pd.DataFrame) -> None:
    feats = boxes[["img_w", "img_h", "box_w", "box_h", "box_area", "box_aspect",
                   "x_center", "y_center", "n_objects_in_image", "class_id"]].copy()
    feats["img_aspect"] = feats.img_w / feats.img_h
    corr = feats.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(8.5, 7))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)), corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr)), corr.columns)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center",
                    fontsize=7,
                    color="white" if abs(corr.iloc[i, j]) > 0.55 else "black")
    ax.set_title("Correlation matrix of derived numeric features\n"
                 "(associations only - no causal interpretation implied)")
    ax.grid(False)
    fig.colorbar(im, shrink=0.8)
    _save(fig, "07_correlation_matrix.png")


def chart_brightness(images: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(images.brightness, bins=50, color="#6c757d")
    p10 = images.brightness.quantile(0.10)
    ax.axvline(p10, color="#c1121f", ls="--", label=f"10th pct = {p10:.0f} (low-light)")
    ax.set_title("Mean image brightness distribution")
    ax.set_xlabel("mean grayscale intensity (0-255)")
    ax.legend()
    _save(fig, "08_brightness_distribution.png")


def _grid_from(dataset_dir: Path, images: pd.DataFrame, mask: pd.Series,
               class_names: dict[int, str], name: str, title_col: str = "filename",
               n: int = 9, seed: int = 42) -> None:
    subset = images[mask]
    if subset.empty:
        log.warning("grid %s: no matching images", name)
        return
    picks = subset.sample(min(n, len(subset)), random_state=seed)
    tiles, titles = [], []
    valid_ids = set(class_names)
    for _, row in picks.iterrows():
        img_path = dataset_dir / row.split / "images" / row.filename
        parsed = parse_label_file(label_path_for_image(img_path), valid_ids)
        tiles.append(draw_ground_truth(img_path, parsed.boxes, class_names))
        titles.append(str(row[title_col]))
    image_grid(tiles, cols=3, titles=titles).save(EDA_OUTPUT_DIR / name)
    log.info("wrote %s (%d tiles)", name, len(tiles))


def sample_grids(dataset_dir: Path, boxes: pd.DataFrame, images: pd.DataFrame,
                 class_names: dict[int, str]) -> None:
    _grid_from(dataset_dir, images, images.content == "fire_only",
               class_names, "10_grid_fire_only.png")
    _grid_from(dataset_dir, images, images.content == "smoke_only",
               class_names, "11_grid_smoke_only.png")
    _grid_from(dataset_dir, images, images.content == "both",
               class_names, "12_grid_both_classes.png")
    small_imgs = set(boxes.loc[boxes.size_category == "small", "filename"])
    _grid_from(dataset_dir, images, images.filename.isin(small_imgs),
               class_names, "13_grid_small_objects.png")
    lowlight = images.brightness < images.brightness.quantile(0.10)
    _grid_from(dataset_dir, images, lowlight & (images.n_objects > 0),
               class_names, "14_grid_low_light.png")
    # "difficult": low light OR only-small-boxes OR crowded scenes
    only_small = images.filename.isin(
        set(boxes.groupby("filename").size().index) -
        set(boxes.loc[boxes.size_category != "small", "filename"])
    ) & (images.n_objects > 0)
    difficult = lowlight & (images.n_objects > 0) | only_small | (images.n_objects >= 6)
    _grid_from(dataset_dir, images, difficult, class_names, "15_grid_difficult.png")
    _grid_from(dataset_dir, images, images.n_objects > 0,
               class_names, "09_grid_annotated_samples.png", n=9)


def outlier_report(boxes: pd.DataFrame, images: pd.DataFrame) -> None:
    """CSV of statistical outliers for manual annotation review."""
    rows = []
    for _, b in boxes[boxes.box_area < 0.0002].iterrows():
        rows.append({"filename": b.filename, "split": b.split, "class": b.class_name,
                     "issue": f"extremely small box (area={b.box_area:.5f})",
                     "severity": "review", "recommended_action": "verify annotation",
                     "action_taken": "kept - plausible distant object"})
    for _, b in boxes[boxes.box_area > 0.9].iterrows():
        rows.append({"filename": b.filename, "split": b.split, "class": b.class_name,
                     "issue": f"box covers {b.box_area:.0%} of image",
                     "severity": "review", "recommended_action": "verify annotation",
                     "action_taken": "kept - plausible full-frame smoke/fire"})
    for _, im in images[images.n_objects >= 8].iterrows():
        rows.append({"filename": im.filename, "split": im.split, "class": "-",
                     "issue": f"crowded scene ({im.n_objects} boxes)",
                     "severity": "info", "recommended_action": "spot-check boxes",
                     "action_taken": "kept"})
    df = pd.DataFrame(rows)
    df.to_csv(EDA_OUTPUT_DIR / "outlier_report.csv", index=False)
    df.to_csv(EDA_OUTPUT_DIR / "annotation_quality_review.csv", index=False)
    log.info("outlier/annotation review rows: %d", len(rows))


def write_summary(boxes: pd.DataFrame, images: pd.DataFrame) -> None:
    summary = {
        "total_images": len(images),
        "images_per_split": images.groupby("split").size().to_dict(),
        "total_annotations": len(boxes),
        "annotations_per_split": boxes.groupby("split").size().to_dict(),
        "annotations_per_class": boxes.groupby("class_name").size().to_dict(),
        "images_with_fire": int((images.n_fire > 0).sum()),
        "images_with_smoke": int((images.n_smoke > 0).sum()),
        "images_with_both": int((images.content == "both").sum()),
        "background_images": int((images.content == "background").sum()),
        "mean_objects_per_image": float(images.n_objects.mean()),
        "class_imbalance_ratio": float(
            boxes.groupby("class_id").size().max() / boxes.groupby("class_id").size().min()),
        "small_boxes_pct": float((boxes.size_category == "small").mean() * 100),
        "medium_boxes_pct": float((boxes.size_category == "medium").mean() * 100),
        "large_boxes_pct": float((boxes.size_category == "large").mean() * 100),
    }
    with (EDA_OUTPUT_DIR / "eda_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    images.to_csv(DATA_DIR / "dataset_summary.csv", index=False)
    boxes.to_csv(EDA_OUTPUT_DIR / "box_features.csv", index=False)
    log.info("summary: %s", summary)


def run_eda(dataset_dir: Path) -> None:
    EDA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    boxes, images = build_feature_tables(dataset_dir)
    with (dataset_dir / "data.yaml").open("r", encoding="utf-8") as fh:
        names_list = yaml.safe_load(fh)["names"]
    if isinstance(names_list, dict):
        names_list = [names_list[k] for k in sorted(names_list)]
    class_names = dict(enumerate(names_list))

    chart_dataset_composition(images)
    chart_class_balance(boxes, images)
    chart_box_geometry(boxes)
    chart_center_heatmap(boxes)
    chart_size_categories(boxes)
    chart_objects_per_image(images)
    chart_correlation_matrix(boxes)
    chart_brightness(images)
    sample_grids(dataset_dir, boxes, images, class_names)
    outlier_report(boxes, images)
    write_summary(boxes, images)
    log.info("EDA complete -> %s", EDA_OUTPUT_DIR)
