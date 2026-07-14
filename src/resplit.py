"""Leakage-aware dataset re-splitting.

The upstream Roboflow v1 export contains baked-in augmented copies
(``Mirror*``, ``Noise*`` filename prefixes) and near-duplicate frame sequences
(``fire1-NNN-``), and those copies were distributed across train/valid/test.
That leaks training content into the evaluation splits, inflating metrics.

Repair strategy (fully deterministic, seed-controlled):
  1. Canonical stem: strip the Roboflow ``.rf.<hash>`` suffix and the
     augmentation prefixes -> images sharing a stem are variants of one source.
  2. Perceptual hashing (pHash) over every image; images whose hashes differ by
     <= HAMMING_THRESHOLD bits are treated as near-duplicates.
  3. Union-find merges stem groups and pHash matches into source groups.
  4. Groups (not images) are stratified by content (fire/smoke/both/background)
     and greedily assigned to train/valid/test at ~70/20/10, so every variant
     of a source image lands in exactly one split.
  5. Images + labels are copied into data/processed/fire_smoke_resplit and a
     fresh data.yaml is written. A JSON/CSV audit trail records every decision.
"""
from __future__ import annotations

import json
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from src.annotation_parser import label_path_for_image, parse_label_file
from src.utils import setup_logging

log = setup_logging("flameguard.resplit")

RF_SUFFIX = re.compile(r"_(jpg|jpeg|png|bmp|webp)\.rf\.[0-9a-f]+$", re.IGNORECASE)
AUG_PREFIXES = re.compile(r"^(Mirror|Noise|FireDetectionImage_|BowFire_)+", re.IGNORECASE)
HAMMING_THRESHOLD = 6          # pHash bits; <=6 of 64 is a conservative near-dup call
SPLIT_FRACTIONS = {"train": 0.70, "valid": 0.20, "test": 0.10}
STRATA = ("both", "fire_only", "smoke_only", "background")


def canonical_stem(filename: str) -> str:
    """Reduce a filename to its source-image identity."""
    stem = Path(filename).stem
    stem = RF_SUFFIX.sub("", stem)
    prev = None
    while prev != stem:                       # strip stacked prefixes (Mirror+Noise)
        prev = stem
        stem = AUG_PREFIXES.sub("", stem)
    return stem.lower()


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, i: int) -> int:
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


@dataclass
class ImageRecord:
    path: Path
    split: str                 # original split (for the before/after report)
    stem: str
    classes: frozenset[int]
    phash: int


def _phash64(path: Path) -> int:
    import imagehash

    with Image.open(path) as im:
        return int(str(imagehash.phash(im)), 16)


def _content_stratum(classes: set[int]) -> str:
    if classes == {0, 1}:
        return "both"
    if classes == {0}:
        return "fire_only"
    if classes == {1}:
        return "smoke_only"
    return "background"


def _collect_records(dataset_dir: Path, valid_ids: set[int]) -> list[ImageRecord]:
    records: list[ImageRecord] = []
    for split in ("train", "valid", "test"):
        img_dir = dataset_dir / split / "images"
        for img in sorted(img_dir.iterdir()):
            if not img.is_file():
                continue
            parsed = parse_label_file(label_path_for_image(img), valid_ids)
            classes = frozenset(b.class_id for b in parsed.boxes)
            records.append(ImageRecord(img, split, canonical_stem(img.name), classes, 0))
    log.info("Collected %d image records", len(records))
    return records


def _compute_phashes(records: list[ImageRecord]) -> None:
    for i, rec in enumerate(records):
        rec.phash = _phash64(rec.path)
        if (i + 1) % 1000 == 0:
            log.info("pHash %d/%d", i + 1, len(records))


def _phash_pairs(records: list[ImageRecord], threshold: int) -> list[tuple[int, int]]:
    """All index pairs whose pHash hamming distance is <= threshold (chunked numpy)."""
    hashes = np.array([r.phash for r in records], dtype=np.uint64)
    n = len(hashes)
    pairs: list[tuple[int, int]] = []
    chunk = 512
    for start in range(0, n, chunk):
        block = hashes[start:start + chunk, None] ^ hashes[None, :]   # (c, n) xor
        dist = np.bitwise_count(block)
        ii, jj = np.nonzero(dist <= threshold)
        for a, b in zip(ii + start, jj):
            if a < b:
                pairs.append((int(a), int(b)))
    return pairs


def build_groups(records: list[ImageRecord]) -> list[list[int]]:
    """Union stem-equality and pHash-similarity into source groups."""
    uf = _UnionFind(len(records))
    by_stem: dict[str, list[int]] = defaultdict(list)
    for idx, rec in enumerate(records):
        by_stem[rec.stem].append(idx)
    for idxs in by_stem.values():
        for other in idxs[1:]:
            uf.union(idxs[0], other)

    pairs = _phash_pairs(records, HAMMING_THRESHOLD)
    log.info("pHash near-duplicate pairs (<=%d bits): %d", HAMMING_THRESHOLD, len(pairs))
    for a, b in pairs:
        uf.union(a, b)

    groups: dict[int, list[int]] = defaultdict(list)
    for idx in range(len(records)):
        groups[uf.find(idx)].append(idx)
    result = sorted(groups.values(), key=len, reverse=True)
    log.info("Source groups: %d (largest=%d images)", len(result), len(result[0]))
    return result


def _leakage_table(records: list[ImageRecord], groups: list[list[int]],
                   split_of: dict[int, str]) -> dict[str, int]:
    """Count groups whose members span more than one split."""
    spanning = Counter()
    for grp in groups:
        splits = {split_of[i] for i in grp}
        if len(splits) > 1:
            spanning["groups_spanning_splits"] += 1
            spanning["images_in_spanning_groups"] += len(grp)
    return dict(spanning) or {"groups_spanning_splits": 0, "images_in_spanning_groups": 0}


def assign_splits(records: list[ImageRecord], groups: list[list[int]],
                  seed: int = 42) -> dict[int, str]:
    """Greedy stratified assignment of whole groups to splits (~70/20/10)."""
    rng = np.random.default_rng(seed)
    assignment: dict[int, str] = {}

    strata_groups: dict[str, list[list[int]]] = defaultdict(list)
    for grp in groups:
        classes: set[int] = set()
        for i in grp:
            classes |= records[i].classes
        strata_groups[_content_stratum(classes)].append(grp)

    for stratum in STRATA:
        grps = strata_groups.get(stratum, [])
        order = rng.permutation(len(grps))
        grps = [grps[i] for i in order]
        grps.sort(key=len, reverse=True)      # big groups placed first, stable overall
        total = sum(len(g) for g in grps)
        targets = {s: total * f for s, f in SPLIT_FRACTIONS.items()}
        filled = {s: 0 for s in SPLIT_FRACTIONS}
        for grp in grps:
            deficit = {s: (targets[s] - filled[s]) / max(targets[s], 1e-9)
                       for s in SPLIT_FRACTIONS}
            dest = max(deficit, key=deficit.get)
            for i in grp:
                assignment[i] = dest
            filled[dest] += len(grp)
        log.info("stratum %-11s total=%5d -> %s", stratum, total,
                 {s: filled[s] for s in SPLIT_FRACTIONS})
    return assignment


def materialise(records: list[ImageRecord], assignment: dict[int, str],
                out_dir: Path, class_names: list[str]) -> dict[str, dict[str, int]]:
    """Copy images/labels into the new split layout and write data.yaml."""
    if out_dir.exists():
        shutil.rmtree(out_dir)
    counts: dict[str, dict[str, int]] = {}
    for split in ("train", "valid", "test"):
        (out_dir / split / "images").mkdir(parents=True)
        (out_dir / split / "labels").mkdir(parents=True)
    for idx, rec in enumerate(records):
        split = assignment[idx]
        shutil.copy2(rec.path, out_dir / split / "images" / rec.path.name)
        lbl = label_path_for_image(rec.path)
        if lbl.exists():
            shutil.copy2(lbl, out_dir / split / "labels" / lbl.name)
        c = counts.setdefault(split, {"images": 0, "fire_imgs": 0, "smoke_imgs": 0,
                                      "both": 0, "background": 0})
        c["images"] += 1
        if 0 in rec.classes:
            c["fire_imgs"] += 1
        if 1 in rec.classes:
            c["smoke_imgs"] += 1
        if rec.classes == frozenset({0, 1}):
            c["both"] += 1
        if not rec.classes:
            c["background"] += 1

    data_yaml = {
        "path": ".",
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": len(class_names),
        "names": class_names,
    }
    with (out_dir / "data.yaml").open("w", encoding="utf-8") as fh:
        yaml.safe_dump(data_yaml, fh, sort_keys=False)
    return counts


def resplit_dataset(dataset_dir: Path, out_dir: Path, report_dir: Path,
                    seed: int = 42) -> dict:
    """Full pipeline: group -> assign -> materialise -> audit. Returns report."""
    report_dir.mkdir(parents=True, exist_ok=True)
    with (dataset_dir / "data.yaml").open("r", encoding="utf-8") as fh:
        class_names = yaml.safe_load(fh)["names"]
    if isinstance(class_names, dict):
        class_names = [class_names[k] for k in sorted(class_names)]

    records = _collect_records(dataset_dir, set(range(len(class_names))))
    log.info("Computing perceptual hashes...")
    _compute_phashes(records)
    groups = build_groups(records)

    before = _leakage_table(records, groups, {i: r.split for i, r in enumerate(records)})
    assignment = assign_splits(records, groups, seed)
    after = _leakage_table(records, groups, assignment)
    assert after["groups_spanning_splits"] == 0, "re-split failed to isolate groups"

    counts = materialise(records, assignment, out_dir, class_names)

    group_sizes = sorted((len(g) for g in groups), reverse=True)
    report = {
        "seed": seed,
        "hamming_threshold": HAMMING_THRESHOLD,
        "split_fractions": SPLIT_FRACTIONS,
        "images": len(records),
        "source_groups": len(groups),
        "largest_group_sizes": group_sizes[:20],
        "leakage_before": before,
        "leakage_after": after,
        "new_split_counts": counts,
    }
    with (report_dir / "resplit_report.json").open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    with (report_dir / "group_audit.csv").open("w", encoding="utf-8", newline="") as fh:
        import csv as _csv

        writer = _csv.writer(fh)
        writer.writerow(["group_id", "size", "new_split", "original_splits", "members"])
        for gid, grp in enumerate(groups):
            writer.writerow([
                gid, len(grp), assignment[grp[0]],
                ";".join(sorted({records[i].split for i in grp})),
                ";".join(records[i].path.name for i in grp[:50]),
            ])
    log.info("Re-split complete. before=%s after=%s", before, after)
    return report
