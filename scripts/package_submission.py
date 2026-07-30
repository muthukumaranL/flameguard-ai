"""Build the final submission archive.

Includes: all code, notebooks, configs, the final model weights, every output
(EDA, training logs, evaluation, benchmarking, error analysis, samples,
screenshots), the report (DOCX + PDF), the presentation, the Scrum package, the
test report, and dataset instructions.

Excludes: the virtual environment, caches, temporary files, intermediate
checkpoints, and the dataset itself (too large for a submission ZIP - a small
sample set plus exact download instructions are included instead, and the
omission is documented in the archive's DATASET_NOTE.md).

Usage:
    python scripts/package_submission.py [--group 07]
"""
from __future__ import annotations

import argparse
import sys
import zipfile
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import paths
from src.utils import file_size_mb, setup_logging

log = setup_logging("flameguard.package")

# Directories copied wholesale (relative to project root)
INCLUDE_DIRS = [
    "config", "src", "scripts", "tests", "notebooks",
    "outputs", "report", "presentation", "agile",
]
INCLUDE_FILES = [
    "app.py", "README.md", "requirements.txt", "requirements-dev.txt",
    "packages.txt", "environment.yml",
    "LICENSE_NOTES.md", "CHANGELOG.md", ".gitignore",
]
MODEL_FILES = ["models/final/best.pt", "models/final/model_metadata.yaml"]

EXCLUDE_PARTS = {".venv", ".pipcache", ".tmp", "__pycache__", ".pytest_cache",
                 ".git", ".ipynb_checkpoints"}
EXCLUDE_SUFFIXES = {".pyc", ".pyo"}
# Per-run checkpoints are large; keep only the winning weights (models/final).
EXCLUDE_NAMES = {"last.pt", "_resolved_data.yaml"}

DATASET_NOTE = """# Dataset - not included in this archive

The training dataset is **not bundled** with this submission because it is far
too large for a Blackboard upload (5,300 images, ~230 MB compressed).
Everything needed to reproduce it exactly is included instead.

## Source

- Roboflow Universe: **fire and smoke**, version 1
- Workspace/project: `fire-detector-cqdzi / fire-and-smoke-b5lli`
- Direct link: https://universe.roboflow.com/fire-detector-cqdzi/fire-and-smoke-b5lli/dataset/1
- Licence: **CC BY 4.0**
- Format: YOLOv8 (`train/`, `valid/`, `test/`, each with `images/` and `labels/`, plus `data.yaml`)
- Classes: `0 = Fire`, `1 = Smoke`

## How to reproduce our exact dataset

1. Download the v1 YOLOv8 export from the link above and extract it anywhere.
2. From the project root, run:

       python scripts/validate_dataset.py --source "<path to the extracted export>"

   This copies the export into `data/raw/`, audits every image and label, then
   rebuilds the **leakage-repaired** splits into
   `data/processed/fire_smoke_resplit/` using perceptual-hash grouping with a
   fixed seed (42). The result is bit-for-bit reproducible.

3. `python scripts/run_eda.py` regenerates the full EDA package.

## Why the splits were rebuilt

The published split leaks: mirrored/noise-augmented copies of the same source
images (and sequential video frames) appear in more than one of train/valid/test.
The exact counts we measured are in
`outputs/dataset_validation/resplit_report.json`, and the full audit trail of
every group decision is in `outputs/dataset_validation/group_audit.csv`.
**All results in the report use the repaired splits.**

## What IS included here

- `outputs/sample_inputs/` - 10 sample images from the test split.
- `outputs/sample_outputs/` - the model's predictions on them, with CSVs.
- `data.yaml` (below) - the class mapping used for training.
- Every validation and EDA artefact computed from the full dataset.

```yaml
{data_yaml}
```
"""


def _skip(path: Path) -> bool:
    if any(part in EXCLUDE_PARTS for part in path.parts):
        return True
    if path.suffix in EXCLUDE_SUFFIXES or path.name in EXCLUDE_NAMES:
        return True
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", default="##",
                        help="group number for the archive name (default placeholder)")
    args = parser.parse_args()

    root = paths.PROJECT_ROOT
    paths.SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = paths.SUBMISSION_DIR / f"Group{args.group}_FlameGuard_AI_Final_Submission.zip"

    missing = [m for m in MODEL_FILES if not (root / m).exists()]
    if missing:
        log.error("cannot package - missing model artefacts: %s", missing)
        return 1
    for required in ("report/FlameGuard_AI_Final_Report.docx",
                     "report/FlameGuard_AI_Final_Report.pdf",
                     "presentation/FlameGuard_AI_Presentation.pptx"):
        if not (root / required).exists():
            log.warning("expected deliverable missing: %s", required)

    files: list[Path] = []
    for rel in INCLUDE_FILES:
        p = root / rel
        if p.exists():
            files.append(p)
    for rel in MODEL_FILES:
        files.append(root / rel)
    for d in INCLUDE_DIRS:
        base = root / d
        if not base.exists():
            continue
        for p in base.rglob("*"):
            if p.is_file() and not _skip(p):
                files.append(p)

    data_yaml_text = ""
    if paths.PROCESSED_DATA_YAML.exists():
        data_yaml_text = paths.PROCESSED_DATA_YAML.read_text(encoding="utf-8").strip()

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for p in sorted(set(files)):
            zf.write(p, Path("FlameGuard_AI") / p.relative_to(root))
        zf.writestr("FlameGuard_AI/DATASET_NOTE.md",
                    DATASET_NOTE.format(data_yaml=data_yaml_text))
        zf.writestr("FlameGuard_AI/SUBMISSION_MANIFEST.txt", _manifest(args.group, files))

    size = file_size_mb(zip_path)
    log.info("submission archive: %s (%d files, %.1f MB)",
             paths.rel_to_root(zip_path), len(set(files)), size)

    with zipfile.ZipFile(zip_path) as zf:
        bad = zf.testzip()
    if bad:
        log.error("archive is corrupt at %s", bad)
        return 1
    log.info("archive verified OK")
    return 0


def _manifest(group: str, files: list[Path]) -> str:
    by_top: dict[str, int] = {}
    for p in files:
        rel = p.relative_to(paths.PROJECT_ROOT)
        top = rel.parts[0] if len(rel.parts) > 1 else "(root files)"
        by_top[top] = by_top.get(top, 0) + 1
    lines = [
        "FlameGuard AI - Final Submission",
        "AASD 4014 Deep Learning II",
        f"Group {group}",
        f"Packaged: {date.today().isoformat()}",
        "",
        "CONTENTS",
        "--------",
    ]
    lines += [f"  {k:<28} {v:>4} files" for k, v in sorted(by_top.items())]
    lines += [
        "",
        "KEY DELIVERABLES",
        "----------------",
        "  report/FlameGuard_AI_Final_Report.docx | .pdf   - the full report",
        "  presentation/FlameGuard_AI_Presentation.pptx    - 15-slide deck",
        "  presentation/speaker_notes.md, demo_script.md   - delivery material",
        "  models/final/best.pt                            - the trained model",
        "  outputs/sample_outputs/                         - 10 sample predictions",
        "  outputs/evaluation/                             - test-set metrics + curves",
        "  outputs/error_analysis/                         - failure galleries",
        "  agile/                                          - Scrum artefacts",
        "  outputs/test_report.txt                         - automated test results",
        "  DATASET_NOTE.md                                 - how to obtain the dataset",
        "",
        "TO RUN",
        "------",
        "  scripts/setup_environment.bat   (or bash scripts/setup_environment.sh)",
        "  streamlit run app.py",
        "",
        "NOT INCLUDED (deliberately)",
        "---------------------------",
        "  The dataset itself (too large) - see DATASET_NOTE.md for exact reproduction.",
        "  The virtual environment, caches, and intermediate training checkpoints.",
        "",
        "DISCLAIMER",
        "----------",
        "  FlameGuard AI is an educational computer-vision prototype. It is not a",
        "  certified fire-detection or emergency-response system.",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
