"""Central path configuration for FlameGuard AI.

Every module resolves file locations through this file so the project stays
relocatable (no absolute user paths anywhere else in the codebase).
"""
from __future__ import annotations

from pathlib import Path

# <project root>/src/paths.py -> parents[1] == project root
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

CONFIG_DIR = PROJECT_ROOT / "config"
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SAMPLES_DIR = DATA_DIR / "samples"
DATA_REPORTS_DIR = DATA_DIR / "reports"

MODELS_DIR = PROJECT_ROOT / "models"
BASELINE_MODEL_DIR = MODELS_DIR / "baseline"
COMPARISON_MODEL_DIR = MODELS_DIR / "comparisons"
TUNED_MODEL_DIR = MODELS_DIR / "tuned"
FINAL_MODEL_DIR = MODELS_DIR / "final"
FINAL_MODEL_PATH = FINAL_MODEL_DIR / "best.pt"
FINAL_MODEL_METADATA_PATH = FINAL_MODEL_DIR / "model_metadata.yaml"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
VALIDATION_OUTPUT_DIR = OUTPUTS_DIR / "dataset_validation"
EDA_OUTPUT_DIR = OUTPUTS_DIR / "eda"
TRAINING_OUTPUT_DIR = OUTPUTS_DIR / "training"
EVALUATION_OUTPUT_DIR = OUTPUTS_DIR / "evaluation"
BENCHMARK_OUTPUT_DIR = OUTPUTS_DIR / "benchmarking"
ERROR_ANALYSIS_OUTPUT_DIR = OUTPUTS_DIR / "error_analysis"
SAMPLE_INPUTS_DIR = OUTPUTS_DIR / "sample_inputs"
SAMPLE_OUTPUTS_DIR = OUTPUTS_DIR / "sample_outputs"
IMAGE_PREDICTIONS_DIR = OUTPUTS_DIR / "image_predictions"
VIDEO_PREDICTIONS_DIR = OUTPUTS_DIR / "video_predictions"
SCREENSHOTS_DIR = OUTPUTS_DIR / "application_screenshots"

REPORT_DIR = PROJECT_ROOT / "report"
REPORT_FIGURES_DIR = REPORT_DIR / "figures"
PRESENTATION_DIR = PROJECT_ROOT / "presentation"
AGILE_DIR = PROJECT_ROOT / "agile"
SUBMISSION_DIR = PROJECT_ROOT / "submission"

# Dataset (populated by scripts/validate_dataset.py, consumed everywhere else)
RAW_DATASET_DIR = RAW_DATA_DIR / "fire_and_smoke_v1"
PROCESSED_DATASET_DIR = PROCESSED_DATA_DIR / "fire_smoke_resplit"
PROCESSED_DATA_YAML = PROCESSED_DATASET_DIR / "data.yaml"

SPLITS = ("train", "valid", "test")


def ensure_output_dirs() -> None:
    """Create every output directory the pipeline writes to."""
    for d in (
        RAW_DATA_DIR, PROCESSED_DATA_DIR, SAMPLES_DIR, DATA_REPORTS_DIR,
        BASELINE_MODEL_DIR, COMPARISON_MODEL_DIR, TUNED_MODEL_DIR,
        FINAL_MODEL_DIR, VALIDATION_OUTPUT_DIR, EDA_OUTPUT_DIR,
        TRAINING_OUTPUT_DIR, EVALUATION_OUTPUT_DIR, BENCHMARK_OUTPUT_DIR,
        ERROR_ANALYSIS_OUTPUT_DIR, SAMPLE_INPUTS_DIR, SAMPLE_OUTPUTS_DIR,
        IMAGE_PREDICTIONS_DIR, VIDEO_PREDICTIONS_DIR, SCREENSHOTS_DIR,
        REPORT_FIGURES_DIR, PRESENTATION_DIR, AGILE_DIR, SUBMISSION_DIR,
    ):
        d.mkdir(parents=True, exist_ok=True)


def rel_to_root(path: Path) -> str:
    """Render a path relative to the project root for logs and reports."""
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)
