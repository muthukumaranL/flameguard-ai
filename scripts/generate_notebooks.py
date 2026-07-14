"""Generate the analysis notebooks.

The notebooks are thin, readable views over the same `src/` modules the scripts
use - they never re-implement logic, so a notebook cannot drift away from the
pipeline that produced the reported numbers.

Usage:
    python scripts/generate_notebooks.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import nbformat as nbf

from src.paths import PROJECT_ROOT
from src.utils import setup_logging

log = setup_logging("flameguard.notebooks")

NB_DIR = PROJECT_ROOT / "notebooks"

BOOT = """\
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent))   # import the project's src/ package
import pandas as pd
from IPython.display import Image, display
from src import paths
"""

NOTEBOOKS: dict[str, list[tuple[str, str]]] = {
    "01_dataset_validation.ipynb": [
        ("md", "# 1 · Dataset Validation\n\n"
               "Audits the raw Roboflow export and the leakage-repaired split.\n\n"
               "Run `python scripts/validate_dataset.py` first - this notebook reads its "
               "saved artefacts rather than recomputing them, so what you see here is "
               "exactly what the report cites."),
        ("code", BOOT),
        ("md", "## Raw dataset audit"),
        ("code", "raw = json.loads((paths.VALIDATION_OUTPUT_DIR / 'raw' / "
                 "'validation_report.json').read_text())\n"
                 "print('classes:', raw['classes'])\n"
                 "print('images per class:', raw['images_per_class_total'])\n"
                 "print('>=200 images per class:', raw['min_200_images_per_class'])\n"
                 "pd.DataFrame(raw['splits']).T[['images','label_files','background_images',"
                 "'images_with_both_classes','label_issue_count']]"),
        ("md", "## Leakage: what we found, and the repair\n\n"
               "Mirrored/noise-augmented copies and sequential frames of the same source "
               "image were scattered across train/valid/test. Groups - not images - were "
               "then reassigned to splits."),
        ("code", "rs = json.loads((paths.VALIDATION_OUTPUT_DIR / 'resplit_report.json').read_text())\n"
                 "print(f\"images                : {rs['images']:,}\")\n"
                 "print(f\"source groups         : {rs['source_groups']:,}\")\n"
                 "print(f\"BEFORE - groups spanning splits: {rs['leakage_before']['groups_spanning_splits']}\")\n"
                 "print(f\"BEFORE - images in those groups: {rs['leakage_before']['images_in_spanning_groups']:,}\")\n"
                 "print(f\"AFTER  - groups spanning splits: {rs['leakage_after']['groups_spanning_splits']}\")\n"
                 "pd.DataFrame(rs['new_split_counts']).T"),
        ("md", "## Audit of the repaired split"),
        ("code", "proc = json.loads((paths.VALIDATION_OUTPUT_DIR / 'processed' / "
                 "'validation_report.json').read_text())\n"
                 "print('cross-split exact duplicates:', proc['exact_duplicates_cross_split'])\n"
                 "print('images per class:', proc['images_per_class_total'])\n"
                 "pd.DataFrame(proc['splits']).T[['images','background_images','images_with_both_classes']]"),
    ],
    "02_exploratory_data_analysis.ipynb": [
        ("md", "# 2 · Exploratory Data Analysis\n\n"
               "Every figure below was produced by `python scripts/run_eda.py` from the "
               "**repaired** split."),
        ("code", BOOT),
        ("code", "summary = json.loads((paths.EDA_OUTPUT_DIR / 'eda_summary.json').read_text())\n"
                 "summary"),
        ("md", "## Composition and class balance"),
        ("code", "for f in ['01_dataset_composition.png', '02_class_balance.png']:\n"
                 "    display(Image(str(paths.EDA_OUTPUT_DIR / f)))"),
        ("md", "## Object geometry - why we kept 640px inputs\n\n"
               "A large fraction of objects are small. Downscaling for speed would push "
               "distant flames below the detector's smallest stride."),
        ("code", "for f in ['03_box_geometry.png', '05_size_categories.png', "
                 "'04_center_heatmap.png']:\n"
                 "    display(Image(str(paths.EDA_OUTPUT_DIR / f)))"),
        ("md", "The centre heatmap shows smoke sitting high in the frame and fire low - "
               "smoke rises. This is why **vertical flip augmentation is disabled**: an "
               "upside-down plume is not a scene that exists."),
        ("md", "## Correlations, brightness, and difficult cases"),
        ("code", "for f in ['07_correlation_matrix.png', '08_brightness_distribution.png', "
                 "'15_grid_difficult.png', '16_augmentation_preview.png']:\n"
                 "    display(Image(str(paths.EDA_OUTPUT_DIR / f)))"),
        ("md", "The correlation matrix describes **associations only** - no causal claim "
               "is made, and none of it is used as a modelling signal."),
    ],
    "03_baseline_training.ipynb": [
        ("md", "# 3 · Baseline Training (E1 · YOLOv8n)\n\n"
               "Transfer learning from COCO-pretrained weights. Reproduce with "
               "`python scripts/train_baseline.py`."),
        ("code", BOOT),
        ("code", "from src.config import load_training_config\n"
                 "load_training_config()['experiments']['e1_baseline_v8n']"),
        ("code", "log = pd.read_csv(paths.TRAINING_OUTPUT_DIR / 'experiment_log.csv')\n"
                 "log[log.experiment_id == 'e1_baseline_v8n'].T"),
        ("md", "## Training curves"),
        ("code", "display(Image(str(paths.TRAINING_OUTPUT_DIR / 'e1_baseline_v8n' / 'results.png')))"),
        ("code", "hist = pd.read_csv(paths.TRAINING_OUTPUT_DIR / 'e1_baseline_v8n' / 'results.csv')\n"
                 "hist.columns = [c.strip() for c in hist.columns]\n"
                 "hist[['epoch','metrics/precision(B)','metrics/recall(B)',"
                 "'metrics/mAP50(B)','metrics/mAP50-95(B)']].tail(10)"),
    ],
    "04_model_comparison.ipynb": [
        ("md", "# 4 · Model Comparison and Benchmarking\n\n"
               "All models evaluated on the **same** repaired validation split. The test "
               "split is reserved for the final model alone."),
        ("code", BOOT),
        ("code", "pd.read_csv(paths.TRAINING_OUTPUT_DIR / 'experiment_log.csv')[\n"
                 "    ['experiment_id','model','epochs_run','best_epoch','batch',"
                 "'map50','map50_95','recall','duration']]"),
        ("md", "## Benchmark and the recall-weighted selection\n\n"
               "Selection is **not** raw mAP: `0.35·mAP50-95 + 0.25·recall + "
               "0.25·smoke_recall + 0.15·speed`. Missing a fire costs more than a false "
               "alarm, and smoke is both the harder class and the earlier warning."),
        ("code", "pd.read_csv(paths.BENCHMARK_OUTPUT_DIR / 'benchmark_table.csv')"),
        ("code", "display(Image(str(paths.BENCHMARK_OUTPUT_DIR / 'benchmark_chart.png')))\n"
                 "json.loads((paths.BENCHMARK_OUTPUT_DIR / 'selection_report.json').read_text())"),
        ("md", "## Fair equal-epoch comparison\n\n"
               "The models were trained for different numbers of epochs (VRAM and time "
               "limits), so their final numbers also encode training length. Because "
               "validation metrics are logged every epoch, we can compare them at the "
               "same epoch."),
        ("code", "from src.train import metrics_at_epoch\n"
                 "log = pd.read_csv(paths.TRAINING_OUTPUT_DIR / 'experiment_log.csv').set_index('experiment_id')\n"
                 "for exp in [e for e in ['e2_stronger_v8s','e3_compare_11n'] if e in log.index]:\n"
                 "    n = int(log.loc[exp,'epochs_run'])\n"
                 "    base = metrics_at_epoch(paths.TRAINING_OUTPUT_DIR/'e1_baseline_v8n'/'results.csv', n)\n"
                 "    print(f\"at epoch {n}:  YOLOv8n mAP50={base['map50']:.3f}  |  \"\n"
                 "          f\"{exp} mAP50={log.loc[exp,'map50']:.3f}\")"),
    ],
    "05_hyperparameter_tuning.ipynb": [
        ("md", "# 5 · Hyperparameter Tuning\n\n"
               "A **controlled** study: one control run, then one probe per factor, all at "
               "the same budget, seed, data and batch size - so any difference is "
               "attributable to the single factor that moved."),
        ("code", BOOT),
        ("code", "log = pd.read_csv(paths.TRAINING_OUTPUT_DIR / 'experiment_log.csv')\n"
                 "probes = log[log.experiment_id.str.startswith('e4')]\n"
                 "probes[['experiment_id','optimizer','lr0','epochs_run','precision',"
                 "'recall','map50','map50_95','notes']]"),
        ("md", "### Effect of each change, measured against the control"),
        ("code", "ctrl = probes[probes.experiment_id == 'e4d_probe_baseline'].iloc[0]\n"
                 "for _, r in probes[probes.experiment_id != 'e4d_probe_baseline'].iterrows():\n"
                 "    d = r['map50_95'] - ctrl['map50_95']\n"
                 "    print(f\"{r['experiment_id']:22s} mAP50-95 {r['map50_95']:.4f}  \"\n"
                 "          f\"({d:+.4f} vs control)  ->  {'better' if d > 0 else 'not better'}\")"),
        ("md", "**Caveat we state out loud:** short probes rank configurations *under a "
               "short schedule*. The learning-rate schedule is a function of total epochs, "
               "so a setting that wins at 5 epochs is not guaranteed to win at 50. With "
               "more compute the honest design repeats the probes at full length."),
        ("md", "## Final model (E5)\n\n"
               "Continued fine-tuning of the strongest checkpoint using the winning "
               "configuration."),
        ("code", "log[log.experiment_id == 'e5_final'].T"),
    ],
    "06_final_evaluation.ipynb": [
        ("md", "# 6 · Final Evaluation\n\n"
               "Threshold chosen on **validation**; the test split evaluated **exactly "
               "once**, afterwards. Nothing below informed any decision."),
        ("code", BOOT),
        ("md", "## Threshold selection (validation only)"),
        ("code", "pd.read_csv(paths.EVALUATION_OUTPUT_DIR / 'threshold_analysis.csv').round(3)"),
        ("code", "display(Image(str(paths.EVALUATION_OUTPUT_DIR / 'threshold_analysis.png')))"),
        ("md", "## Test-set results"),
        ("code", "m = json.loads((paths.EVALUATION_OUTPUT_DIR / 'metrics_test.json').read_text())\n"
                 "print(f\"precision {m['precision']:.4f} | recall {m['recall']:.4f} | \"\n"
                 "      f\"F1 {m['f1']:.4f} | mAP50 {m['map50']:.4f} | mAP50-95 {m['map50_95']:.4f}\")\n"
                 "pd.DataFrame(m['per_class']).T.round(4)"),
        ("code", "print(m['confusion_counts'])\n"
                 "for f in ['test_confusion_matrix.png','test_confusion_matrix_normalized.png',\n"
                 "          'test_BoxPR_curve.png','test_BoxF1_curve.png']:\n"
                 "    p = paths.EVALUATION_OUTPUT_DIR / f\n"
                 "    if p.exists(): display(Image(str(p)))"),
        ("md", "## Measured inference speed"),
        ("code", "json.loads((paths.EVALUATION_OUTPUT_DIR / 'inference_speed.json').read_text())"),
        ("md", "## Error analysis"),
        ("code", "err = json.loads((paths.ERROR_ANALYSIS_OUTPUT_DIR / 'error_summary.json').read_text())\n"
                 "print('by category:', err['category_counts'])\n"
                 "print('missed by class:', err['missed_by_class'])\n"
                 "print('false positives by class:', err['false_positives_by_class'])\n"
                 "display(Image(str(paths.ERROR_ANALYSIS_OUTPUT_DIR / 'error_gallery.png')))"),
    ],
    "07_sample_predictions.ipynb": [
        ("md", "# 7 · Sample Predictions\n\n"
               "Ten samples from the held-out test split - deliberately including "
               "**failures**, not only successes."),
        ("code", BOOT),
        ("code", "pd.read_csv(paths.SAMPLE_OUTPUTS_DIR / 'sample_summary.csv')"),
        ("code", "display(Image(str(paths.SAMPLE_OUTPUTS_DIR / 'sample_grid.png')))"),
        ("md", "## Ground truth vs prediction, side by side"),
        ("code", "import itertools\n"
                 "for i in range(1, 11):\n"
                 "    gt = paths.SAMPLE_OUTPUTS_DIR / f'sample_{i:02d}_ground_truth.png'\n"
                 "    pr = paths.SAMPLE_OUTPUTS_DIR / f'sample_{i:02d}_pred.png'\n"
                 "    if gt.exists() and pr.exists():\n"
                 "        print(f'--- sample_{i:02d}: ground truth (left) vs prediction (right) ---')\n"
                 "        display(Image(str(gt), width=380), Image(str(pr), width=380))"),
    ],
}


def build(name: str, cells: list[tuple[str, str]]) -> None:
    nb = nbf.v4.new_notebook()
    nb.cells = [nbf.v4.new_markdown_cell(src) if kind == "md"
                else nbf.v4.new_code_cell(src) for kind, src in cells]
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python",
                       "name": "python3"},
        "language_info": {"name": "python"},
    }
    path = NB_DIR / name
    nbf.write(nb, path)
    log.info("wrote %s (%d cells)", name, len(nb.cells))


def main() -> int:
    NB_DIR.mkdir(parents=True, exist_ok=True)
    for name, cells in NOTEBOOKS.items():
        build(name, cells)
    log.info("%d notebooks generated in %s", len(NOTEBOOKS), NB_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
