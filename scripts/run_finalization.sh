#!/usr/bin/env bash
# Everything that happens AFTER the models are trained, in the order it must happen.
#
#   benchmark  -> selects models/final/best.pt  (validation only)
#   evaluate   -> picks the threshold on validation, then touches the test set ONCE
#   errors     -> categorised failures + the colour-prior probe
#   samples    -> the 10 sample predictions (successes and failures)
#   demo video -> refresh the annotated backup clip with the final model
#   pytest     -> automated suite, logged to outputs/test_report.txt
#   report / slides / package / verify
#
# Usage:  bash scripts/run_finalization.sh [group_number]
set -e

cd "$(dirname "$0")/.."
GROUP="${1:-##}"
PY=".venv/Scripts/python.exe"
[ -f "$PY" ] || PY=".venv/bin/python"
export PYTHONIOENCODING=utf-8

step() { echo; echo "=============== $1 ==============="; }

step "1/10  Benchmark + final model selection (validation split)"
"$PY" scripts/benchmark.py

step "2/10  Threshold analysis + ONE-TIME test evaluation"
"$PY" scripts/evaluate_final.py

step "3/10  Error analysis (+ colour-prior probe)"
"$PY" scripts/error_analysis.py

step "4/10  Sample predictions"
"$PY" scripts/generate_samples.py

step "5/10  Refresh backup demo video with the final model"
"$PY" scripts/make_demo_video.py

step "6/10  Automated test suite"
"$PY" -m pytest tests/ -q | tee outputs/test_report.txt

step "7/10  Scrum artefacts"
"$PY" scripts/generate_agile.py

step "8/10  Report (Markdown + DOCX + PDF)"
"$PY" scripts/generate_report.py

step "9/10  Presentation + speaker notes + demo script"
"$PY" scripts/generate_slides.py

step "10/10 Package submission"
"$PY" scripts/package_submission.py --group "$GROUP"

echo
echo "=============== VERIFICATION ==============="
"$PY" scripts/verify_project.py
