#!/usr/bin/env bash
# FlameGuard AI - environment setup (Git Bash / Linux / macOS)
# Creates .venv, installs PyTorch (CUDA if available) + all project dependencies.
# Keeps pip cache and temp files inside the project to avoid filling the system drive.
set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON="${PYTHON:-python}"
echo "== FlameGuard AI environment setup =="
echo "Project root: $PROJECT_ROOT"
"$PYTHON" --version

if [ ! -d ".venv" ]; then
    echo "-- Creating virtual environment (.venv)"
    "$PYTHON" -m venv .venv
fi

if [ -f ".venv/Scripts/python.exe" ]; then
    VPY=".venv/Scripts/python.exe"   # Windows layout
else
    VPY=".venv/bin/python"           # POSIX layout
fi

export PIP_CACHE_DIR="$PROJECT_ROOT/.pipcache"
export TMPDIR="$PROJECT_ROOT/.tmp"
export TMP="$PROJECT_ROOT/.tmp"
export TEMP="$PROJECT_ROOT/.tmp"
mkdir -p "$PIP_CACHE_DIR" "$TMPDIR"

"$VPY" -m pip install --upgrade pip

echo "-- Installing PyTorch (trying CUDA wheel indexes, falling back to CPU)"
TORCH_OK=0
for IDX in cu128 cu126 cu130; do
    echo "   trying https://download.pytorch.org/whl/$IDX"
    if "$VPY" -m pip install torch torchvision --index-url "https://download.pytorch.org/whl/$IDX"; then
        TORCH_OK=1
        echo "   installed torch from $IDX"
        break
    fi
done
if [ "$TORCH_OK" -eq 0 ]; then
    echo "   CUDA indexes failed; installing CPU wheels from PyPI"
    "$VPY" -m pip install torch torchvision
fi

echo "-- Verifying torch"
"$VPY" - <<'PYEOF'
import torch
print("torch", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
PYEOF

echo "-- Installing project dependencies"
"$VPY" -m pip install -r requirements.txt

echo "-- Done. Activate with:"
echo "   source .venv/Scripts/activate   (Git Bash on Windows)"
echo "   .venv\\Scripts\\activate         (PowerShell/cmd)"
