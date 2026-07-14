@echo off
REM FlameGuard AI - environment setup (Windows cmd)
REM Creates .venv and installs PyTorch (CUDA when available) + dependencies.
setlocal
cd /d "%~dp0.."

echo == FlameGuard AI environment setup ==
python --version || goto :error

if not exist .venv (
    echo -- Creating virtual environment
    python -m venv .venv || goto :error
)

set "PIP_CACHE_DIR=%CD%\.pipcache"
set "TMP=%CD%\.tmp"
set "TEMP=%CD%\.tmp"
if not exist "%PIP_CACHE_DIR%" mkdir "%PIP_CACHE_DIR%"
if not exist "%TMP%" mkdir "%TMP%"

set "VPY=.venv\Scripts\python.exe"
"%VPY%" -m pip install --upgrade pip || goto :error

echo -- Installing PyTorch (CUDA wheels, CPU fallback)
"%VPY%" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
if errorlevel 1 (
    echo    CUDA cu128 failed, trying cu126...
    "%VPY%" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
)
if errorlevel 1 (
    echo    CUDA indexes failed, installing CPU wheels
    "%VPY%" -m pip install torch torchvision || goto :error
)

echo -- Verifying torch
"%VPY%" -c "import torch; print('torch', torch.__version__, '| cuda:', torch.cuda.is_available())" || goto :error

echo -- Installing project dependencies
"%VPY%" -m pip install -r requirements.txt || goto :error

echo -- Done. Activate with: .venv\Scripts\activate
exit /b 0

:error
echo Setup failed - see messages above.
exit /b 1
