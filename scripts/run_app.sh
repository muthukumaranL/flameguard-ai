#!/usr/bin/env bash
# Launch the FlameGuard AI Streamlit application.
cd "$(dirname "$0")/.."
if [ -f ".venv/Scripts/python.exe" ]; then
    ".venv/Scripts/python.exe" -m streamlit run app.py "$@"
else
    ".venv/bin/python" -m streamlit run app.py "$@"
fi
