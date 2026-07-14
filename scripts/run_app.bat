@echo off
REM Launch the FlameGuard AI Streamlit application.
cd /d "%~dp0.."
".venv\Scripts\python.exe" -m streamlit run app.py %*
