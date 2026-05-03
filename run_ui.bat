@echo off
REM Launch the Streamlit UI. Requires `setup.bat` to have been run first.
setlocal
set HERE=%~dp0
cd /d "%HERE%"

if not exist .venv\Scripts\python.exe (
    echo [error] .venv not found. Run setup.bat first.
    exit /b 1
)

set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
.venv\Scripts\python.exe -m streamlit run src\ada\ui\app.py --browser.gatherUsageStats false
