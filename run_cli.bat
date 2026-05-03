@echo off
REM Headless CLI runner. Forwards all args to `python -m ada.cli`.
REM Example: run_cli.bat run path\to\data.csv --project myproj --auto-confirm
setlocal
set HERE=%~dp0
cd /d "%HERE%"

if not exist .venv\Scripts\python.exe (
    echo [error] .venv not found. Run setup.bat first.
    exit /b 1
)

set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
.venv\Scripts\python.exe -m ada.cli %*
