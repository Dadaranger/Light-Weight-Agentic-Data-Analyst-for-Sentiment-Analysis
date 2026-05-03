@echo off
REM First-time setup: create a .venv with uv and install all dependencies.
REM Re-running this is safe — uv will sync incrementally.
setlocal
set HERE=%~dp0
cd /d "%HERE%"

where uv >nul 2>nul
if errorlevel 1 (
    echo [error] uv not found. Install uv first:
    echo   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 ^| iex"
    exit /b 1
)

if not exist .venv (
    echo [setup] Creating .venv with Python 3.13
    uv venv .venv --python 3.13
)

echo [setup] Installing dependencies into .venv (editable + dev)
uv pip install --python .venv\Scripts\python.exe -e ".[dev]"

echo.
echo [done] Setup complete. Launch the UI with: run_ui.bat
