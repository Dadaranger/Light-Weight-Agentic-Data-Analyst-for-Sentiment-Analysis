#!/usr/bin/env bash
# First-time setup: create a .venv with uv and install all dependencies.
# Re-running this is safe — uv will sync incrementally.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

if ! command -v uv >/dev/null 2>&1; then
    echo "[error] uv not found. Install uv first:"
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

if [ ! -d .venv ]; then
    echo "[setup] Creating .venv with Python 3.13"
    uv venv .venv --python 3.13
fi

# Pick the right python path for the platform
if [ -x .venv/Scripts/python.exe ]; then
    PY=.venv/Scripts/python.exe
else
    PY=.venv/bin/python
fi

echo "[setup] Installing dependencies into .venv (editable + dev)"
uv pip install --python "$PY" -e ".[dev]"

echo
echo "[done] Setup complete. Launch the UI with: ./run_ui.sh"
