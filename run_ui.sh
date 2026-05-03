#!/usr/bin/env bash
# Launch the Streamlit UI. Requires `./setup.sh` to have been run first.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

if [ -x .venv/Scripts/python.exe ]; then
    PY=.venv/Scripts/python.exe
elif [ -x .venv/bin/python ]; then
    PY=.venv/bin/python
else
    echo "[error] .venv not found. Run ./setup.sh first."
    exit 1
fi

export PYTHONIOENCODING=utf-8
export PYTHONUTF8=1
exec "$PY" -m streamlit run src/ada/ui/app.py --browser.gatherUsageStats false
