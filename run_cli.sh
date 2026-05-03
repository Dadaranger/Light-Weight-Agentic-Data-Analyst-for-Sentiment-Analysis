#!/usr/bin/env bash
# Headless CLI runner. Forwards all args to `python -m ada.cli`.
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
exec "$PY" -m ada.cli "$@"
