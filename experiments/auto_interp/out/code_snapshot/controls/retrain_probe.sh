#!/usr/bin/env bash
# Wrapper: run retrain_probe.py with the session venv python (has numpy+sklearn).
set -euo pipefail
VENV=/tmp/claude-1000/-home-hal-dev-interp-dm-hack-mats-open-auto-interp/4b5dae78-2508-4760-a39f-8c08559607fb/scratchpad/venv
exec "$VENV/bin/python" controls/retrain_probe.py "$@"
