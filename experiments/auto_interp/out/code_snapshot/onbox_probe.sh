#!/usr/bin/env bash
# Runner for the auto_interp probe study: remote_run.sh has already verified CUDA.
set -euo pipefail
python3 run_probe_study.py
