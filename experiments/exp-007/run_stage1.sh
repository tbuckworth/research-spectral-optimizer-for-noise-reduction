#!/usr/bin/env bash
set -euo pipefail
mkdir -p out logs
uv run python -u src/run_overfit.py --arm adamw --steps 20000 > logs/adamw.log 2>&1
uv run python -u src/run_overfit.py --arm spectral --rank 16 --steps 20000 > logs/spectral-r16.log 2>&1
