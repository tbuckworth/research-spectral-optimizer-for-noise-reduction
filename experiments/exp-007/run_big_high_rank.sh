#!/usr/bin/env bash
set -euo pipefail
mkdir -p out logs
uv run python -u src/run_overfit.py --arm spectral --rank 256 \
  --relative-eig-tol 1e-12 --steps 10000 \
  > logs/big-spectral-r256.log 2>&1
uv run python -u src/run_overfit.py --arm spectral --rank 512 \
  --relative-eig-tol 1e-12 --steps 10000 \
  > logs/big-spectral-r512.log 2>&1
