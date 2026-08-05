#!/usr/bin/env bash
set -euo pipefail
mkdir -p out logs
for rank in 256 512 1024 2048; do
  uv run python -u src/run_overfit.py --arm spectral --rank "$rank" \
    --blockwise --block-size 250000 --relative-eig-tol 1e-12 --steps 10000 \
    > "logs/big-block-spectral-r${rank}.log" 2>&1
done
for mode in remove remove-renorm; do
  for rank in 256 1024; do
    uv run python -u src/run_overfit.py --arm spectral --rank "$rank" \
      --blockwise --block-size 250000 --projection-mode "$mode" \
      --relative-eig-tol 1e-12 --steps 10000 \
      > "logs/big-block-${mode}-r${rank}.log" 2>&1
  done
done
