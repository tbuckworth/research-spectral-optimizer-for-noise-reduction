#!/usr/bin/env bash
set -euo pipefail
mkdir -p out logs

common=(--steps 100000 --eval-every 500 --seed 20260805 --run-tag 100k)

uv run python -u src/run_overfit.py --arm adamw --rank 0 "${common[@]}" \
  > logs/big-100k-adamw.log 2>&1

for rank in 256 512 1024 2048; do
  uv run python -u src/run_overfit.py --arm spectral --rank "$rank" \
    --blockwise --block-size 250000 --projection-mode top \
    --relative-eig-tol 1e-12 "${common[@]}" \
    > "logs/big-100k-top-r${rank}.log" 2>&1
done

for mode in remove remove-renorm; do
  for rank in 256 1024; do
    uv run python -u src/run_overfit.py --arm spectral --rank "$rank" \
      --blockwise --block-size 250000 --projection-mode "$mode" \
      --relative-eig-tol 1e-12 "${common[@]}" \
      > "logs/big-100k-${mode}-r${rank}.log" 2>&1
  done
done
