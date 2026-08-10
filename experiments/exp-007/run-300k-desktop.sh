#!/usr/bin/env bash
set -euo pipefail

rank="${1:?usage: run-300k-desktop.sh RANK}"
case "$rank" in
  1024|1536) ;;
  *) echo "rank must be 1024 or 1536" >&2; exit 2 ;;
esac

root="$(cd "$(dirname "$0")" && pwd)"
export NUMERAI_SHARD="/media/titus/big/tmp/numerai-v5-full-shard"
export CUDA_VISIBLE_DEVICES=0
cd "$root"

exec /usr/bin/python3 -u src/run_overfit.py \
  --arm spectral \
  --rank "$rank" \
  --blockwise \
  --block-size 250000 \
  --projection-mode top \
  --relative-eig-tol 1e-12 \
  --steps 300000 \
  --eval-every 10000 \
  --checkpoint-every 100000 \
  --seed 20260805 \
  --run-tag 300k-3090 \
  --full-valid-test
