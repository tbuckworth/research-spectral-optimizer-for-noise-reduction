#!/usr/bin/env bash
set -euo pipefail
mkdir -p out logs
run() {
  local mode="$1" rank="$2"
  uv run python -u src/run_rank_modes.py --mode "$mode" --rank "$rank" \
    --steps 10000 > "logs/rankmode-${mode}-r${rank}.log" 2>&1
}
run adamw 0
for rank in 64 256 1024 2048; do run top "$rank"; done
for rank in 1 16 64 256; do run remove "$rank"; done
for rank in 1 16 64 256; do run remove-renorm "$rank"; done
