#!/bin/bash
set -euo pipefail

if [[ $# -lt 5 || $# -gt 6 ]]; then
  echo "usage: $0 FIRST_JOB LAST_JOB SPLIT UPDATES SEED [EXPECTED_CONFIGS]" >&2
  exit 2
fi
FIRST_JOB=$1
LAST_JOB=$2
SPLIT=$3
UPDATES=$4
SEED=$5
EXPECTED_CONFIGS=${6:-40}
PROJECT=/mnt/nw/home/t.buckworth/numerai-competitive
STATUS="$PROJECT/results/monitor-${SPLIT}-u${UPDATES}-s${SEED}.log"

while squeue -u t.buckworth -h -o '%i' | awk -v first="$FIRST_JOB" -v last="$LAST_JOB" \
    '$1 >= first && $1 <= last {found=1} END {exit !found}'; do
  printf '%s jobs still active\n' "$(date -Is)" >> "$STATUS"
  sleep 60
done

cd "$PROJECT"
if "$PROJECT/uv" run python -m numerai_competitive.summarize \
    --results results --output "results/summary-${SPLIT}-u${UPDATES}-s${SEED}" \
    --split "$SPLIT" --updates "$UPDATES" --seed "$SEED" \
    --expected-configs "$EXPECTED_CONFIGS" >> "$STATUS" 2>&1; then
  printf '%s stage complete and summarized\n' "$(date -Is)" >> "$STATUS"
else
  printf '%s stage ended incomplete; inspection required\n' "$(date -Is)" >> "$STATUS"
  exit 1
fi
