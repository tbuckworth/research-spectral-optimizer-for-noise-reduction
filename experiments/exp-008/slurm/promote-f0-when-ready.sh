#!/bin/bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 ENVIRONMENT_JOB_ID" >&2
  exit 2
fi
ENVIRONMENT_JOB=$1
PROJECT=/mnt/nw/home/t.buckworth/numerai-competitive
SUMMARY="$PROJECT/results/summary-outer_1_inner_1-u5000-s0/scores.csv"
SELECTION="$PROJECT/results/selection-outer_1-f0-top12.json"
MANIFEST="$PROJECT/results/submission-outer_1-f1-u20000-s0.tsv"
LOG="$PROJECT/results/promote-outer_1-f0.log"

while [[ ! -f "$SUMMARY" ]]; do
  printf '%s waiting for audited F0 summary\n' "$(date -Is)" >> "$LOG"
  sleep 60
done

cd "$PROJECT"
"$PROJECT/uv" run --no-sync python -m numerai_competitive.select_configs \
  --scores "$SUMMARY" --top 12 --output "$SELECTION"
EXPECTED=$(python3 -c \
  'import json,sys; print(len(json.load(open(sys.argv[1]))["selected"]["paired_union"]))' \
  "$SELECTION")
if [[ $EXPECTED -lt 12 || $EXPECTED -gt 24 ]]; then
  echo "invalid paired-union size $EXPECTED" >&2
  exit 1
fi

TEMPORARY="${MANIFEST}.tmp"
bash "$PROJECT/slurm/submit-selected.sh" "$SELECTION" 20000 0 "$ENVIRONMENT_JOB" \
  outer_1_inner_1 outer_1_inner_2 > "$TEMPORARY"
mv "$TEMPORARY" "$MANIFEST"
FIRST_JOB=$(head -n 1 "$MANIFEST" | cut -f1 | cut -d';' -f1)
LAST_JOB=$(tail -n 1 "$MANIFEST" | cut -f1 | cut -d';' -f1)
EXPECTED_ROWS=$((EXPECTED * 2 * 2))
if [[ $(wc -l < "$MANIFEST") -ne $EXPECTED_ROWS ]]; then
  echo "submission manifest row count differs from paired union x arms x folds" >&2
  exit 1
fi

tmux new-session -d -s numerai-f1-inner1 \
  "bash '$PROJECT/slurm/monitor-stage.sh' '$FIRST_JOB' '$LAST_JOB' outer_1_inner_1 20000 0 '$EXPECTED'"
tmux new-session -d -s numerai-f1-inner2 \
  "bash '$PROJECT/slurm/monitor-stage.sh' '$FIRST_JOB' '$LAST_JOB' outer_1_inner_2 20000 0 '$EXPECTED'"
printf '%s submitted F1 union=%s jobs=%s--%s env_dependency=%s\n' \
  "$(date -Is)" "$EXPECTED" "$FIRST_JOB" "$LAST_JOB" "$ENVIRONMENT_JOB" >> "$LOG"
