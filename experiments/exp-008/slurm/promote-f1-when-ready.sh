#!/bin/bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 DEPENDENCY_JOB_ID" >&2
  exit 2
fi
DEPENDENCY_JOB=$1
PROJECT=/mnt/nw/home/t.buckworth/numerai-competitive
SUMMARY_1="$PROJECT/results/summary-outer_1_inner_1-u20000-s0/scores.csv"
SUMMARY_2="$PROJECT/results/summary-outer_1_inner_2-u20000-s0/scores.csv"
SELECTION="$PROJECT/results/selection-outer_1-f1-top4.json"
MANIFEST="$PROJECT/results/submission-outer_1-f2-u100000.tsv"
LOG="$PROJECT/results/promote-outer_1-f1.log"

while [[ ! -f "$SUMMARY_1" || ! -f "$SUMMARY_2" ]]; do
  printf '%s waiting for both audited F1 summaries\n' "$(date -Is)" >> "$LOG"
  sleep 60
done

cd "$PROJECT"
"$PROJECT/uv" run --no-sync python -m numerai_competitive.select_configs \
  --scores "$SUMMARY_1" "$SUMMARY_2" --top 4 --output "$SELECTION"
EXPECTED=$(python3 -c \
  'import json,sys; print(len(json.load(open(sys.argv[1]))["selected"]["paired_union"]))' \
  "$SELECTION")
if [[ $EXPECTED -lt 4 || $EXPECTED -gt 8 ]]; then
  echo "invalid paired-union size $EXPECTED" >&2
  exit 1
fi

TEMPORARY="${MANIFEST}.tmp"
: > "$TEMPORARY"
for SEED in 0 1 2; do
  bash "$PROJECT/slurm/submit-selected.sh" "$SELECTION" 100000 "$SEED" \
    "$DEPENDENCY_JOB" outer_1_inner_1 outer_1_inner_2 >> "$TEMPORARY"
done
mv "$TEMPORARY" "$MANIFEST"
FIRST_JOB=$(head -n 1 "$MANIFEST" | cut -f1 | cut -d';' -f1)
LAST_JOB=$(tail -n 1 "$MANIFEST" | cut -f1 | cut -d';' -f1)
EXPECTED_ROWS=$((EXPECTED * 2 * 2 * 3))
if [[ $(wc -l < "$MANIFEST") -ne $EXPECTED_ROWS ]]; then
  echo "submission manifest row count differs from paired union x arms x folds x seeds" >&2
  exit 1
fi

for SPLIT in outer_1_inner_1 outer_1_inner_2; do
  for SEED in 0 1 2; do
    tmux new-session -d -s "numerai-f2-${SPLIT##*_}-s${SEED}" \
      "bash '$PROJECT/slurm/monitor-stage.sh' '$FIRST_JOB' '$LAST_JOB' '$SPLIT' 100000 '$SEED' '$EXPECTED'"
  done
done
printf '%s submitted F2 union=%s jobs=%s--%s dependency=%s\n' \
  "$(date -Is)" "$EXPECTED" "$FIRST_JOB" "$LAST_JOB" "$DEPENDENCY_JOB" >> "$LOG"
