#!/bin/bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: $0 ENVIRONMENT_JOB_ID [OUTER_SPLIT]" >&2
  exit 2
fi
ENVIRONMENT_JOB=$1
OUTER_SPLIT=${2:-outer_1}
if [[ ! $OUTER_SPLIT =~ ^outer_[123]$ ]]; then
  echo "outer split must be outer_1, outer_2 or outer_3" >&2
  exit 1
fi
OUTER_NUMBER=${OUTER_SPLIT#outer_}
INNER_COUNT=$((OUTER_NUMBER + 1))
PROJECT=${NUMERAI_PROJECT:-/mnt/nw/home/t.buckworth/numerai-competitive}
SUMMARY="$PROJECT/results/summary-${OUTER_SPLIT}_inner_1-u5000-s0/scores.csv"
SUMMARY_MARKER="$PROJECT/results/summary-${OUTER_SPLIT}_inner_1-u5000-s0/summary-complete.json"
SELECTION="$PROJECT/results/selection-${OUTER_SPLIT}-f0-top12.json"
MANIFEST="$PROJECT/results/submission-${OUTER_SPLIT}-f1-u20000-s0.tsv"
LOG="$PROJECT/results/promote-${OUTER_SPLIT}-f0.log"
if [[ $OUTER_NUMBER == 1 ]]; then
  F0_MONITOR=numerai-f0-monitor
else
  F0_MONITOR="numerai-outer${OUTER_NUMBER}-f0-monitor"
fi

while [[ ! -f "$SUMMARY_MARKER" ]]; do
  if ! tmux has-session -t "$F0_MONITOR" 2>/dev/null; then
    echo "F0 monitor exited without an audited completion marker" >&2
    exit 1
  fi
  printf '%s waiting for audited F0 summary\n' "$(date -Is)" >> "$LOG"
  sleep 60
done

cd "$PROJECT"
if [[ -e "$SELECTION" || -e "$MANIFEST" || -e "${MANIFEST}.tmp" ]]; then
  echo "F1 selection, manifest or temporary file already exists" >&2
  exit 1
fi
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
SPLITS=()
for INDEX in $(seq 1 "$INNER_COUNT"); do
  SPLITS+=("${OUTER_SPLIT}_inner_${INDEX}")
done
bash "$PROJECT/slurm/submit-selected.sh" "$SELECTION" 20000 0 "$ENVIRONMENT_JOB" \
  "${SPLITS[@]}" > "$TEMPORARY"
mv "$TEMPORARY" "$MANIFEST"
FIRST_JOB=$(head -n 1 "$MANIFEST" | cut -f1 | cut -d';' -f1)
LAST_JOB=$(tail -n 1 "$MANIFEST" | cut -f1 | cut -d';' -f1)
EXPECTED_ROWS=$((EXPECTED * 2 * INNER_COUNT))
if [[ $(wc -l < "$MANIFEST") -ne $EXPECTED_ROWS ]]; then
  echo "submission manifest row count differs from paired union x arms x folds" >&2
  exit 1
fi

for INDEX in $(seq 1 "$INNER_COUNT"); do
  if [[ $OUTER_NUMBER == 1 ]]; then
    SESSION="numerai-f1-inner${INDEX}"
  else
    SESSION="numerai-outer${OUTER_NUMBER}-f1-inner${INDEX}"
  fi
  tmux new-session -d -s "$SESSION" \
    "bash '$PROJECT/slurm/monitor-stage.sh' '$FIRST_JOB' '$LAST_JOB' '${OUTER_SPLIT}_inner_${INDEX}' 20000 0 '$EXPECTED'"
done
printf '%s submitted F1 union=%s jobs=%s--%s env_dependency=%s\n' \
  "$(date -Is)" "$EXPECTED" "$FIRST_JOB" "$LAST_JOB" "$ENVIRONMENT_JOB" >> "$LOG"
