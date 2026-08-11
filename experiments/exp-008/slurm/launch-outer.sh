#!/bin/bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 OUTER_SPLIT ENVIRONMENT_JOB_ID" >&2
  exit 2
fi
OUTER_SPLIT=$1
ENVIRONMENT_JOB=$2
if [[ ! $OUTER_SPLIT =~ ^outer_[123]$ || ! $ENVIRONMENT_JOB =~ ^[0-9]+$ ]]; then
  echo "outer split must be outer_1, outer_2 or outer_3 and dependency must be numeric" >&2
  exit 1
fi
OUTER_NUMBER=${OUTER_SPLIT#outer_}
PROJECT=${NUMERAI_PROJECT:-/mnt/nw/home/t.buckworth/numerai-competitive}
SPLIT="${OUTER_SPLIT}_inner_1"
MANIFEST="$PROJECT/results/submission-${OUTER_SPLIT}-f0-u5000-s0.tsv"
SUMMARY_MARKER="$PROJECT/results/summary-${SPLIT}-u5000-s0/summary-complete.json"
if [[ $OUTER_NUMBER == 1 ]]; then
  MONITOR_SESSION=numerai-f0-monitor
  PROMOTE_SESSION=numerai-f0-promote
else
  MONITOR_SESSION="numerai-outer${OUTER_NUMBER}-f0-monitor"
  PROMOTE_SESSION="numerai-outer${OUTER_NUMBER}-f0-promote"
fi

if [[ -e "$MANIFEST" || -e "${MANIFEST}.tmp" || -e "$SUMMARY_MARKER" ]] \
    || tmux has-session -t "$MONITOR_SESSION" 2>/dev/null \
    || tmux has-session -t "$PROMOTE_SESSION" 2>/dev/null; then
  echo "outer F0 manifest, summary or controller session already exists" >&2
  exit 1
fi

TEMPORARY="${MANIFEST}.tmp"
bash "$PROJECT/slurm/submit-stage.sh" \
  "$SPLIT" 5000 0 "$ENVIRONMENT_JOB" > "$TEMPORARY"
if ! awk -F $'\t' -v expected_split="$SPLIT" '
  NF != 6 || $1 !~ /^[0-9]+(;[^[:space:]]+)?$/ || $2 != expected_split \
    || $3 != 5000 || $4 != 0 \
    || ($5 != "adamw" && $5 != "spectral") || $6 !~ /^[0-9]+$/ || $6 < 0 || $6 > 39 \
    || ++seen[$5 FS $6] != 1 { bad=1 }
  END {
    if (NR != 80) bad=1
    for (config=0; config<40; config++) {
      if (seen["adamw" FS config] != 1 || seen["spectral" FS config] != 1) bad=1
    }
    exit bad
  }
' "$TEMPORARY"; then
  echo "F0 submission must contain 40 configs x two arms" >&2
  exit 1
fi
mv "$TEMPORARY" "$MANIFEST"
FIRST_JOB=$(head -n 1 "$MANIFEST" | cut -f1 | cut -d';' -f1)
LAST_JOB=$(tail -n 1 "$MANIFEST" | cut -f1 | cut -d';' -f1)
tmux new-session -d -s "$MONITOR_SESSION" \
  "bash '$PROJECT/slurm/monitor-stage.sh' '$FIRST_JOB' '$LAST_JOB' '$SPLIT' 5000 0 40"
tmux new-session -d -s "$PROMOTE_SESSION" \
  "bash '$PROJECT/slurm/promote-f0-when-ready.sh' '$LAST_JOB' '$OUTER_SPLIT'"
printf '%s submitted %s F0 jobs=%s--%s environment_dependency=%s\n' \
  "$(date -Is)" "$OUTER_SPLIT" "$FIRST_JOB" "$LAST_JOB" "$ENVIRONMENT_JOB" \
  >> "$PROJECT/results/launch-${OUTER_SPLIT}.log"
