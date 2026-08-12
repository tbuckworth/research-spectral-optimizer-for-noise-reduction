#!/bin/bash
set -euo pipefail

if [[ $# -ne 2 || ! $2 =~ ^[0-9]+$ ]]; then
  echo "usage: $0 FROZEN_SELECTION_JSON DEPENDENCY_JOB_ID" >&2
  exit 2
fi
SELECTION=$1
DEPENDENCY=$2
PROJECT=${NUMERAI_PROJECT:-/mnt/nw/home/t.buckworth/numerai-competitive}
SEARCH=${NUMERAI_SEARCH_CONFIG:-$PROJECT/results/search-v1-high-rank.json}
[[ -f $SELECTION && -f $SEARCH ]] || { echo "selection or augmented search is missing" >&2; exit 1; }
if tmux has-session -t numerai-nested-continuation 2>/dev/null; then
  echo "walk-forward continuation controller already exists" >&2
  exit 1
fi
export NUMERAI_SEARCH_CONFIG="$SEARCH"

for NUMBER in 1 2 3; do
  SPLIT="outer_${NUMBER}"
  MANIFEST="$PROJECT/results/submission-${SPLIT}-walkforward.tsv"
  OUTPUT="$PROJECT/results/audit-${SPLIT}-budgeted"
  SUPERVISOR="numerai-outer${NUMBER}-supervisor"
  AUDITOR="numerai-outer${NUMBER}-audit"
  if [[ -e $MANIFEST || -e ${MANIFEST}.tmp || -e $OUTPUT \
        || -e $PROJECT/results/submission-${SPLIT}-f0-u5000-s0.tsv ]] \
        || tmux has-session -t "$SUPERVISOR" 2>/dev/null \
        || tmux has-session -t "$AUDITOR" 2>/dev/null; then
    echo "$SPLIT walk-forward artifact, obsolete search, or session already exists" >&2
    exit 1
  fi
  bash "$PROJECT/slurm/submit-outer-eval.sh" \
    "$SELECTION" "$SPLIT" selected "$DEPENDENCY" 0,1,2 > "${MANIFEST}.tmp"
  mv "${MANIFEST}.tmp" "$MANIFEST"
  [[ $(wc -l < "$MANIFEST") -eq 6 ]] \
    || { echo "$SPLIT walk-forward manifest must contain six cells" >&2; exit 1; }
  tmux new-session -d -s "$SUPERVISOR" \
    "NUMERAI_SEARCH_CONFIG='$SEARCH' bash '$PROJECT/slurm/supervise-resumable-stage.sh' '$MANIFEST' --skip-summary"
  tmux new-session -d -s "$AUDITOR" \
    "NUMERAI_SEARCH_CONFIG='$SEARCH' bash '$PROJECT/slurm/audit-outer-when-ready.sh' '$MANIFEST' '$SELECTION' '$SPLIT' '$OUTPUT'"
done
tmux new-session -d -s numerai-nested-continuation \
  "NUMERAI_SEARCH_CONFIG='$SEARCH' bash '$PROJECT/slurm/continue-nested-pipeline.sh'"
printf '%s submitted fixed-config walk-forward outer evaluation dependency=%s\n' \
  "$(date -Is)" "$DEPENDENCY" >> "$PROJECT/results/launch-walkforward.log"
