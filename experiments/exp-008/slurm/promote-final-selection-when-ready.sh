#!/bin/bash
set -euo pipefail

if [[ $# -ne 2 || ! $1 =~ ^[0-9]+$ || ! -f $2 ]]; then
  echo "usage: $0 DEPENDENCY_JOB_ID FINAL_SELECTION_MANIFEST" >&2
  exit 2
fi
DEPENDENCY_JOB=$1
FINAL_SELECTION_MANIFEST=$2
PROJECT=${NUMERAI_PROJECT:-/mnt/nw/home/t.buckworth/numerai-competitive}
SEARCH=${NUMERAI_SEARCH_CONFIG:-$PROJECT/results/search-v1-high-rank.json}
[[ -f $SEARCH ]] || { echo "audited high-rank search is missing" >&2; exit 1; }
export NUMERAI_SEARCH_CONFIG="$SEARCH"
SELECTION="$PROJECT/results/selection-final-top1.json"
MANIFEST="$PROJECT/results/submission-final-refit-budgeted.tsv"
LOG="$PROJECT/results/promote-final-selection.log"
SUPERVISOR_SESSION=numerai-final-selection-supervisor
REFIT_SESSION=numerai-final-refit-supervisor
SUMMARIES=()
MARKERS=()
while IFS=$'\t' read -r SPLIT BUDGET SEED; do
  SUMMARIES+=("$PROJECT/results/summary-final-${SPLIT}-u${BUDGET}-s${SEED}/scores.csv")
  MARKERS+=("$PROJECT/results/summary-final-${SPLIT}-u${BUDGET}-s${SEED}/summary-complete.json")
done < <(awk -F $'\t' '{print $2 "\t" $3 "\t" $4}' "$FINAL_SELECTION_MANIFEST" | sort -u)

while true; do
  MISSING=0
  for MARKER in "${MARKERS[@]}"; do [[ -f $MARKER ]] || MISSING=$((MISSING + 1)); done
  [[ $MISSING -eq 0 ]] && break
  if ! tmux has-session -t "$SUPERVISOR_SESSION" 2>/dev/null; then
    echo "final-selection supervisor exited without all audited summaries" >&2
    exit 1
  fi
  printf '%s waiting for %s final-selection summaries\n' "$(date -Is)" "$MISSING" >> "$LOG"
  sleep 60
done

if [[ -e "$SELECTION" || -e "$MANIFEST" || -e "${MANIFEST}.tmp" ]] \
    || tmux has-session -t "$REFIT_SESSION" 2>/dev/null; then
  echo "final winner, refit manifest or supervisor already exists" >&2
  exit 1
fi
"$PROJECT/uv" run --no-sync python -m numerai_competitive.select_budgeted_configs \
  --scores "${SUMMARIES[@]}" --top 1 --output "$SELECTION"
bash "$PROJECT/slurm/submit-refit.sh" \
  "$SELECTION" selected "$DEPENDENCY_JOB" 0,1,2 > "${MANIFEST}.tmp"
if [[ $(wc -l < "${MANIFEST}.tmp") -ne 6 ]] \
    || [[ $(cut -f2-5 "${MANIFEST}.tmp" | sort -u | wc -l) -ne 6 ]]; then
  echo "refit manifest must contain two selected arms x three seeds" >&2
  exit 1
fi
mv "${MANIFEST}.tmp" "$MANIFEST"
tmux new-session -d -s "$REFIT_SESSION" \
  "NUMERAI_SEARCH_CONFIG='$SEARCH' bash '$PROJECT/slurm/supervise-refits.sh' '$MANIFEST' '$SELECTION'"
printf '%s submitted six final refits dependency=%s\n' \
  "$(date -Is)" "$DEPENDENCY_JOB" >> "$LOG"
