#!/bin/bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: $0 DEPENDENCY_JOB_ID [OUTER_SPLIT]" >&2
  exit 2
fi
DEPENDENCY_JOB=$1
OUTER_SPLIT=${2:-outer_1}
if [[ ! $OUTER_SPLIT =~ ^outer_[123]$ ]]; then
  echo "outer split must be outer_1, outer_2 or outer_3" >&2
  exit 1
fi
OUTER_NUMBER=${OUTER_SPLIT#outer_}
INNER_COUNT=$((OUTER_NUMBER + 1))
SUPERVISOR_SESSION="numerai-outer${OUTER_NUMBER}-supervisor"
AUDIT_SESSION="numerai-outer${OUTER_NUMBER}-audit"
if [[ $OUTER_NUMBER == 1 ]]; then
  F2_SUPERVISOR=numerai-f2-supervisor
else
  F2_SUPERVISOR="numerai-outer${OUTER_NUMBER}-f2-supervisor"
fi
PROJECT=${NUMERAI_PROJECT:-/mnt/nw/home/t.buckworth/numerai-competitive}
SEARCH=${NUMERAI_SEARCH_CONFIG:-$PROJECT/results/search-v1-high-rank.json}
[[ -f $SEARCH ]] || SEARCH="$PROJECT/configs/search-v1.json"
export NUMERAI_SEARCH_CONFIG="$SEARCH"
SELECTION="$PROJECT/results/selection-${OUTER_SPLIT}-f2-budget-top1.json"
MANIFEST="$PROJECT/results/submission-${OUTER_SPLIT}-eval-budgeted.tsv"
OUTPUT="$PROJECT/results/audit-${OUTER_SPLIT}-budgeted"
LOG="$PROJECT/results/promote-${OUTER_SPLIT}-f2.log"
SUMMARIES=()
MARKERS=()
for INDEX in $(seq 1 "$INNER_COUNT"); do
  SPLIT="${OUTER_SPLIT}_inner_${INDEX}"
  for BUDGET in 5000 20000 100000; do
    for SEED in 0 1 2; do
      SUMMARIES+=("$PROJECT/results/summary-f2-${SPLIT}-u${BUDGET}-s${SEED}/scores.csv")
      MARKERS+=("$PROJECT/results/summary-f2-${SPLIT}-u${BUDGET}-s${SEED}/summary-complete.json")
    done
  done
done

while true; do
  missing=0
  for marker in "${MARKERS[@]}"; do
    [[ -f "$marker" ]] || missing=$((missing + 1))
  done
  [[ $missing -eq 0 ]] && break
  if ! tmux has-session -t "$F2_SUPERVISOR" 2>/dev/null; then
    echo "F2 supervisor exited without all audited completion markers" >&2
    exit 1
  fi
  printf '%s waiting for %s audited F2 summaries\n' "$(date -Is)" "$missing" >> "$LOG"
  sleep 60
done

cd "$PROJECT"
if [[ -e "$SELECTION" || -e "$MANIFEST" || -e "${MANIFEST}.tmp" ]] \
    || tmux has-session -t "$SUPERVISOR_SESSION" 2>/dev/null \
    || tmux has-session -t "$AUDIT_SESSION" 2>/dev/null; then
  echo "outer selection, manifest, temporary file or supervisor session already exists" >&2
  exit 1
fi
"$PROJECT/uv" run --no-sync python -m numerai_competitive.select_budgeted_configs \
  --scores "${SUMMARIES[@]}" --top 1 --allow-asymmetric --output "$SELECTION"
mapfile -t CANDIDATE_IDS < <(python3 -c '
import json,sys
value=json.load(open(sys.argv[1]))["selected"]
print(*sorted(set(value["paired_union"] + value["high_rank_spectral"])), sep="\n")
' "$PROJECT/results/selection-${OUTER_SPLIT}-f1-top4.json")
"$PROJECT/uv" run --no-sync python -m numerai_competitive.budget_diagnostics \
  --scores "${SUMMARIES[@]}" --config-id "${CANDIDATE_IDS[@]}" --top 1 \
  --allow-asymmetric --output "$PROJECT/results/budget-diagnostics-${OUTER_SPLIT}"
TEMPORARY="${MANIFEST}.tmp"
bash "$PROJECT/slurm/submit-outer-eval.sh" \
  "$SELECTION" "$OUTER_SPLIT" selected "$DEPENDENCY_JOB" 0,1,2 > "$TEMPORARY"
mv "$TEMPORARY" "$MANIFEST"
if [[ $(wc -l < "$MANIFEST") -ne 6 ]]; then
  echo "outer submission manifest must contain two arms x three seeds" >&2
  exit 1
fi
tmux new-session -d -s "$SUPERVISOR_SESSION" \
  "NUMERAI_SEARCH_CONFIG='$SEARCH' bash '$PROJECT/slurm/supervise-resumable-stage.sh' '$MANIFEST' --skip-summary"
tmux new-session -d -s "$AUDIT_SESSION" \
  "NUMERAI_SEARCH_CONFIG='$SEARCH' bash '$PROJECT/slurm/audit-outer-when-ready.sh' '$MANIFEST' '$SELECTION' '$OUTER_SPLIT' '$OUTPUT'"
printf '%s submitted selected %s evaluation dependency=%s\n' \
  "$(date -Is)" "$OUTER_SPLIT" "$DEPENDENCY_JOB" >> "$LOG"
