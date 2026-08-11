#!/bin/bash
set -euo pipefail

if [[ $# -ne 1 || ! $1 =~ ^[0-9]+$ ]]; then
  echo "usage: $0 DEPENDENCY_JOB_ID" >&2
  exit 2
fi
DEPENDENCY_JOB=$1
PROJECT=${NUMERAI_PROJECT:-/mnt/nw/home/t.buckworth/numerai-competitive}
SEARCH=${NUMERAI_SEARCH_CONFIG:-$PROJECT/results/search-v1-high-rank.json}
[[ -f $SEARCH ]] || { echo "audited high-rank search is missing" >&2; exit 1; }
export NUMERAI_SEARCH_CONFIG="$SEARCH"
SELECTION="$PROJECT/results/selection-final-outer-winner-union.json"
MANIFEST="$PROJECT/results/submission-final-selection-budgeted.tsv"
SUPERVISOR_SESSION=numerai-final-selection-supervisor
PROMOTE_SESSION=numerai-final-selection-promote
OUTER_SELECTIONS=()
OUTER_AUDITS=()
for OUTER_NUMBER in 1 2 3; do
  OUTER_SELECTIONS+=("$PROJECT/results/selection-outer_${OUTER_NUMBER}-f2-budget-top1.json")
  OUTER_AUDITS+=("$PROJECT/results/audit-outer_${OUTER_NUMBER}-budgeted/outer-audit.json")
done

for PATH_REQUIRED in "${OUTER_SELECTIONS[@]}" "${OUTER_AUDITS[@]}"; do
  if [[ ! -f $PATH_REQUIRED ]]; then
    echo "required audited outer artifact is missing: $PATH_REQUIRED" >&2
    exit 1
  fi
done
if [[ -e "$SELECTION" || -e "$MANIFEST" || -e "${MANIFEST}.tmp" ]] \
    || tmux has-session -t "$SUPERVISOR_SESSION" 2>/dev/null \
    || tmux has-session -t "$PROMOTE_SESSION" 2>/dev/null; then
  echo "final selection artifact, manifest or supervisor already exists" >&2
  exit 1
fi

cd "$PROJECT"
"$PROJECT/uv" run --no-sync python -m numerai_competitive.final_candidates \
  --selection "${OUTER_SELECTIONS[0]}" --selection "${OUTER_SELECTIONS[1]}" \
  --selection "${OUTER_SELECTIONS[2]}" \
  --audit "${OUTER_AUDITS[0]}" --audit "${OUTER_AUDITS[1]}" \
  --audit "${OUTER_AUDITS[2]}" --output "$SELECTION"
EXPECTED=$(python3 -c \
  'import json,sys; print(len(json.load(open(sys.argv[1]))["budgeted_candidates"]))' \
  "$SELECTION")
if [[ $EXPECTED -lt 1 || $EXPECTED -gt 6 ]]; then
  echo "invalid union of outer winners" >&2
  exit 1
fi

TEMPORARY="${MANIFEST}.tmp"
: > "$TEMPORARY"
for SEED in 0 1 2; do
  NUMERAI_REUSE_COMPLETE=1 bash "$PROJECT/slurm/submit-budgeted-selected.sh" \
    "$SELECTION" "$SEED" "$DEPENDENCY_JOB" \
    outer_3_inner_1 outer_3_inner_2 outer_3_inner_3 outer_3_inner_4 >> "$TEMPORARY"
done
EXPECTED_ROWS=$((EXPECTED * 2 * 4 * 3))
if [[ $(wc -l < "$TEMPORARY") -ne $EXPECTED_ROWS ]] \
    || [[ $(cut -f2-6 "$TEMPORARY" | sort -u | wc -l) -ne $EXPECTED_ROWS ]]; then
  echo "final manifest differs from union x arms x canonical folds x seeds" >&2
  exit 1
fi
mv "$TEMPORARY" "$MANIFEST"
LAST_SUBMITTED=$(awk -F $'\t' '$1 != 0 {value=$1} END {sub(/;.*/, "", value); print value}' \
  "$MANIFEST")
[[ $LAST_SUBMITTED =~ ^[0-9]+$ ]] || LAST_SUBMITTED=$DEPENDENCY_JOB
tmux new-session -d -s "$SUPERVISOR_SESSION" \
  "NUMERAI_SEARCH_CONFIG='$SEARCH' NUMERAI_SUMMARY_PREFIX=final- bash '$PROJECT/slurm/supervise-resumable-stage.sh' '$MANIFEST' --selection '$SELECTION'"
tmux new-session -d -s "$PROMOTE_SESSION" \
  "NUMERAI_SEARCH_CONFIG='$SEARCH' bash '$PROJECT/slurm/promote-final-selection-when-ready.sh' '$LAST_SUBMITTED' '$MANIFEST'"
printf '%s launched final canonical selection union=%s dependency=%s last_submitted=%s\n' \
  "$(date -Is)" "$EXPECTED" "$DEPENDENCY_JOB" "$LAST_SUBMITTED" \
  >> "$PROJECT/results/launch-final-selection.log"
