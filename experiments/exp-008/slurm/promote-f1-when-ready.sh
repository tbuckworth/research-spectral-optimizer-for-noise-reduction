#!/bin/bash
set -euo pipefail

if [[ $# -ne 1 || ! $1 =~ ^[0-9]+$ ]]; then
  echo "usage: $0 DEPENDENCY_JOB_ID" >&2
  exit 2
fi
DEPENDENCY_JOB=$1
PROJECT=${NUMERAI_PROJECT:-/mnt/nw/home/t.buckworth/numerai-competitive}
RESULTS="$PROJECT/results"
PLAN="$RESULTS/selection-outer_1-successive-plan.json"
MANIFEST="$RESULTS/submission-outer_1-f2a-successive.tsv"
LOG="$RESULTS/promote-outer_1-f1.log"
F0_SUMMARY="$RESULTS/summary-outer_1_inner_1-u5000-s0/scores.csv"
SPLITS=(outer_1_inner_1 outer_1_inner_2)
F1_SUMMARIES=()
F1_MARKERS=()
for SPLIT in "${SPLITS[@]}"; do
  F1_SUMMARIES+=("$RESULTS/summary-${SPLIT}-u20000-s0/scores.csv")
  F1_MARKERS+=("$RESULTS/summary-${SPLIT}-u20000-s0/summary-complete.json")
done

while true; do
  MISSING=0
  for INDEX in "${!F1_MARKERS[@]}"; do
    if [[ ! -f ${F1_MARKERS[$INDEX]} ]]; then
      MISSING=$((MISSING + 1))
      if ! tmux has-session -t "numerai-f1-inner$((INDEX + 1))" 2>/dev/null; then
        echo "an F1 supervisor exited without an audited completion marker" >&2
        exit 1
      fi
    fi
  done
  [[ $MISSING -eq 0 ]] && break
  printf '%s waiting for all audited F1 summaries\n' "$(date -Is)" >> "$LOG"
  sleep 60
done

cd "$PROJECT"
if [[ -e $PLAN || -e $MANIFEST || -e ${MANIFEST}.tmp \
      || -e $RESULTS/search-v1-high-rank.json ]] \
      || tmux has-session -t numerai-f2a-supervisor 2>/dev/null \
      || tmux has-session -t numerai-f2a-promote 2>/dev/null; then
  echo "successive-halving phase-A artifact or session already exists" >&2
  exit 1
fi
bash "$PROJECT/slurm/prepare-high-rank-search.sh" \
  "$DEPENDENCY_JOB" "${F1_SUMMARIES[@]}"
SEARCH="$RESULTS/search-v1-high-rank.json"
EXTENSION="$RESULTS/search-v1-high-rank-extension.json"
SOURCE_ID=$(python3 -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["source_config_id"])' "$EXTENSION")
[[ $SOURCE_ID =~ ^[0-9]+$ ]] || { echo "invalid high-rank source ID" >&2; exit 1; }
"$PROJECT/uv" run --no-sync python -m numerai_competitive.select_successive_halving \
  --score-group "$F0_SUMMARY" --score-group "${F1_SUMMARIES[@]}" \
  --confirmation-top 2 --long-scout-top 1 \
  --high-rank-source-config-id "$SOURCE_ID" --augmented-search "$SEARCH" \
  --output "$PLAN"

TEMPORARY="${MANIFEST}.tmp"
: > "$TEMPORARY"
export NUMERAI_REUSE_COMPLETE=1
export NUMERAI_SEARCH_CONFIG="$SEARCH"
for BUDGET in 5000 20000; do
  for SEED in 0 1 2; do
    bash "$PROJECT/slurm/submit-successive-plan.sh" \
      "$PLAN" confirmation "$BUDGET" "$SEED" "$DEPENDENCY_JOB" "${SPLITS[@]}" \
      >> "$TEMPORARY"
  done
done
bash "$PROJECT/slurm/submit-high-rank-spectral.sh" \
  "$PLAN" 20000 0 "$DEPENDENCY_JOB" "${SPLITS[@]}" >> "$TEMPORARY"
bash "$PROJECT/slurm/submit-successive-plan.sh" \
  "$PLAN" long-scout 100000 0 "$DEPENDENCY_JOB" "${SPLITS[0]}" >> "$TEMPORARY"
mv "$TEMPORARY" "$MANIFEST"

CONFIRM_5=$(python3 -c \
  'import json,sys; print(len(json.load(open(sys.argv[1]))["confirmation_selections"]["5000"]["paired_union"]))' \
  "$PLAN")
CONFIRM_20=$(python3 -c \
  'import json,sys; print(len(json.load(open(sys.argv[1]))["confirmation_selections"]["20000"]["paired_union"]))' \
  "$PLAN")
LONG=$(python3 -c \
  'import json,sys; print(len(json.load(open(sys.argv[1]))["long_scout_paired_union"]))' "$PLAN")
HIGH=$(python3 -c \
  'import json,sys; print(len(json.load(open(sys.argv[1]))["high_rank_spectral"]))' "$PLAN")
EXPECTED_ROWS=$((CONFIRM_5 * 12 + CONFIRM_20 * 12 + HIGH * 2 + LONG * 2))
if [[ $(wc -l < "$MANIFEST") -ne $EXPECTED_ROWS \
      || $(cut -f2-6 "$MANIFEST" | sort -u | wc -l) -ne $EXPECTED_ROWS ]]; then
  echo "phase-A manifest differs from frozen successive-halving coverage" >&2
  exit 1
fi
LAST_JOB=$(awk -F $'\t' '$1 != 0 {value=$1} END {sub(/;.*/, "", value); print value}' "$MANIFEST")
[[ $LAST_JOB =~ ^[0-9]+$ ]] || { echo "phase-A manifest has no submitted job" >&2; exit 1; }
tmux new-session -d -s numerai-f2a-supervisor \
  "NUMERAI_SEARCH_CONFIG='$SEARCH' NUMERAI_SUMMARY_PREFIX=f2a- bash '$PROJECT/slurm/supervise-resumable-stage.sh' '$MANIFEST'"
tmux new-session -d -s numerai-f2a-promote \
  "NUMERAI_SEARCH_CONFIG='$SEARCH' bash '$PROJECT/slurm/promote-successive-scout-when-ready.sh' '$LAST_JOB'"
printf '%s submitted successive phase A rows=%s dependency=%s\n' \
  "$(date -Is)" "$EXPECTED_ROWS" "$DEPENDENCY_JOB" >> "$LOG"
