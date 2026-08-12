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
FINALISTS="$RESULTS/selection-outer_1-successive-finalists.json"
MANIFEST="$RESULTS/submission-outer_1-f2b-successive.tsv"
SEARCH="$RESULTS/search-v1-high-rank.json"
LOG="$RESULTS/promote-outer_1-f2a.log"
SPLITS=(outer_1_inner_1 outer_1_inner_2)
MARKERS=()
for SPLIT in "${SPLITS[@]}"; do
  for BUDGET in 5000 20000; do
    for SEED in 0 1 2; do
      MARKERS+=("$RESULTS/summary-f2a-${SPLIT}-u${BUDGET}-s${SEED}/summary-complete.json")
    done
  done
done
MARKERS+=("$RESULTS/summary-f2a-${SPLITS[0]}-u100000-s0/summary-complete.json")

while true; do
  MISSING=0
  for MARKER in "${MARKERS[@]}"; do [[ -f $MARKER ]] || MISSING=$((MISSING + 1)); done
  [[ $MISSING -eq 0 ]] && break
  if ! tmux has-session -t numerai-f2a-supervisor 2>/dev/null; then
    echo "phase-A supervisor exited without all exact summaries" >&2
    exit 1
  fi
  printf '%s waiting for %s phase-A summaries\n' "$(date -Is)" "$MISSING" >> "$LOG"
  sleep 60
done

if [[ -e $FINALISTS || -e $MANIFEST || -e ${MANIFEST}.tmp ]] \
      || tmux has-session -t numerai-f2b-supervisor 2>/dev/null \
      || tmux has-session -t numerai-f2b-promote 2>/dev/null; then
  echo "successive-halving phase-B artifact or session already exists" >&2
  exit 1
fi
mapfile -t HIGH_IDS < <(python3 -c \
  'import json,sys; print(*json.load(open(sys.argv[1]))["high_rank_spectral"], sep="\n")' "$PLAN")
HIGH_TOP=2
[[ ${#HIGH_IDS[@]} -ge 2 ]] || HIGH_TOP=1
SOURCE_ID=$(python3 -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["high_rank_source_config_id"])' "$PLAN")
HIGH_ARGS=()
for CONFIG_ID in "${HIGH_IDS[@]}"; do HIGH_ARGS+=(--high-rank-id "$CONFIG_ID"); done
SCORE_ARGS=()
for SPLIT in "${SPLITS[@]}"; do
  SCORE_ARGS+=(--high-rank-score "$RESULTS/summary-f2a-${SPLIT}-u20000-s0/scores.csv")
done
"$PROJECT/uv" run --no-sync python -m numerai_competitive.select_successive_finalists \
  --ordinary-score "$RESULTS/summary-f2a-${SPLITS[0]}-u100000-s0/scores.csv" \
  "${SCORE_ARGS[@]}" "${HIGH_ARGS[@]}" --high-rank-source-config-id "$SOURCE_ID" \
  --high-rank-top "$HIGH_TOP" --output "$FINALISTS"

TEMPORARY="${MANIFEST}.tmp"
: > "$TEMPORARY"
export NUMERAI_REUSE_COMPLETE=1
export NUMERAI_SEARCH_CONFIG="$SEARCH"
for SEED in 1 2; do
  bash "$PROJECT/slurm/submit-successive-plan.sh" \
    "$PLAN" confirmation 20000 "$SEED" "$DEPENDENCY_JOB" "${SPLITS[@]}" >> "$TEMPORARY"
  bash "$PROJECT/slurm/submit-successive-finalists.sh" \
    "$FINALISTS" high-rank 20000 "$SEED" "$DEPENDENCY_JOB" "${SPLITS[@]}" >> "$TEMPORARY"
done
for SEED in 0 1 2; do
  bash "$PROJECT/slurm/submit-successive-finalists.sh" \
    "$FINALISTS" ordinary 100000 "$SEED" "$DEPENDENCY_JOB" "${SPLITS[@]}" >> "$TEMPORARY"
  bash "$PROJECT/slurm/submit-successive-finalists.sh" \
    "$FINALISTS" high-rank 100000 "$SEED" "$DEPENDENCY_JOB" "${SPLITS[@]}" >> "$TEMPORARY"
done
mv "$TEMPORARY" "$MANIFEST"

CONFIRM_20=$(python3 -c \
  'import json,sys; print(len(json.load(open(sys.argv[1]))["confirmation_selections"]["20000"]["paired_union"]))' \
  "$PLAN")
ORDINARY=$(python3 -c \
  'import json,sys; print(len(json.load(open(sys.argv[1]))["ordinary_confirmation_paired_union"]))' \
  "$FINALISTS")
HIGH=$(python3 -c \
  'import json,sys; print(len(json.load(open(sys.argv[1]))["high_rank_spectral"]))' "$FINALISTS")
EXPECTED_ROWS=$((CONFIRM_20 * 8 + HIGH * 4 + ORDINARY * 12 + HIGH * 6))
if [[ $(wc -l < "$MANIFEST") -ne $EXPECTED_ROWS \
      || $(cut -f2-6 "$MANIFEST" | sort -u | wc -l) -ne $EXPECTED_ROWS ]]; then
  echo "phase-B manifest differs from frozen finalist coverage" >&2
  exit 1
fi
LAST_JOB=$(awk -F $'\t' '$1 != 0 {value=$1} END {sub(/;.*/, "", value); print value}' "$MANIFEST")
[[ $LAST_JOB =~ ^[0-9]+$ ]] || { echo "phase-B manifest has no submitted job" >&2; exit 1; }
tmux new-session -d -s numerai-f2b-supervisor \
  "NUMERAI_SEARCH_CONFIG='$SEARCH' NUMERAI_SUMMARY_PREFIX=f2b- bash '$PROJECT/slurm/supervise-resumable-stage.sh' '$MANIFEST'"
tmux new-session -d -s numerai-f2b-promote \
  "NUMERAI_SEARCH_CONFIG='$SEARCH' bash '$PROJECT/slurm/promote-successive-finalists-when-ready.sh' '$LAST_JOB'"
printf '%s submitted successive phase B rows=%s dependency=%s\n' \
  "$(date -Is)" "$EXPECTED_ROWS" "$DEPENDENCY_JOB" >> "$LOG"
