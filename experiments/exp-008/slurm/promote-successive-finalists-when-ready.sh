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
ASSEMBLED="$RESULTS/summary-outer_1-successive-equal-coverage"
SELECTION="$RESULTS/selection-outer_1-f2-budget-top1.json"
SEARCH="$RESULTS/search-v1-high-rank.json"
LOG="$RESULTS/promote-outer_1-f2b.log"
SPLITS=(outer_1_inner_1 outer_1_inner_2)
MARKERS=()
for SPLIT in "${SPLITS[@]}"; do
  for SEED in 1 2; do
    MARKERS+=("$RESULTS/summary-f2b-${SPLIT}-u20000-s${SEED}/summary-complete.json")
  done
  for SEED in 0 1 2; do
    MARKERS+=("$RESULTS/summary-f2b-${SPLIT}-u100000-s${SEED}/summary-complete.json")
  done
done
while true; do
  MISSING=0
  for MARKER in "${MARKERS[@]}"; do [[ -f $MARKER ]] || MISSING=$((MISSING + 1)); done
  [[ $MISSING -eq 0 ]] && break
  if ! tmux has-session -t numerai-f2b-supervisor 2>/dev/null; then
    echo "phase-B supervisor exited without all exact summaries" >&2
    exit 1
  fi
  printf '%s waiting for %s phase-B summaries\n' "$(date -Is)" "$MISSING" >> "$LOG"
  sleep 60
done

if [[ -e $ASSEMBLED || -e $SELECTION ]]; then
  echo "equal-coverage summary or selected winner already exists" >&2
  exit 1
fi
SCORE_ARGS=()
for SPLIT in "${SPLITS[@]}"; do
  for SEED in 0 1 2; do
    SCORE_ARGS+=(--score "$RESULTS/summary-f2a-${SPLIT}-u5000-s${SEED}/scores.csv")
  done
  SCORE_ARGS+=(--score "$RESULTS/summary-f2a-${SPLIT}-u20000-s0/scores.csv")
  for SEED in 1 2; do
    SCORE_ARGS+=(--score "$RESULTS/summary-f2b-${SPLIT}-u20000-s${SEED}/scores.csv")
  done
  for SEED in 0 1 2; do
    SCORE_ARGS+=(--score "$RESULTS/summary-f2b-${SPLIT}-u100000-s${SEED}/scores.csv")
  done
done
"$PROJECT/uv" run --no-sync python -m numerai_competitive.assemble_successive_scores \
  "${SCORE_ARGS[@]}" --plan "$PLAN" --finalists "$FINALISTS" \
  --split "${SPLITS[0]}" --split "${SPLITS[1]}" --seed 0 --seed 1 --seed 2 \
  --output "$ASSEMBLED"
"$PROJECT/uv" run --no-sync python -m numerai_competitive.select_budgeted_configs \
  --scores "$ASSEMBLED/scores.csv" --top 1 --allow-asymmetric --output "$SELECTION"
bash "$PROJECT/slurm/launch-walkforward-evaluation.sh" "$SELECTION" "$DEPENDENCY_JOB"
printf '%s selected equal-coverage winners and launched walk-forward evaluation\n' \
  "$(date -Is)" >> "$LOG"
