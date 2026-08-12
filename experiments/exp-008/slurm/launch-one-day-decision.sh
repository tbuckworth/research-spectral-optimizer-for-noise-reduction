#!/bin/bash
set -euo pipefail

PROJECT=${NUMERAI_PROJECT:-/mnt/nw/home/t.buckworth/numerai-competitive}
RESULTS="$PROJECT/results"
F0="$RESULTS/summary-outer_1_inner_1-u5000-s0/scores.csv"
PLAN="$RESULTS/selection-one-day-f0-top1.json"
MANIFEST="$RESULTS/submission-one-day-confirm.tsv"
LOG="$RESULTS/one-day-decision.log"

if [[ ! -f $F0 ]]; then
  echo "complete F0 scores are required" >&2
  exit 1
fi
if [[ -e $PLAN || -e $MANIFEST || -e ${MANIFEST}.tmp ]] \
    || tmux has-session -t numerai-one-day-confirm 2>/dev/null \
    || tmux has-session -t numerai-one-day-promote 2>/dev/null; then
  echo "one-day decision artifacts or sessions already exist" >&2
  exit 1
fi

cd "$PROJECT"
"$PROJECT/uv" run --no-sync python -m numerai_competitive.select_configs \
  --scores "$F0" --top 1 --output "$PLAN"
python3 - "$PLAN" <<'PY'
import json, sys
value = json.load(open(sys.argv[1]))
if value["selected"] != {"adamw": [39], "spectral": [38], "paired_union": [38, 39]}:
    raise SystemExit("complete F0 nomination differs from frozen expected IDs 38/39")
PY

ENVIRONMENT=$(sbatch --parsable "$PROJECT/slurm/sync-env.sbatch")
: > "${MANIFEST}.tmp"
export NUMERAI_REUSE_COMPLETE=1
for SEED in 0 1; do
  bash "$PROJECT/slurm/submit-selected.sh" "$PLAN" 20000 "$SEED" "$ENVIRONMENT" \
    outer_1_inner_1 outer_1_inner_2 >> "${MANIFEST}.tmp"
done
mv "${MANIFEST}.tmp" "$MANIFEST"
if [[ $(wc -l < "$MANIFEST") -ne 16 \
      || $(cut -f2-6 "$MANIFEST" | sort -u | wc -l) -ne 16 ]]; then
  echo "one-day confirmation manifest must contain 16 unique cells" >&2
  exit 1
fi
LAST_JOB=$(awk -F $'\t' '$1 != 0 {value=$1} END {sub(/;.*/, "", value); print value}' "$MANIFEST")
[[ $LAST_JOB =~ ^[0-9]+$ ]] || { echo "confirmation manifest has no submitted job" >&2; exit 1; }
tmux new-session -d -s numerai-one-day-confirm \
  "NUMERAI_SUMMARY_PREFIX=one-day- bash '$PROJECT/slurm/supervise-resumable-stage.sh' '$MANIFEST'"
tmux new-session -d -s numerai-one-day-promote \
  "bash '$PROJECT/slurm/promote-one-day-decision.sh' '$LAST_JOB'"
printf '%s launched bounded confirmation environment=%s rows=16\n' \
  "$(date -Is)" "$ENVIRONMENT" >> "$LOG"
printf 'environment\t%s\nlast_job\t%s\n' "$ENVIRONMENT" "$LAST_JOB"
