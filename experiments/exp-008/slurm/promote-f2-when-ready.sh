#!/bin/bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 DEPENDENCY_JOB_ID" >&2
  exit 2
fi
DEPENDENCY_JOB=$1
PROJECT=/mnt/nw/home/t.buckworth/numerai-competitive
SELECTION="$PROJECT/results/selection-outer_1-f2-top1.json"
MANIFEST="$PROJECT/results/submission-outer_1-eval-u100000.tsv"
OUTPUT="$PROJECT/results/audit-outer_1-u100000"
LOG="$PROJECT/results/promote-outer_1-f2.log"
SUMMARIES=()
MARKERS=()
for SPLIT in outer_1_inner_1 outer_1_inner_2; do
  for SEED in 0 1 2; do
    SUMMARIES+=("$PROJECT/results/summary-${SPLIT}-u100000-s${SEED}/scores.csv")
    MARKERS+=("$PROJECT/results/summary-${SPLIT}-u100000-s${SEED}/summary-complete.json")
  done
done

while true; do
  missing=0
  for marker in "${MARKERS[@]}"; do
    [[ -f "$marker" ]] || missing=$((missing + 1))
  done
  [[ $missing -eq 0 ]] && break
  if ! tmux has-session -t numerai-f2-supervisor 2>/dev/null; then
    echo "F2 supervisor exited without all audited completion markers" >&2
    exit 1
  fi
  printf '%s waiting for %s audited F2 summaries\n' "$(date -Is)" "$missing" >> "$LOG"
  sleep 60
done

cd "$PROJECT"
if [[ -e "$SELECTION" || -e "$MANIFEST" || -e "${MANIFEST}.tmp" ]] \
    || tmux has-session -t numerai-outer1-supervisor 2>/dev/null \
    || tmux has-session -t numerai-outer1-audit 2>/dev/null; then
  echo "outer selection, manifest, temporary file or supervisor session already exists" >&2
  exit 1
fi
"$PROJECT/uv" run --no-sync python -m numerai_competitive.select_configs \
  --scores "${SUMMARIES[@]}" --top 1 --output "$SELECTION"
TEMPORARY="${MANIFEST}.tmp"
bash "$PROJECT/slurm/submit-outer-eval.sh" \
  "$SELECTION" outer_1 100000 "$DEPENDENCY_JOB" 0,1,2 > "$TEMPORARY"
mv "$TEMPORARY" "$MANIFEST"
if [[ $(wc -l < "$MANIFEST") -ne 6 ]]; then
  echo "outer submission manifest must contain two arms x three seeds" >&2
  exit 1
fi
tmux new-session -d -s numerai-outer1-supervisor \
  "bash '$PROJECT/slurm/supervise-resumable-stage.sh' '$MANIFEST' --skip-summary"
tmux new-session -d -s numerai-outer1-audit \
  "bash '$PROJECT/slurm/audit-outer-when-ready.sh' '$MANIFEST' '$SELECTION' outer_1 '$OUTPUT'"
printf '%s submitted selected outer_1 evaluation dependency=%s\n' \
  "$(date -Is)" "$DEPENDENCY_JOB" >> "$LOG"
