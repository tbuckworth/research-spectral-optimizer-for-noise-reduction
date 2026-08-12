#!/bin/bash
set -euo pipefail

if [[ $# -ne 1 || ! $1 =~ ^[0-9]+$ ]]; then
  echo "usage: $0 DEPENDENCY_JOB_ID" >&2
  exit 2
fi
DEPENDENCY=$1
PROJECT=${NUMERAI_PROJECT:-/mnt/nw/home/t.buckworth/numerai-competitive}
RESULTS="$PROJECT/results"
PLAN="$RESULTS/selection-one-day-f0-top1.json"
SELECTION="$RESULTS/selection-one-day-confirmed-top1.json"
MANIFEST="$RESULTS/submission-one-day-outer1.tsv"
AUDIT="$RESULTS/audit-one-day-outer1"
LOG="$RESULTS/one-day-decision.log"
SPLITS=(outer_1_inner_1 outer_1_inner_2)
SCORES=()
MARKERS=()
for SPLIT in "${SPLITS[@]}"; do
  for SEED in 0 1; do
    ROOT="$RESULTS/summary-one-day-${SPLIT}-u20000-s${SEED}"
    SCORES+=("$ROOT/scores.csv")
    MARKERS+=("$ROOT/summary-complete.json")
  done
done

while true; do
  MISSING=0
  for MARKER in "${MARKERS[@]}"; do [[ -f $MARKER ]] || MISSING=$((MISSING + 1)); done
  [[ $MISSING -eq 0 ]] && break
  if ! tmux has-session -t numerai-one-day-confirm 2>/dev/null; then
    echo "bounded confirmation supervisor exited without exact summaries" >&2
    exit 1
  fi
  printf '%s waiting for %s confirmation summaries\n' "$(date -Is)" "$MISSING" >> "$LOG"
  sleep 60
done

if [[ -e $SELECTION || -e $MANIFEST || -e ${MANIFEST}.tmp || -e $AUDIT ]] \
    || tmux has-session -t numerai-outer1-supervisor 2>/dev/null \
    || tmux has-session -t numerai-one-day-outer-audit 2>/dev/null; then
  echo "one-day outer artifacts or sessions already exist" >&2
  exit 1
fi
cd "$PROJECT"
"$PROJECT/uv" run --no-sync python -m numerai_competitive.select_configs \
  --scores "${SCORES[@]}" --top 1 --output "${SELECTION}.tmp"
python3 - "$PLAN" "${SELECTION}.tmp" "$SELECTION" <<'PY'
import json, os, sys
plan = json.load(open(sys.argv[1]))
value = json.load(open(sys.argv[2]))
if set(value["selected"]["paired_union"]) - set(plan["selected"]["paired_union"]):
    raise SystemExit("confirmed winner is outside frozen F0 paired union")
value.update({
    "status": "one_day_development_selection_frozen",
    "selected_updates": {"adamw": [20000], "spectral": [20000]},
    "outer_reselection_allowed": False,
})
with open(sys.argv[3] + ".write", "w") as handle:
    json.dump(value, handle, indent=2, sort_keys=True)
    handle.write("\n")
os.replace(sys.argv[3] + ".write", sys.argv[3])
os.unlink(sys.argv[2])
PY

bash "$PROJECT/slurm/submit-outer-eval.sh" "$SELECTION" outer_1 selected \
  "$DEPENDENCY" 0,1,2 > "${MANIFEST}.tmp"
mv "${MANIFEST}.tmp" "$MANIFEST"
if [[ $(wc -l < "$MANIFEST") -ne 6 \
      || $(cut -f2-6 "$MANIFEST" | sort -u | wc -l) -ne 6 ]]; then
  echo "one-day outer manifest must contain two arms x three seeds" >&2
  exit 1
fi
tmux new-session -d -s numerai-outer1-supervisor \
  "bash '$PROJECT/slurm/supervise-resumable-stage.sh' '$MANIFEST' --skip-summary"
tmux new-session -d -s numerai-one-day-outer-audit \
  "bash '$PROJECT/slurm/audit-outer-when-ready.sh' '$MANIFEST' '$SELECTION' outer_1 '$AUDIT'"
printf '%s froze bounded winners and launched six untouched outer cells\n' \
  "$(date -Is)" >> "$LOG"
