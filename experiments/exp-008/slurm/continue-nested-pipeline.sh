#!/bin/bash
set -euo pipefail

if [[ $# -ne 0 ]]; then echo "usage: $0" >&2; exit 2; fi
PROJECT=${NUMERAI_PROJECT:-/mnt/nw/home/t.buckworth/numerai-competitive}
RESULTS="$PROJECT/results"
LOG="$RESULTS/continue-nested-pipeline.log"
POLL_SECONDS=${NUMERAI_POLL_SECONDS:-60}
SOURCE_SELECTION="$RESULTS/selection-outer_1-f2-budget-top1.json"
FINAL_SELECTION="$RESULTS/selection-final-top1.json"
FINAL_MANIFEST="$RESULTS/submission-final-refit-budgeted.tsv"
FINAL_AUDIT="$RESULTS/audit-final-refits-budgeted/refit-audit.json"
AUDITS=()
for NUMBER in 1 2 3; do AUDITS+=("$RESULTS/audit-outer_${NUMBER}-budgeted/outer-audit.json"); done

for NUMBER in 1 2 3; do
  AUDIT=${AUDITS[$((NUMBER - 1))]}
  until [[ -f $AUDIT ]] && python3 -c '
import json,sys
value=json.load(open(sys.argv[1]))
raise SystemExit(0 if value.get("status") == "audit_complete"
                 and value.get("split", {}).get("name") == sys.argv[2] else 1)
' "$AUDIT" "outer_${NUMBER}"; do
    printf '%s waiting for fixed-config outer_%s audit\n' "$(date -Is)" "$NUMBER" >> "$LOG"
    sleep "$POLL_SECONDS"
  done
done

if [[ ! -f $FINAL_SELECTION ]]; then
  [[ ! -e $FINAL_SELECTION && -f $SOURCE_SELECTION ]] \
    || { echo "source or final selection state is inconsistent" >&2; exit 1; }
  "$PROJECT/uv" run --no-sync python -m numerai_competitive.confirm_walkforward \
    --selection "$SOURCE_SELECTION" \
    --audit "${AUDITS[0]}" --audit "${AUDITS[1]}" --audit "${AUDITS[2]}" \
    --output "$FINAL_SELECTION"
  printf '%s confirmed fixed winner without outer-result reselection\n' "$(date -Is)" >> "$LOG"
fi

if [[ ! -f $RESULTS/nested-outer/nested-outer-report.json \
      || ! -f $RESULTS/candidate-plan.json ]]; then
  if [[ -e $RESULTS/nested-outer || -e $RESULTS/candidate-plan.json ]]; then
    echo "partial walk-forward aggregate or candidate output exists" >&2
    exit 1
  fi
  bash "$PROJECT/slurm/build-oof-candidate.sh"
  printf '%s built walk-forward report and frozen candidate blend\n' "$(date -Is)" >> "$LOG"
fi

if [[ ! -f $FINAL_AUDIT && ! -f $FINAL_MANIFEST ]]; then
  ENV_JOB=$(sbatch --parsable "$PROJECT/slurm/sync-env.sbatch")
  ENV_JOB=${ENV_JOB%%;*}
  [[ $ENV_JOB =~ ^[0-9]+$ ]] || { echo "invalid environment dependency" >&2; exit 1; }
  bash "$PROJECT/slurm/submit-refit.sh" \
    "$FINAL_SELECTION" selected "$ENV_JOB" 0,1,2 > "${FINAL_MANIFEST}.tmp"
  [[ $(wc -l < "${FINAL_MANIFEST}.tmp") -eq 6 ]] \
    || { echo "final refit manifest must contain six cells" >&2; exit 1; }
  mv "${FINAL_MANIFEST}.tmp" "$FINAL_MANIFEST"
  tmux new-session -d -s numerai-final-refit-supervisor \
    "NUMERAI_SEARCH_CONFIG='$RESULTS/search-v1-high-rank.json' bash '$PROJECT/slurm/supervise-refits.sh' '$FINAL_MANIFEST' '$FINAL_SELECTION'"
  printf '%s submitted final refits dependency=%s\n' "$(date -Is)" "$ENV_JOB" >> "$LOG"
fi
until [[ -f $FINAL_AUDIT ]] && python3 -c '
import json,sys
value=json.load(open(sys.argv[1]))
raise SystemExit(0 if value.get("status") == "audit_complete"
                 and value.get("cells") == 6 and value.get("seeds") == [0, 1, 2]
                 and set(value.get("updates", {})) == {"adamw", "spectral"} else 1)
' "$FINAL_AUDIT"; do
  printf '%s waiting for audited final refits\n' "$(date -Is)" >> "$LOG"
  sleep "$POLL_SECONDS"
done
printf '%s walk-forward pipeline ready for immutable freeze and sealed validation\n' \
  "$(date -Is)" >> "$LOG"
