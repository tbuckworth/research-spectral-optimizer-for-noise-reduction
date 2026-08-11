#!/bin/bash
set -euo pipefail

if [[ $# -ne 0 ]]; then
  echo "usage: $0" >&2
  exit 2
fi
PROJECT=${NUMERAI_PROJECT:-/mnt/nw/home/t.buckworth/numerai-competitive}
OOF="$PROJECT/results/nested-outer"
CANDIDATE="$PROJECT/results/candidate-plan.json"
ADAMW_RESULTS=()
SPECTRAL_RESULTS=()

if [[ -e "$OOF" || -e "$CANDIDATE" ]]; then
  echo "nested-outer output or candidate plan already exists" >&2
  exit 1
fi
for OUTER_NUMBER in 1 2 3; do
  SELECTION="$PROJECT/results/selection-outer_${OUTER_NUMBER}-f2-budget-top1.json"
  AUDIT="$PROJECT/results/audit-outer_${OUTER_NUMBER}-budgeted/outer-audit.json"
  if [[ ! -f $SELECTION || ! -f $AUDIT ]]; then
    echo "outer_${OUTER_NUMBER} selection or audit is missing" >&2
    exit 1
  fi
  for ARM in adamw spectral; do
    read -r CONFIG_ID UPDATES < <(python3 -c '
import json, sys
selection = json.load(open(sys.argv[1]))
audit = json.load(open(sys.argv[2]))
arm = sys.argv[3]
ids = selection.get("selected", {}).get(arm, [])
budgets = selection.get("selected_updates", {}).get(arm, [])
if (audit.get("status") != "audit_complete"
        or audit.get("split", {}).get("name") != sys.argv[4]
        or len(ids) != 1 or len(budgets) != 1
        or audit.get("selected", {}).get(arm) != ids[0]
        or audit.get("updates", {}).get(arm) != budgets[0]):
    raise SystemExit("selection and completed audit do not agree")
print(ids[0], budgets[0])
' "$SELECTION" "$AUDIT" "$ARM" "outer_${OUTER_NUMBER}")
    for SEED in 0 1 2; do
      RESULT="$PROJECT/results/stage-outer_${OUTER_NUMBER}-u${UPDATES}-s${SEED}-${ARM}-c${CONFIG_ID}/result.json"
      if [[ ! -f $RESULT ]]; then
        echo "audited outer result is missing: $RESULT" >&2
        exit 1
      fi
      if [[ $ARM == adamw ]]; then
        ADAMW_RESULTS+=(--adamw-result "$RESULT")
      else
        SPECTRAL_RESULTS+=(--spectral-result "$RESULT")
      fi
    done
  done
done

cd "$PROJECT"
"$PROJECT/uv" run --no-sync python -m numerai_competitive.oof \
  "${ADAMW_RESULTS[@]}" "${SPECTRAL_RESULTS[@]}" --output "$OOF"
"$PROJECT/uv" run --no-sync python -m numerai_competitive.candidate \
  --oof "$OOF/nested-outer-predictions.npz" --output "$CANDIDATE"
printf '%s built audited nested-outer estimate and frozen train-only candidate\n' \
  "$(date -Is)" >> "$PROJECT/results/build-oof-candidate.log"
