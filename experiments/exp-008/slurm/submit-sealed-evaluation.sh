#!/bin/bash
set -euo pipefail

if [[ $# -ne 1 || ! $1 =~ ^[0-9a-f]{40}$ ]]; then
  echo "usage: $0 FULL_40_CHARACTER_CODE_COMMIT" >&2
  exit 2
fi
CODE_COMMIT=$1
PROJECT=${NUMERAI_PROJECT:-/mnt/nw/home/t.buckworth/numerai-competitive}
SEARCH=${NUMERAI_SEARCH_CONFIG:-$PROJECT/results/search-v1-high-rank.json}
[[ -f $SEARCH ]] || { echo "audited high-rank search is missing" >&2; exit 1; }
SELECTION="$PROJECT/results/selection-final-top1.json"
AUDIT="$PROJECT/results/audit-final-refits-budgeted/refit-audit.json"
CANDIDATE="$PROJECT/results/candidate-plan.json"
FREEZE="$PROJECT/results/freeze.json"
CODE_SNAPSHOT="$PROJECT/code-snapshot.json"
OUTPUT="$PROJECT/results/official-validation"
MANIFEST="$PROJECT/results/submission-sealed-evaluation.tsv"
for REQUIRED in "$SELECTION" "$AUDIT" "$CANDIDATE" "$CODE_SNAPSHOT"; do
  [[ -f $REQUIRED ]] || { echo "required frozen-train artifact missing: $REQUIRED" >&2; exit 1; }
done
python3 -c '
import json,sys
snapshot=json.load(open(sys.argv[1]))
raise SystemExit(0 if snapshot.get("status") == "complete"
                 and snapshot.get("code_commit") == sys.argv[2] else 1)
' "$CODE_SNAPSHOT" "$CODE_COMMIT" || {
  echo "code snapshot is incomplete or differs from requested commit" >&2
  exit 1
}
if [[ -e "$FREEZE" || -e "$OUTPUT" || -e "$MANIFEST" || -e "${MANIFEST}.tmp" ]]; then
  echo "freeze, validation output or sealed submission manifest already exists" >&2
  exit 1
fi
read -r ADAMW_CONFIG ADAMW_UPDATES SPECTRAL_CONFIG SPECTRAL_UPDATES < <(python3 -c '
import json, sys
selection = json.load(open(sys.argv[1]))
audit = json.load(open(sys.argv[2]))
candidate = json.load(open(sys.argv[3]))
values_out = []
for arm in ("adamw", "spectral"):
    values = selection.get("selected", {}).get(arm, [])
    budgets = selection.get("selected_updates", {}).get(arm, [])
    if len(values) != 1 or audit.get("selected", {}).get(arm) != values[0]:
        raise SystemExit("final selection and refit audit disagree")
    if (len(budgets) != 1 or budgets[0] not in {5000, 20000, 100000}
            or audit.get("updates", {}).get(arm) != budgets[0]):
        raise SystemExit("final selected and audited update budgets disagree")
    values_out.extend((values[0], budgets[0]))
if (audit.get("status") != "audit_complete" or audit.get("cells") != 6
        or audit.get("seeds") != [0, 1, 2]
        or candidate.get("status") != "frozen_train_only_selection"):
    raise SystemExit("refit audit or train-only candidate is incomplete")
print(*values_out)
' "$SELECTION" "$AUDIT" "$CANDIDATE")
for ARM_CONFIG_UPDATES in "adamw:$ADAMW_CONFIG:$ADAMW_UPDATES" \
    "spectral:$SPECTRAL_CONFIG:$SPECTRAL_UPDATES"; do
  IFS=: read -r ARM CONFIG UPDATES <<< "$ARM_CONFIG_UPDATES"
  for SEED in 0 1 2; do
    MODEL="$PROJECT/results/final-refit-u${UPDATES}-s${SEED}-${ARM}-c${CONFIG}/model.pt"
    [[ -f $MODEL ]] || { echo "selected refit model missing: $MODEL" >&2; exit 1; }
  done
done

BUILD_JOB=$(sbatch --parsable --job-name=n8-freeze-validation \
  --export="ALL,NUMERAI_SEARCH_CONFIG=${SEARCH},CODE_COMMIT=${CODE_COMMIT},ADAMW_CONFIG=${ADAMW_CONFIG},ADAMW_UPDATES=${ADAMW_UPDATES},SPECTRAL_CONFIG=${SPECTRAL_CONFIG},SPECTRAL_UPDATES=${SPECTRAL_UPDATES}" \
  "$PROJECT/slurm/run-freeze-build-validation.sbatch")
EVAL_JOB=$(sbatch --parsable --job-name=n8-sealed-eval \
  --dependency="afterok:${BUILD_JOB%%;*}" \
  --export="ALL,NUMERAI_SEARCH_CONFIG=${SEARCH},CODE_COMMIT=${CODE_COMMIT},ADAMW_CONFIG=${ADAMW_CONFIG},ADAMW_UPDATES=${ADAMW_UPDATES},SPECTRAL_CONFIG=${SPECTRAL_CONFIG},SPECTRAL_UPDATES=${SPECTRAL_UPDATES}" \
  "$PROJECT/slurm/run-sealed-evaluation.sbatch")
printf 'stage\tjob_id\tdependency\tcode_commit\tadamw_config\tadamw_updates\tspectral_config\tspectral_updates\n' \
  > "${MANIFEST}.tmp"
printf 'freeze_download_build\t%s\t\t%s\t%s\t%s\t%s\t%s\n' \
  "$BUILD_JOB" "$CODE_COMMIT" "$ADAMW_CONFIG" "$ADAMW_UPDATES" \
  "$SPECTRAL_CONFIG" "$SPECTRAL_UPDATES" >> "${MANIFEST}.tmp"
printf 'official_validation\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
  "$EVAL_JOB" "${BUILD_JOB%%;*}" "$CODE_COMMIT" "$ADAMW_CONFIG" "$ADAMW_UPDATES" \
  "$SPECTRAL_CONFIG" "$SPECTRAL_UPDATES" \
  >> "${MANIFEST}.tmp"
mv "${MANIFEST}.tmp" "$MANIFEST"
cat "$MANIFEST"
