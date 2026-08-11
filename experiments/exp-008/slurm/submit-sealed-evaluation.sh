#!/bin/bash
set -euo pipefail

if [[ $# -ne 1 || ! $1 =~ ^[0-9a-f]{40}$ ]]; then
  echo "usage: $0 FULL_40_CHARACTER_CODE_COMMIT" >&2
  exit 2
fi
CODE_COMMIT=$1
PROJECT=${NUMERAI_PROJECT:-/mnt/nw/home/t.buckworth/numerai-competitive}
SEARCH=${NUMERAI_SEARCH_CONFIG:-$PROJECT/configs/search-v1-high-rank.json}
[[ -f $SEARCH ]] || { echo "audited high-rank search is missing" >&2; exit 1; }
SELECTION="$PROJECT/results/selection-final-top1.json"
AUDIT="$PROJECT/results/audit-final-refits-u100000/refit-audit.json"
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
read -r ADAMW_CONFIG SPECTRAL_CONFIG < <(python3 -c '
import json, sys
selection = json.load(open(sys.argv[1]))
audit = json.load(open(sys.argv[2]))
candidate = json.load(open(sys.argv[3]))
ids = []
for arm in ("adamw", "spectral"):
    values = selection.get("selected", {}).get(arm, [])
    if len(values) != 1 or audit.get("selected", {}).get(arm) != values[0]:
        raise SystemExit("final selection and refit audit disagree")
    ids.append(values[0])
if (audit.get("status") != "audit_complete" or audit.get("cells") != 6
        or audit.get("updates") != 100000 or audit.get("seeds") != [0, 1, 2]
        or candidate.get("status") != "frozen_train_only_selection"):
    raise SystemExit("refit audit or train-only candidate is incomplete")
print(*ids)
' "$SELECTION" "$AUDIT" "$CANDIDATE")
for ARM_CONFIG in "adamw:$ADAMW_CONFIG" "spectral:$SPECTRAL_CONFIG"; do
  ARM=${ARM_CONFIG%%:*}
  CONFIG=${ARM_CONFIG##*:}
  for SEED in 0 1 2; do
    MODEL="$PROJECT/results/final-refit-u100000-s${SEED}-${ARM}-c${CONFIG}/model.pt"
    [[ -f $MODEL ]] || { echo "selected refit model missing: $MODEL" >&2; exit 1; }
  done
done

BUILD_JOB=$(sbatch --parsable --job-name=n8-freeze-validation \
  --export="ALL,NUMERAI_SEARCH_CONFIG=${SEARCH},CODE_COMMIT=${CODE_COMMIT},ADAMW_CONFIG=${ADAMW_CONFIG},SPECTRAL_CONFIG=${SPECTRAL_CONFIG}" \
  "$PROJECT/slurm/run-freeze-build-validation.sbatch")
EVAL_JOB=$(sbatch --parsable --job-name=n8-sealed-eval \
  --dependency="afterok:${BUILD_JOB%%;*}" \
  --export="ALL,NUMERAI_SEARCH_CONFIG=${SEARCH},CODE_COMMIT=${CODE_COMMIT},ADAMW_CONFIG=${ADAMW_CONFIG},SPECTRAL_CONFIG=${SPECTRAL_CONFIG}" \
  "$PROJECT/slurm/run-sealed-evaluation.sbatch")
printf 'stage\tjob_id\tdependency\tcode_commit\tadamw_config\tspectral_config\n' \
  > "${MANIFEST}.tmp"
printf 'freeze_download_build\t%s\t\t%s\t%s\t%s\n' \
  "$BUILD_JOB" "$CODE_COMMIT" "$ADAMW_CONFIG" "$SPECTRAL_CONFIG" >> "${MANIFEST}.tmp"
printf 'official_validation\t%s\t%s\t%s\t%s\t%s\n' \
  "$EVAL_JOB" "${BUILD_JOB%%;*}" "$CODE_COMMIT" "$ADAMW_CONFIG" "$SPECTRAL_CONFIG" \
  >> "${MANIFEST}.tmp"
mv "${MANIFEST}.tmp" "$MANIFEST"
cat "$MANIFEST"
