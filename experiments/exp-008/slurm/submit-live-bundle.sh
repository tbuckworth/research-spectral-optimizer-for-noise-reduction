#!/bin/bash
set -euo pipefail

if [[ $# -ne 1 || ! $1 =~ ^[0-9a-f]{40}$ ]]; then
  echo "usage: $0 FULL_40_CHARACTER_PRODUCTION_COMMIT" >&2
  exit 2
fi
PRODUCTION_CODE_COMMIT=$1
PROJECT=${NUMERAI_PROJECT:-/mnt/nw/home/t.buckworth/numerai-competitive}
FREEZE="$PROJECT/results/freeze.json"
EVALUATION="$PROJECT/results/official-validation/evaluation-complete.json"
PRODUCTION_AUDIT="$PROJECT/results/production-refit-audit.json"
BUNDLE="$PROJECT/results/live-bundle"
MANIFEST="$PROJECT/results/submission-live-bundle.tsv"
[[ -f $FREEZE && -f $EVALUATION && -f $PRODUCTION_AUDIT ]] \
  || { echo "freeze, sealed evaluation, or production audit is missing" >&2; exit 1; }
if [[ -e $BUNDLE || -e $MANIFEST || -e "${MANIFEST}.tmp" ]]; then
  echo "live bundle or submission manifest already exists" >&2
  exit 1
fi
read -r CANDIDATE_ARM CANDIDATE_CONFIG CANDIDATE_UPDATES < <(python3 -c '
import json, sys
freeze = json.load(open(sys.argv[1]))
evaluation = json.load(open(sys.argv[2]))
audit = json.load(open(sys.argv[3]))
arm = freeze.get("candidate_transform", {}).get("arm")
if (freeze.get("status") != "frozen" or audit.get("production_code_commit") != sys.argv[4]
        or arm not in {"adamw", "spectral"}
        or evaluation.get("status") != "complete" or audit.get("status") != "audit_complete"
        or audit.get("arm") != arm):
    raise SystemExit("freeze, code commit, evaluation, or production audit is invalid")
print(arm, freeze["selected"][arm]["config_id"], freeze["selected"][arm]["updates"])
' "$FREEZE" "$EVALUATION" "$PRODUCTION_AUDIT" "$PRODUCTION_CODE_COMMIT")
if [[ ! $CANDIDATE_UPDATES =~ ^(5000|20000|100000)$ ]]; then
  echo "frozen candidate update budget is invalid" >&2
  exit 1
fi
for SEED in 0 1 2; do
  MODEL="$PROJECT/results/production-refit-s${SEED}-${CANDIDATE_ARM}-c${CANDIDATE_CONFIG}/model.pt"
  [[ -f $MODEL ]] || { echo "production candidate model missing: $MODEL" >&2; exit 1; }
done
BUILD_JOB=$(sbatch --parsable --job-name=n8-live-bundle \
  --export="ALL,NUMERAI_PROJECT=${PROJECT},CODE_COMMIT=${PRODUCTION_CODE_COMMIT},CANDIDATE_ARM=${CANDIDATE_ARM},CANDIDATE_CONFIG=${CANDIDATE_CONFIG},CANDIDATE_UPDATES=${CANDIDATE_UPDATES}" \
  "$PROJECT/slurm/run-build-live-bundle.sbatch")
VALIDATE_JOB=$(sbatch --parsable --job-name=n8-live-validate \
  --dependency="afterok:${BUILD_JOB%%;*}" \
  --export="ALL,NUMERAI_PROJECT=${PROJECT}" \
  "$PROJECT/slurm/run-validate-live-bundle.sbatch")
printf 'stage\tjob_id\tdependency\tcode_commit\tcandidate_arm\tcandidate_config\tcandidate_updates\n' \
  > "${MANIFEST}.tmp"
printf 'build_live_bundle\t%s\t\t%s\t%s\t%s\t%s\n' \
  "$BUILD_JOB" "$PRODUCTION_CODE_COMMIT" "$CANDIDATE_ARM" "$CANDIDATE_CONFIG" "$CANDIDATE_UPDATES" \
  >> "${MANIFEST}.tmp"
printf 'validate_live_bundle\t%s\t%s\t%s\t%s\t%s\t%s\n' \
  "$VALIDATE_JOB" "${BUILD_JOB%%;*}" "$PRODUCTION_CODE_COMMIT" "$CANDIDATE_ARM" \
  "$CANDIDATE_CONFIG" "$CANDIDATE_UPDATES" \
  >> "${MANIFEST}.tmp"
mv "${MANIFEST}.tmp" "$MANIFEST"
cat "$MANIFEST"
