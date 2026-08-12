#!/bin/bash
set -euo pipefail

if [[ $# -ne 2 || ! $1 =~ ^[0-9a-f]{40}$ || ! $2 =~ ^[0-9]+$ ]]; then
  echo "usage: $0 FULL_40_CHARACTER_PRODUCTION_COMMIT ENVIRONMENT_JOB_ID" >&2
  exit 2
fi
PRODUCTION_CODE_COMMIT=$1
DEPENDENCY=$2
PROJECT=${NUMERAI_PROJECT:-/mnt/nw/home/t.buckworth/numerai-competitive}
FREEZE=$PROJECT/results/freeze.json
EVALUATION=$PROJECT/results/official-validation/evaluation-complete.json
PRODUCTION_SNAPSHOT=$PROJECT/production-code-snapshot.json
MANIFEST=$PROJECT/results/submission-production-refits.tsv
[[ -f $FREEZE && -f $EVALUATION && -f $PRODUCTION_SNAPSHOT ]] \
  || { echo "freeze/evaluation/production code snapshot missing" >&2; exit 1; }
[[ ! -e $MANIFEST && ! -e $PROJECT/results/production-refit-audit.json ]] \
  || { echo "production refit submission already exists" >&2; exit 1; }
"$PROJECT/uv" run --no-sync python -m numerai_competitive.code_snapshot verify \
  --root "$PROJECT" --snapshot "$PRODUCTION_SNAPSHOT" --commit "$PRODUCTION_CODE_COMMIT"
read -r PROCEDURE_CODE_COMMIT CANDIDATE_ARM CANDIDATE_CONFIG CANDIDATE_UPDATES CANDIDATE_FEATURE_SET < <(python3 -c '
import json,sys
freeze=json.load(open(sys.argv[1])); arm=freeze.get("candidate_transform",{}).get("arm")
snapshot=json.load(open(sys.argv[2]))
if (freeze.get("status") != "frozen" or arm not in {"adamw","spectral"}
        or snapshot.get("status") != "complete" or snapshot.get("code_commit") != sys.argv[3]):
    raise SystemExit("invalid freeze/production code snapshot/candidate")
selected=freeze["selected"][arm]
print(freeze["code_commit"], arm, selected["config_id"], selected["updates"], selected["feature_set"])
' "$FREEZE" "$PRODUCTION_SNAPSHOT" "$PRODUCTION_CODE_COMMIT")
BUILD=$(sbatch --parsable --dependency="afterok:${DEPENDENCY}" \
  --export="ALL,NUMERAI_PROJECT=${PROJECT},PROCEDURE_CODE_COMMIT=${PROCEDURE_CODE_COMMIT},PRODUCTION_CODE_COMMIT=${PRODUCTION_CODE_COMMIT},CANDIDATE_FEATURE_SET=${CANDIDATE_FEATURE_SET}" \
  "$PROJECT/slurm/run-build-production-shard.sbatch")
printf 'stage\tjob_id\tdependency\tseed\n' > "${MANIFEST}.tmp"
printf 'build_production_shard\t%s\t%s\t\n' "$BUILD" "$DEPENDENCY" >> "${MANIFEST}.tmp"
REFITS=()
for SEED in 0 1 2; do
  TASK_NAME="production-refit-s${SEED}-${CANDIDATE_ARM}-c${CANDIDATE_CONFIG}"
  JOB=$(sbatch --parsable --job-name="n8-prod-${CANDIDATE_ARM:0:1}-${SEED}" \
    --dependency="afterok:${BUILD%%;*}" \
    --export="ALL,NUMERAI_PROJECT=${PROJECT},PRODUCTION_CODE_COMMIT=${PRODUCTION_CODE_COMMIT},CANDIDATE_FEATURE_SET=${CANDIDATE_FEATURE_SET},TASK_NAME=${TASK_NAME},SEED=${SEED}" \
    "$PROJECT/slurm/run-production-refit.sbatch")
  REFITS+=("${JOB%%;*}")
  printf 'production_refit\t%s\t%s\t%s\n' "$JOB" "${BUILD%%;*}" "$SEED" >> "${MANIFEST}.tmp"
done
DEPENDENCIES=$(IFS=:; echo "${REFITS[*]}")
AUDIT=$(sbatch --parsable --dependency="afterok:${DEPENDENCIES}" \
  --export="ALL,NUMERAI_PROJECT=${PROJECT},PRODUCTION_CODE_COMMIT=${PRODUCTION_CODE_COMMIT},CANDIDATE_FEATURE_SET=${CANDIDATE_FEATURE_SET},CANDIDATE_ARM=${CANDIDATE_ARM},CANDIDATE_CONFIG=${CANDIDATE_CONFIG}" \
  "$PROJECT/slurm/run-audit-production-refits.sbatch")
printf 'audit_production_refits\t%s\t%s\t\n' "$AUDIT" "$DEPENDENCIES" >> "${MANIFEST}.tmp"
mv "${MANIFEST}.tmp" "$MANIFEST"
cat "$MANIFEST"
