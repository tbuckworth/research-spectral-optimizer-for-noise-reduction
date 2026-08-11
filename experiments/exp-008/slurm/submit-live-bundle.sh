#!/bin/bash
set -euo pipefail

if [[ $# -ne 1 || ! $1 =~ ^[0-9a-f]{40}$ ]]; then
  echo "usage: $0 FULL_40_CHARACTER_CODE_COMMIT" >&2
  exit 2
fi
CODE_COMMIT=$1
PROJECT=${NUMERAI_PROJECT:-/mnt/nw/home/t.buckworth/numerai-competitive}
FREEZE="$PROJECT/results/freeze.json"
EVALUATION="$PROJECT/results/official-validation/evaluation-complete.json"
BUNDLE="$PROJECT/results/live-bundle"
MANIFEST="$PROJECT/results/submission-live-bundle.tsv"
[[ -f $FREEZE && -f $EVALUATION ]] \
  || { echo "freeze or sealed evaluation marker is missing" >&2; exit 1; }
if [[ -e $BUNDLE || -e $MANIFEST || -e "${MANIFEST}.tmp" ]]; then
  echo "live bundle or submission manifest already exists" >&2
  exit 1
fi
read -r CANDIDATE_ARM CANDIDATE_CONFIG < <(python3 -c '
import json, sys
freeze = json.load(open(sys.argv[1]))
evaluation = json.load(open(sys.argv[2]))
arm = freeze.get("candidate_transform", {}).get("arm")
if (freeze.get("status") != "frozen" or freeze.get("code_commit") != sys.argv[3]
        or arm not in {"adamw", "spectral"}
        or evaluation.get("status") != "complete"):
    raise SystemExit("freeze, code commit, candidate arm, or evaluation marker is invalid")
print(arm, freeze["selected"][arm]["config_id"])
' "$FREEZE" "$EVALUATION" "$CODE_COMMIT")
for SEED in 0 1 2; do
  MODEL="$PROJECT/results/final-refit-u100000-s${SEED}-${CANDIDATE_ARM}-c${CANDIDATE_CONFIG}/model.pt"
  [[ -f $MODEL ]] || { echo "frozen candidate model missing: $MODEL" >&2; exit 1; }
done
BUILD_JOB=$(sbatch --parsable --job-name=n8-live-bundle \
  --export="ALL,CODE_COMMIT=${CODE_COMMIT},CANDIDATE_ARM=${CANDIDATE_ARM},CANDIDATE_CONFIG=${CANDIDATE_CONFIG}" \
  "$PROJECT/slurm/run-build-live-bundle.sbatch")
VALIDATE_JOB=$(sbatch --parsable --job-name=n8-live-validate \
  --dependency="afterok:${BUILD_JOB%%;*}" \
  "$PROJECT/slurm/run-validate-live-bundle.sbatch")
printf 'stage\tjob_id\tdependency\tcode_commit\tcandidate_arm\tcandidate_config\n' \
  > "${MANIFEST}.tmp"
printf 'build_live_bundle\t%s\t\t%s\t%s\t%s\n' \
  "$BUILD_JOB" "$CODE_COMMIT" "$CANDIDATE_ARM" "$CANDIDATE_CONFIG" \
  >> "${MANIFEST}.tmp"
printf 'validate_live_bundle\t%s\t%s\t%s\t%s\t%s\n' \
  "$VALIDATE_JOB" "${BUILD_JOB%%;*}" "$CODE_COMMIT" "$CANDIDATE_ARM" "$CANDIDATE_CONFIG" \
  >> "${MANIFEST}.tmp"
mv "${MANIFEST}.tmp" "$MANIFEST"
cat "$MANIFEST"
