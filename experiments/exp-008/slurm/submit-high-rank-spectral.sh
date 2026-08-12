#!/bin/bash
set -euo pipefail

if [[ $# -lt 5 ]]; then
  echo "usage: $0 SELECTION_JSON UPDATES SEED DEPENDENCY_JOB_ID SPLIT..." >&2
  exit 2
fi
SELECTION=$1
UPDATES=$2
SEED=$3
DEPENDENCY=$4
shift 4
PROJECT=${NUMERAI_PROJECT:-/mnt/nw/home/t.buckworth/numerai-competitive}
mapfile -t IDS < <(python3 -c \
  'import json,sys
value=json.load(open(sys.argv[1]))
ids=value.get("high_rank_spectral", value.get("selected", {}).get("high_rank_spectral", []))
print(*ids, sep="\n")' \
  "$SELECTION")
[[ ${#IDS[@]} -gt 0 ]] || { echo "selection has no high-rank spectral IDs" >&2; exit 1; }

for SPLIT_NAME in "$@"; do
  for CONFIG_ID in "${IDS[@]}"; do
    TASK_NAME="stage-${SPLIT_NAME}-u${UPDATES}-s${SEED}-spectral-c${CONFIG_ID}"
    JOB_ID=$(sbatch --parsable --job-name="n8-hs-${CONFIG_ID}" \
      --dependency="afterok:${DEPENDENCY}" \
      --export="ALL,TASK_NAME=${TASK_NAME},TASK_ARM=spectral,CONFIG_ID=${CONFIG_ID},SPLIT_NAME=${SPLIT_NAME},UPDATES=${UPDATES},SEED=${SEED}" \
      "$PROJECT/slurm/run-one.sbatch")
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$JOB_ID" "$SPLIT_NAME" "$UPDATES" "$SEED" spectral "$CONFIG_ID"
  done
done
