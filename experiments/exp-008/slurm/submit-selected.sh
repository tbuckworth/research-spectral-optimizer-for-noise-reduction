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
REUSE_COMPLETE=${NUMERAI_REUSE_COMPLETE:-0}
if [[ $REUSE_COMPLETE != 0 && $REUSE_COMPLETE != 1 ]]; then
  echo "NUMERAI_REUSE_COMPLETE must be 0 or 1" >&2
  exit 1
fi

for SPLIT_NAME in "$@"; do
  for TASK_ARM in adamw spectral; do
    mapfile -t IDS < <(python3 -c \
      'import json,sys; print(*json.load(open(sys.argv[1]))["selected"]["paired_union"], sep="\n")' \
      "$SELECTION")
    for CONFIG_ID in "${IDS[@]}"; do
      TASK_NAME="stage-${SPLIT_NAME}-u${UPDATES}-s${SEED}-${TASK_ARM}-c${CONFIG_ID}"
      RESULT="$PROJECT/results/$TASK_NAME/result.json"
      if [[ $REUSE_COMPLETE == 1 && -f $RESULT ]]; then
        JOB_ID=0
      else
        JOB_ID=$(sbatch --parsable --job-name="n8-${TASK_ARM:0:1}-${CONFIG_ID}" \
          --dependency="afterok:${DEPENDENCY}" \
          --export="ALL,TASK_NAME=${TASK_NAME},TASK_ARM=${TASK_ARM},CONFIG_ID=${CONFIG_ID},SPLIT_NAME=${SPLIT_NAME},UPDATES=${UPDATES},SEED=${SEED}" \
          "$PROJECT/slurm/run-one.sbatch")
      fi
      printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$JOB_ID" "$SPLIT_NAME" "$UPDATES" "$SEED" "$TASK_ARM" "$CONFIG_ID"
    done
  done
done
