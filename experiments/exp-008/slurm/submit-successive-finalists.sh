#!/bin/bash
set -euo pipefail

if [[ $# -lt 6 ]]; then
  echo "usage: $0 FINALISTS_JSON ordinary|high-rank UPDATES SEED DEPENDENCY_JOB_ID SPLIT..." >&2
  exit 2
fi
SELECTION=$1
MODE=$2
UPDATES=$3
SEED=$4
DEPENDENCY=$5
shift 5
PROJECT=${NUMERAI_PROJECT:-/mnt/nw/home/t.buckworth/numerai-competitive}
REUSE_COMPLETE=${NUMERAI_REUSE_COMPLETE:-0}
if [[ ! -f $SELECTION || ! $UPDATES =~ ^(20000|100000)$ \
      || ! $SEED =~ ^[0-9]+$ || ! $DEPENDENCY =~ ^[0-9]+$ \
      || ( $MODE != ordinary && $MODE != high-rank ) \
      || ( $MODE == ordinary && $UPDATES != 100000 ) \
      || ( $REUSE_COMPLETE != 0 && $REUSE_COMPLETE != 1 ) ]]; then
  echo "invalid successive-finalist submission arguments" >&2
  exit 1
fi
if [[ $MODE == ordinary ]]; then
  mapfile -t IDS < <(python3 -c '
import json,sys
print(*json.load(open(sys.argv[1]))["ordinary_confirmation_paired_union"], sep="\n")
' "$SELECTION")
  ARMS=(adamw spectral)
else
  mapfile -t IDS < <(python3 -c '
import json,sys
print(*json.load(open(sys.argv[1]))["high_rank_spectral"], sep="\n")
' "$SELECTION")
  ARMS=(spectral)
fi
[[ ${#IDS[@]} -gt 0 ]] || { echo "successive-finalist selection is empty" >&2; exit 1; }

for SPLIT_NAME in "$@"; do
  for TASK_ARM in "${ARMS[@]}"; do
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
