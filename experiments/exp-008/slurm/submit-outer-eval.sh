#!/bin/bash
set -euo pipefail

if [[ $# -ne 5 ]]; then
  echo "usage: $0 SELECTION_JSON OUTER_SPLIT UPDATES DEPENDENCY_JOB_ID SEEDS_CSV" >&2
  exit 2
fi
SELECTION=$1
SPLIT_NAME=$2
UPDATES=$3
DEPENDENCY=$4
IFS=',' read -ra SEEDS <<< "$5"
PROJECT=/mnt/nw/home/t.buckworth/numerai-competitive

if [[ ! $SPLIT_NAME =~ ^outer_[123]$ ]]; then
  echo "outer split must be outer_1, outer_2 or outer_3" >&2
  exit 1
fi
if [[ ! $UPDATES =~ ^[0-9]+$ || $UPDATES -ne 100000 \
    || ${SEEDS[*]} != "0 1 2" || ! $DEPENDENCY =~ ^[0-9]+$ ]]; then
  echo "outer evaluation requires 100000 updates, seeds 0,1,2 and a numeric dependency" >&2
  exit 1
fi
declare -A WINNERS=()
for TASK_ARM in adamw spectral; do
  mapfile -t IDS < <(python3 -c \
    'import json,sys; print(*json.load(open(sys.argv[1]))["selected"][sys.argv[2]], sep="\n")' \
    "$SELECTION" "$TASK_ARM")
  if [[ ${#IDS[@]} -ne 1 ]]; then
    echo "$TASK_ARM selection must contain exactly one config ID" >&2
    exit 1
  fi
  if [[ ! ${IDS[0]} =~ ^[0-9]+$ ]]; then
    echo "$TASK_ARM selection contains a non-numeric config ID" >&2
    exit 1
  fi
  WINNERS[$TASK_ARM]=${IDS[0]}
done
for TASK_ARM in adamw spectral; do
  CONFIG_ID=${WINNERS[$TASK_ARM]}
  for SEED in "${SEEDS[@]}"; do
    TASK_NAME="stage-${SPLIT_NAME}-u${UPDATES}-s${SEED}-${TASK_ARM}-c${CONFIG_ID}"
    JOB_ID=$(sbatch --parsable --job-name="n8-out-${TASK_ARM:0:1}-${SEED}" \
      --dependency="afterok:${DEPENDENCY}" \
      --export="ALL,TASK_NAME=${TASK_NAME},TASK_ARM=${TASK_ARM},CONFIG_ID=${CONFIG_ID},SPLIT_NAME=${SPLIT_NAME},UPDATES=${UPDATES},SEED=${SEED}" \
      "$PROJECT/slurm/run-one.sbatch")
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$JOB_ID" "$SPLIT_NAME" "$UPDATES" "$SEED" "$TASK_ARM" "$CONFIG_ID"
  done
done
