#!/bin/bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "usage: $0 SPLIT UPDATES SEED DEPENDENCY_JOB_ID" >&2
  exit 2
fi
SPLIT_NAME=$1
UPDATES=$2
SEED=$3
DEPENDENCY=$4
PROJECT=${NUMERAI_PROJECT:-/mnt/nw/home/t.buckworth/numerai-competitive}
if [[ ! $SPLIT_NAME =~ ^outer_[123]_inner_[1-4]$ \
    || ! $UPDATES =~ ^[0-9]+$ || ! $SEED =~ ^[0-9]+$ \
    || ! $DEPENDENCY =~ ^[0-9]+$ ]]; then
  echo "invalid split, update count, seed or dependency" >&2
  exit 1
fi

for CONFIG_ID in $(seq 0 39); do
  for TASK_ARM in adamw spectral; do
    TASK_NAME="stage-${SPLIT_NAME}-u${UPDATES}-s${SEED}-${TASK_ARM}-c${CONFIG_ID}"
    JOB_ID=$(sbatch --parsable --job-name="n8-${TASK_ARM:0:1}-${CONFIG_ID}" \
      --dependency="afterok:${DEPENDENCY}" \
      --export="ALL,TASK_NAME=${TASK_NAME},TASK_ARM=${TASK_ARM},CONFIG_ID=${CONFIG_ID},SPLIT_NAME=${SPLIT_NAME},UPDATES=${UPDATES},SEED=${SEED}" \
      "$PROJECT/slurm/run-one.sbatch")
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$JOB_ID" "$SPLIT_NAME" "$UPDATES" "$SEED" "$TASK_ARM" "$CONFIG_ID"
  done
done
