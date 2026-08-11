#!/bin/bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 SUBMISSION_TSV DEPENDENCY_JOB_ID" >&2
  exit 2
fi
MANIFEST=$1
DEPENDENCY=$2
PROJECT=/mnt/nw/home/t.buckworth/numerai-competitive
RESUBMITTED=0

while IFS=$'\t' read -r _OLD_JOB SPLIT_NAME UPDATES SEED TASK_ARM CONFIG_ID EXTRA; do
  if [[ -n ${EXTRA:-} || -z ${CONFIG_ID:-} ]]; then
    echo "invalid six-column stage manifest row" >&2
    exit 1
  fi
  TASK_NAME="stage-${SPLIT_NAME}-u${UPDATES}-s${SEED}-${TASK_ARM}-c${CONFIG_ID}"
  DIRECTORY="$PROJECT/results/$TASK_NAME"
  if [[ -f "$DIRECTORY/result.json" ]]; then
    continue
  fi
  if [[ ! -f "$DIRECTORY/checkpoint-status.json" || ! -f "$DIRECTORY/checkpoint.pt" ]]; then
    echo "$TASK_NAME is incomplete without a restart checkpoint" >&2
    exit 1
  fi
  JOB_ID=$(sbatch --parsable --job-name="n8-${TASK_ARM:0:1}-${CONFIG_ID}r" \
    --dependency="afterok:${DEPENDENCY}" \
    --export="ALL,TASK_NAME=${TASK_NAME},TASK_ARM=${TASK_ARM},CONFIG_ID=${CONFIG_ID},SPLIT_NAME=${SPLIT_NAME},UPDATES=${UPDATES},SEED=${SEED}" \
    "$PROJECT/slurm/run-one.sbatch")
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$JOB_ID" "$SPLIT_NAME" "$UPDATES" "$SEED" "$TASK_ARM" "$CONFIG_ID"
  RESUBMITTED=$((RESUBMITTED + 1))
done < "$MANIFEST"

if [[ $RESUBMITTED -eq 0 ]]; then
  echo "no checkpointed tasks require resubmission" >&2
fi
