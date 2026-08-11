#!/bin/bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 SUBMISSION_TSV DEPENDENCY_JOB_ID" >&2
  exit 2
fi
MANIFEST=$1
DEPENDENCY=$2
PROJECT=${NUMERAI_PROJECT:-/mnt/nw/home/t.buckworth/numerai-competitive}
RESUBMITTED=0

while IFS=$'\t' read -r _OLD_JOB TASK_ARM CONFIG_ID UPDATES SEED EXTRA; do
  if [[ -n ${EXTRA:-} || ! $TASK_ARM =~ ^(adamw|spectral)$ \
      || ! $CONFIG_ID =~ ^[0-9]+$ || ! $UPDATES =~ ^[0-9]+$ || $UPDATES -ne 100000 \
      || ! $SEED =~ ^[0-2]$ ]]; then
    echo "invalid five-column refit manifest row" >&2
    exit 1
  fi
  TASK_NAME="final-refit-u${UPDATES}-s${SEED}-${TASK_ARM}-c${CONFIG_ID}"
  DIRECTORY="$PROJECT/results/$TASK_NAME"
  if [[ -f "$DIRECTORY/result.json" && -f "$DIRECTORY/model.pt" ]]; then
    continue
  fi
  if [[ ! -f "$DIRECTORY/checkpoint-status.json" || ! -f "$DIRECTORY/checkpoint.pt" ]]; then
    echo "$TASK_NAME is incomplete without a restart checkpoint" >&2
    exit 1
  fi
  JOB_ID=$(sbatch --parsable --job-name="n8-ref-${TASK_ARM:0:1}-${SEED}r" \
    --dependency="afterok:${DEPENDENCY}" \
    --export="ALL,TASK_NAME=${TASK_NAME},TASK_ARM=${TASK_ARM},CONFIG_ID=${CONFIG_ID},UPDATES=${UPDATES},SEED=${SEED}" \
    "$PROJECT/slurm/run-refit.sbatch")
  printf '%s\t%s\t%s\t%s\t%s\n' \
    "$JOB_ID" "$TASK_ARM" "$CONFIG_ID" "$UPDATES" "$SEED"
  RESUBMITTED=$((RESUBMITTED + 1))
done < "$MANIFEST"

if [[ $RESUBMITTED -eq 0 ]]; then
  echo "no checkpointed refits require resubmission" >&2
fi
