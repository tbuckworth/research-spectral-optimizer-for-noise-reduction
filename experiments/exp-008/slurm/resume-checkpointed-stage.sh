#!/bin/bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 SUBMISSION_TSV DEPENDENCY_JOB_ID" >&2
  exit 2
fi
MANIFEST=$1
DEPENDENCY=$2
PROJECT=${NUMERAI_PROJECT:-/mnt/nw/home/t.buckworth/numerai-competitive}
PROBE_MODE=${NUMERAI_PROBE_MODE:-0}
if [[ $PROBE_MODE != 0 && $PROBE_MODE != 1 ]]; then
  echo "NUMERAI_PROBE_MODE must be 0 or 1" >&2
  exit 1
fi
RUNNER="$PROJECT/slurm/run-one.sbatch"
[[ $PROBE_MODE == 1 ]] && RUNNER="$PROJECT/slurm/run-high-rank-probe.sbatch"
RESUBMITTED=0

while IFS=$'\t' read -r _OLD_JOB SPLIT_NAME UPDATES SEED TASK_ARM CONFIG_ID EXTRA; do
  if [[ -n ${EXTRA:-} || -z ${CONFIG_ID:-} ]]; then
    echo "invalid six-column stage manifest row" >&2
    exit 1
  fi
  if [[ $PROBE_MODE == 1 && $TASK_ARM != spectral ]]; then
    echo "high-rank probe retry manifest contains a non-spectral arm" >&2
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
    "$RUNNER")
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$JOB_ID" "$SPLIT_NAME" "$UPDATES" "$SEED" "$TASK_ARM" "$CONFIG_ID"
  RESUBMITTED=$((RESUBMITTED + 1))
done < "$MANIFEST"

if [[ $RESUBMITTED -eq 0 ]]; then
  echo "no checkpointed tasks require resubmission" >&2
fi
