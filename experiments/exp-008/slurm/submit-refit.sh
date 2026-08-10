#!/bin/bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "usage: $0 SELECTION_JSON UPDATES DEPENDENCY_JOB_ID SEEDS_CSV" >&2
  exit 2
fi
SELECTION=$1
UPDATES=$2
DEPENDENCY=$3
IFS=',' read -ra SEEDS <<< "$4"
PROJECT=/mnt/nw/home/t.buckworth/numerai-competitive

for TASK_ARM in adamw spectral; do
  mapfile -t IDS < <(python3 -c \
    'import json,sys; print(*json.load(open(sys.argv[1]))["selected"][sys.argv[2]], sep="\n")' \
    "$SELECTION" "$TASK_ARM")
  if [[ ${#IDS[@]} -ne 1 ]]; then
    echo "$TASK_ARM selection must contain exactly one config ID" >&2
    exit 1
  fi
  CONFIG_ID=${IDS[0]}
  for SEED in "${SEEDS[@]}"; do
    TASK_NAME="final-refit-u${UPDATES}-s${SEED}-${TASK_ARM}-c${CONFIG_ID}"
    JOB_ID=$(sbatch --parsable --job-name="n8-ref-${TASK_ARM:0:1}-${SEED}" \
      --dependency="afterok:${DEPENDENCY}" \
      --export="ALL,TASK_NAME=${TASK_NAME},TASK_ARM=${TASK_ARM},CONFIG_ID=${CONFIG_ID},UPDATES=${UPDATES},SEED=${SEED}" \
      "$PROJECT/slurm/run-refit.sbatch")
    printf '%s\t%s\t%s\t%s\t%s\n' "$JOB_ID" "$TASK_ARM" "$CONFIG_ID" "$UPDATES" "$SEED"
  done
done
