#!/bin/bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "usage: $0 SELECTION_JSON UPDATES DEPENDENCY_JOB_ID SEEDS_CSV" >&2
  exit 2
fi
SELECTION=$1
UPDATES_ARGUMENT=$2
DEPENDENCY=$3
IFS=',' read -ra SEEDS <<< "$4"
PROJECT=${NUMERAI_PROJECT:-/mnt/nw/home/t.buckworth/numerai-competitive}
if [[ ( ! $UPDATES_ARGUMENT =~ ^[0-9]+$ && $UPDATES_ARGUMENT != selected ) \
    || ! $DEPENDENCY =~ ^[0-9]+$ || ${SEEDS[*]} != "0 1 2" ]]; then
  echo "final refit requires numeric or selected updates, seeds 0,1,2 and a dependency" >&2
  exit 1
fi

declare -A WINNERS=()
declare -A ARM_UPDATES=()
for TASK_ARM in adamw spectral; do
  mapfile -t IDS < <(python3 -c \
    'import json,sys; print(*json.load(open(sys.argv[1]))["selected"][sys.argv[2]], sep="\n")' \
    "$SELECTION" "$TASK_ARM")
  if [[ ${#IDS[@]} -ne 1 || ! ${IDS[0]} =~ ^[0-9]+$ ]]; then
    echo "$TASK_ARM selection must contain exactly one config ID" >&2
    exit 1
  fi
  WINNERS[$TASK_ARM]=${IDS[0]}
  if [[ $UPDATES_ARGUMENT == selected ]]; then
    mapfile -t BUDGETS < <(python3 -c \
      'import json,sys; print(*json.load(open(sys.argv[1]))["selected_updates"][sys.argv[2]], sep="\n")' \
      "$SELECTION" "$TASK_ARM")
    if [[ ${#BUDGETS[@]} -ne 1 ]]; then
      echo "$TASK_ARM selection must contain one update budget" >&2
      exit 1
    fi
    ARM_UPDATES[$TASK_ARM]=${BUDGETS[0]}
  else
    ARM_UPDATES[$TASK_ARM]=$UPDATES_ARGUMENT
  fi
  if [[ ! ${ARM_UPDATES[$TASK_ARM]} =~ ^(5000|20000|100000)$ ]]; then
    echo "$TASK_ARM update budget is outside the preregistered set" >&2
    exit 1
  fi
done
for TASK_ARM in adamw spectral; do
  CONFIG_ID=${WINNERS[$TASK_ARM]}
  UPDATES=${ARM_UPDATES[$TASK_ARM]}
  for SEED in "${SEEDS[@]}"; do
    TASK_NAME="final-refit-u${UPDATES}-s${SEED}-${TASK_ARM}-c${CONFIG_ID}"
    JOB_ID=$(sbatch --parsable --job-name="n8-ref-${TASK_ARM:0:1}-${SEED}" \
      --dependency="afterok:${DEPENDENCY}" \
      --export="ALL,TASK_NAME=${TASK_NAME},TASK_ARM=${TASK_ARM},CONFIG_ID=${CONFIG_ID},UPDATES=${UPDATES},SEED=${SEED}" \
      "$PROJECT/slurm/run-refit.sbatch")
    printf '%s\t%s\t%s\t%s\t%s\n' "$JOB_ID" "$TASK_ARM" "$CONFIG_ID" "$UPDATES" "$SEED"
  done
done
