#!/bin/bash
set -euo pipefail

if [[ $# -lt 4 ]]; then
  echo "usage: $0 SELECTION_JSON SEED DEPENDENCY_JOB_ID SPLIT..." >&2
  exit 2
fi
SELECTION=$1
SEED=$2
DEPENDENCY=$3
shift 3
PROJECT=${NUMERAI_PROJECT:-/mnt/nw/home/t.buckworth/numerai-competitive}
REUSE_COMPLETE=${NUMERAI_REUSE_COMPLETE:-0}
if [[ ! $SEED =~ ^[0-9]+$ || ! $DEPENDENCY =~ ^[0-9]+$ \
    || ( $REUSE_COMPLETE != 0 && $REUSE_COMPLETE != 1 ) ]]; then
  echo "invalid seed, dependency, or reuse flag" >&2
  exit 1
fi
mapfile -t CANDIDATES < <(python3 -c '
import json,sys
value=json.load(open(sys.argv[1]))
for entry in value.get("budgeted_candidates", []):
    config_id, updates = entry.get("config_id"), entry.get("updates")
    if not isinstance(config_id, int) or config_id < 0 or updates not in {5000, 20000, 100000}:
        raise SystemExit("invalid budgeted candidate")
    print(config_id, updates)
' "$SELECTION")
if [[ ${#CANDIDATES[@]} -lt 1 || ${#CANDIDATES[@]} -gt 6 ]]; then
  echo "budgeted candidate union must contain one to six pairs" >&2
  exit 1
fi

for SPLIT_NAME in "$@"; do
  for CANDIDATE in "${CANDIDATES[@]}"; do
    read -r CONFIG_ID UPDATES <<< "$CANDIDATE"
    for TASK_ARM in adamw spectral; do
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
