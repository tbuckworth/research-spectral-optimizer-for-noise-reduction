#!/bin/bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "usage: $0 DEPENDENCY_JOB_ID SCORE_CSV SCORE_CSV [...]" >&2
  exit 2
fi
DEPENDENCY=$1
shift
if [[ ! $DEPENDENCY =~ ^[0-9]+$ ]]; then
  echo "dependency must be a numeric Slurm job ID" >&2
  exit 1
fi
PROJECT=${NUMERAI_PROJECT:-/mnt/nw/home/t.buckworth/numerai-competitive}
BASE="$PROJECT/configs/search-v1.json"
SOURCE="$PROJECT/results/selection-high-rank-source-r2048.json"
EXTENSION="$PROJECT/configs/search-v1-high-rank-extension.json"
MANIFEST="$PROJECT/results/submission-high-rank-probes.tsv"
AUDIT="$PROJECT/results/audit-high-rank-probes.json"
AUGMENTED="$PROJECT/configs/search-v1-high-rank.json"
LOG="$PROJECT/results/prepare-high-rank-search.log"
for SCORE in "$@"; do
  [[ -f $SCORE ]] || { echo "missing development score file $SCORE" >&2; exit 1; }
done
if [[ -e $SOURCE || -e $EXTENSION || -e $MANIFEST || -e ${MANIFEST}.tmp \
      || -e $AUDIT || -e $AUGMENTED ]]; then
  echo "high-rank preparation artifact already exists" >&2
  exit 1
fi

cd "$PROJECT"
"$PROJECT/uv" run --no-sync python -m numerai_competitive.high_rank_source \
  --search "$BASE" --scores "$@" --rank 2048 --output "$SOURCE" >> "$LOG" 2>&1
"$PROJECT/uv" run --no-sync python -m numerai_competitive.high_rank \
  --search "$BASE" --selection "$SOURCE" \
  --rank 512 --rank 1024 --rank 1536 --rank 2048 --rank 4096 \
  --output "$EXTENSION" >> "$LOG" 2>&1

TEMPORARY="${MANIFEST}.tmp"
: > "$TEMPORARY"
while IFS=$'\t' read -r CONFIG_ID UPDATES; do
  TASK_NAME="stage-outer_1_inner_1-u${UPDATES}-s0-spectral-c${CONFIG_ID}"
  JOB_ID=$(sbatch --parsable --job-name="n8-hrp-${CONFIG_ID}" \
    --dependency="afterok:${DEPENDENCY}" \
    --export="ALL,NUMERAI_PROBE_MODE=1,NUMERAI_SEARCH_CONFIG=${EXTENSION},TASK_NAME=${TASK_NAME},TASK_ARM=spectral,CONFIG_ID=${CONFIG_ID},SPLIT_NAME=outer_1_inner_1,UPDATES=${UPDATES},SEED=0" \
    "$PROJECT/slurm/run-high-rank-probe.sbatch")
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$JOB_ID" outer_1_inner_1 "$UPDATES" 0 spectral "$CONFIG_ID" >> "$TEMPORARY"
done < <(python3 -c '
import json,sys
value=json.load(open(sys.argv[1]))
for row in value["configs"]:
    if row.get("arm") == "spectral" and row["memory_screen"]["analytically_feasible_48gb"]:
        print(row["config_id"], row["required_probe_updates"], sep="\t")
' "$EXTENSION")
[[ -s $TEMPORARY ]] || { echo "no analytically feasible probe was generated" >&2; exit 1; }
mv "$TEMPORARY" "$MANIFEST"

export NUMERAI_SEARCH_CONFIG="$EXTENSION"
export NUMERAI_PROBE_MODE=1
bash "$PROJECT/slurm/supervise-resumable-stage.sh" "$MANIFEST" --skip-summary
RESULT_ARGS=()
while IFS=$'\t' read -r _JOB SPLIT UPDATES SEED ARM CONFIG_ID; do
  RESULT_ARGS+=(--result "$PROJECT/results/stage-${SPLIT}-u${UPDATES}-s${SEED}-${ARM}-c${CONFIG_ID}")
done < "$MANIFEST"
"$PROJECT/uv" run --no-sync python -m numerai_competitive.high_rank_probe \
  --extension "$EXTENSION" "${RESULT_ARGS[@]}" --output "$AUDIT" >> "$LOG" 2>&1
"$PROJECT/uv" run --no-sync python -m numerai_competitive.high_rank_search \
  --search "$BASE" --extension "$EXTENSION" --probe-audit "$AUDIT" \
  --output "$AUGMENTED" >> "$LOG" 2>&1
printf '%s high-rank probes audited and augmented search built\n' "$(date -Is)" >> "$LOG"
