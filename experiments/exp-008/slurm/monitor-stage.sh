#!/bin/bash
set -euo pipefail

if [[ $# -lt 5 || $# -gt 6 ]]; then
  echo "usage: $0 FIRST_JOB LAST_JOB SPLIT UPDATES SEED [EXPECTED_CONFIGS]" >&2
  exit 2
fi
FIRST_JOB=$1
LAST_JOB=$2
SPLIT=$3
UPDATES=$4
SEED=$5
EXPECTED_CONFIGS=${6:-40}
PROJECT=/mnt/nw/home/t.buckworth/numerai-competitive
STATUS="$PROJECT/results/monitor-${SPLIT}-u${UPDATES}-s${SEED}.log"
ADMISSION="$PROJECT/results/base-search-memory-admission.json"

while squeue -u t.buckworth -h -o '%i' | awk -v first="$FIRST_JOB" -v last="$LAST_JOB" \
    '$1 >= first && $1 <= last {found=1} END {exit !found}'; do
  if squeue -u t.buckworth -h -o '%i|%r' | awk -F'|' -v first="$FIRST_JOB" -v last="$LAST_JOB" \
      '$1 >= first && $1 <= last && $2 ~ /DependencyNeverSatisfied/ {found=1} END {exit !found}'; then
    printf '%s dependency failure; stage cannot run\n' "$(date -Is)" >> "$STATUS"
    exit 1
  fi
  printf '%s jobs still active\n' "$(date -Is)" >> "$STATUS"
  sleep 60
done

cd "$PROJECT"
if [[ ! -f "$ADMISSION" ]]; then
  "$PROJECT/uv" run --no-sync python -m numerai_competitive.base_admission \
    --search "$PROJECT/configs/search-v1.json" --results "$PROJECT/results" \
    --output "$ADMISSION" --final >> "$STATUS" 2>&1
fi
mapfile -t ADMITTED_IDS < <(python3 -c '
import json,sys
value=json.load(open(sys.argv[1]))
if value.get("status") != "complete" or value.get("pending_probe_config_ids"):
    raise SystemExit("base-search memory admission is not final")
for config_id in value.get("admitted_config_ids", []):
    print(config_id)
' "$ADMISSION")
if [[ ${#ADMITTED_IDS[@]} -eq 0 ]]; then
  echo "base-search memory admission contains no eligible paired IDs" >&2
  exit 1
fi
EXPECTED_ARGS=()
for CONFIG_ID in "${ADMITTED_IDS[@]}"; do
  EXPECTED_ARGS+=(--expected-config-id "$CONFIG_ID")
done
if "$PROJECT/uv" run python -m numerai_competitive.summarize \
    --results results --output "results/summary-${SPLIT}-u${UPDATES}-s${SEED}" \
    --split "$SPLIT" --updates "$UPDATES" --seed "$SEED" \
    --search configs/search-v1.json \
    --features /mnt/nw/home/t.buckworth/numerai-v5.3-source/features.json \
    "${EXPECTED_ARGS[@]}" >> "$STATUS" 2>&1; then
  printf '%s stage complete and summarized admitted=%s requested=%s\n' \
    "$(date -Is)" "${#ADMITTED_IDS[@]}" "$EXPECTED_CONFIGS" >> "$STATUS"
else
  printf '%s stage ended incomplete; inspection required\n' "$(date -Is)" >> "$STATUS"
  exit 1
fi
