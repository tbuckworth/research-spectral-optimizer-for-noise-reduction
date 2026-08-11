#!/bin/bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 SUBMISSION_TSV" >&2
  exit 2
fi
ORIGINAL_MANIFEST=$1
PROJECT=/mnt/nw/home/t.buckworth/numerai-competitive
FEATURES=/mnt/nw/home/t.buckworth/numerai-v5.3-source/features.json
SEARCH="$PROJECT/configs/search-v1.json"
LOG="${ORIGINAL_MANIFEST%.tsv}-supervisor.log"
CURRENT_MANIFEST=$ORIGINAL_MANIFEST
RETRY=0

manifest_jobs_active() {
  declare -A wanted=()
  while IFS=$'\t' read -r job _rest; do
    wanted["${job%%;*}"]=1
  done < "$CURRENT_MANIFEST"
  while read -r queued; do
    if [[ -n ${wanted[$queued]+present} ]]; then
      return 0
    fi
  done < <(squeue -u t.buckworth -h -o '%i')
  return 1
}

all_results_complete() {
  while IFS=$'\t' read -r _job split updates seed arm config_id extra; do
    if [[ -n ${extra:-} || -z ${config_id:-} ]]; then
      echo "invalid six-column stage manifest row" >&2
      return 1
    fi
    result="$PROJECT/results/stage-${split}-u${updates}-s${seed}-${arm}-c${config_id}/result.json"
    [[ -f "$result" ]] || return 1
  done < "$ORIGINAL_MANIFEST"
}

while true; do
  while manifest_jobs_active; do
    printf '%s submitted tasks still active (retry=%s)\n' "$(date -Is)" "$RETRY" >> "$LOG"
    sleep 60
  done
  if all_results_complete; then
    break
  fi
  if [[ $RETRY -ge 10 ]]; then
    echo "more than ten checkpoint cycles required" >&2
    exit 1
  fi
  dependency=$(tail -n 1 "$CURRENT_MANIFEST" | cut -f1 | cut -d';' -f1)
  RETRY=$((RETRY + 1))
  retry_manifest="${ORIGINAL_MANIFEST%.tsv}.retry-${RETRY}.tsv"
  bash "$PROJECT/slurm/resume-checkpointed-stage.sh" \
    "$ORIGINAL_MANIFEST" "$dependency" > "${retry_manifest}.tmp"
  mv "${retry_manifest}.tmp" "$retry_manifest"
  CURRENT_MANIFEST=$retry_manifest
  printf '%s resubmitted %s checkpointed tasks (retry=%s)\n' \
    "$(date -Is)" "$(wc -l < "$CURRENT_MANIFEST")" "$RETRY" >> "$LOG"
done

EXPECTED=$(awk -F $'\t' '{print $6}' "$ORIGINAL_MANIFEST" | sort -nu | wc -l)
while IFS=$'\t' read -r split updates seed; do
  output="$PROJECT/results/summary-${split}-u${updates}-s${seed}"
  "$PROJECT/uv" run --no-sync python -m numerai_competitive.summarize \
    --results "$PROJECT/results" --output "$output" --split "$split" \
    --updates "$updates" --seed "$seed" --expected-configs "$EXPECTED" \
    --search "$SEARCH" --features "$FEATURES" >> "$LOG" 2>&1
done < <(awk -F $'\t' '{print $2 "\t" $3 "\t" $4}' "$ORIGINAL_MANIFEST" | sort -u)
printf '%s stage complete and all cells summarized (retries=%s)\n' \
  "$(date -Is)" "$RETRY" >> "$LOG"
