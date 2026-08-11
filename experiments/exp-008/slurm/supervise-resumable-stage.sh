#!/bin/bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 || (${2:-} != "" && ${2:-} != "--skip-summary") ]]; then
  echo "usage: $0 SUBMISSION_TSV [--skip-summary]" >&2
  exit 2
fi
ORIGINAL_MANIFEST=$1
SKIP_SUMMARY=${2:-}
PROJECT=${NUMERAI_PROJECT:-/mnt/nw/home/t.buckworth/numerai-competitive}
QUEUE_USER=${NUMERAI_QUEUE_USER:-t.buckworth}
FEATURES=/mnt/nw/home/t.buckworth/numerai-v5.3-source/features.json
SEARCH="$PROJECT/configs/search-v1.json"
LOG="${ORIGINAL_MANIFEST%.tsv}-supervisor.log"
CURRENT_MANIFEST=$ORIGINAL_MANIFEST
RETRY=0

manifest_jobs_active() {
  declare -A wanted=()
  local queued_jobs
  while IFS=$'\t' read -r job _rest; do
    wanted["${job%%;*}"]=1
  done < "$CURRENT_MANIFEST"
  if ! queued_jobs=$(squeue -u "$QUEUE_USER" -h -o '%i'); then
    echo "squeue failed; refusing to infer that submitted tasks ended" >&2
    exit 1
  fi
  while read -r queued; do
    [[ -n $queued ]] || continue
    if [[ -n ${wanted[$queued]+present} ]]; then
      return 0
    fi
  done <<< "$queued_jobs"
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

if [[ $SKIP_SUMMARY != "--skip-summary" ]]; then
  EXPECTED=$(awk -F $'\t' '{print $6}' "$ORIGINAL_MANIFEST" | sort -nu | wc -l)
  while IFS=$'\t' read -r split updates seed; do
    output="$PROJECT/results/summary-${split}-u${updates}-s${seed}"
    "$PROJECT/uv" run --no-sync python -m numerai_competitive.summarize \
      --results "$PROJECT/results" --output "$output" --split "$split" \
      --updates "$updates" --seed "$seed" --expected-configs "$EXPECTED" \
      --search "$SEARCH" --features "$FEATURES" >> "$LOG" 2>&1
  done < <(awk -F $'\t' '{print $2 "\t" $3 "\t" $4}' "$ORIGINAL_MANIFEST" | sort -u)
  completion="stage complete and all cells summarized"
else
  completion="stage complete; downstream audit required"
fi
printf '%s %s (retries=%s)\n' "$(date -Is)" "$completion" "$RETRY" >> "$LOG"
