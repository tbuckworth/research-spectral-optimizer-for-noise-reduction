#!/bin/bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 3 ]]; then
  echo "usage: $0 SUBMISSION_TSV [--skip-summary | --selection SELECTION_JSON]" >&2
  exit 2
fi
ORIGINAL_MANIFEST=$1
SUMMARY_MODE=${2:-}
SELECTION=${3:-}
if [[ $SUMMARY_MODE == --selection ]]; then
  if [[ -z $SELECTION || ! -f $SELECTION ]]; then
    echo "--selection requires an existing selection JSON" >&2
    exit 2
  fi
elif [[ ( -n $SUMMARY_MODE && $SUMMARY_MODE != --skip-summary ) || -n $SELECTION ]]; then
  echo "usage: $0 SUBMISSION_TSV [--skip-summary | --selection SELECTION_JSON]" >&2
  exit 2
fi
PROJECT=${NUMERAI_PROJECT:-/mnt/nw/home/t.buckworth/numerai-competitive}
QUEUE_USER=${NUMERAI_QUEUE_USER:-t.buckworth}
FEATURES=/mnt/nw/home/t.buckworth/numerai-v5.3-source/features.json
SEARCH=${NUMERAI_SEARCH_CONFIG:-$PROJECT/configs/search-v1.json}
SUMMARY_PREFIX=${NUMERAI_SUMMARY_PREFIX:-}
if [[ ! $SUMMARY_PREFIX =~ ^[A-Za-z0-9_-]*$ ]]; then
  echo "NUMERAI_SUMMARY_PREFIX contains unsafe characters" >&2
  exit 2
fi
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
  dependency=$(awk -F $'\t' '$1 != 0 {value=$1} END {sub(/;.*/, "", value); print value}' \
    "$CURRENT_MANIFEST")
  if [[ ! $dependency =~ ^[0-9]+$ ]]; then
    echo "incomplete tasks exist but manifest has no submitted dependency job" >&2
    exit 1
  fi
  RETRY=$((RETRY + 1))
  retry_manifest="${ORIGINAL_MANIFEST%.tsv}.retry-${RETRY}.tsv"
  bash "$PROJECT/slurm/resume-checkpointed-stage.sh" \
    "$ORIGINAL_MANIFEST" "$dependency" > "${retry_manifest}.tmp"
  mv "${retry_manifest}.tmp" "$retry_manifest"
  CURRENT_MANIFEST=$retry_manifest
  printf '%s resubmitted %s checkpointed tasks (retry=%s)\n' \
    "$(date -Is)" "$(wc -l < "$CURRENT_MANIFEST")" "$RETRY" >> "$LOG"
done

if [[ $SUMMARY_MODE != "--skip-summary" ]]; then
  while IFS=$'\t' read -r split updates seed; do
    mapfile -t EXPECTED_PAIRS < <(
      awk -F $'\t' -v target_split="$split" -v target_updates="$updates" \
        -v target_seed="$seed" \
        '$2 == target_split && $3 == target_updates && $4 == target_seed {print $5 ":" $6}' \
        "$ORIGINAL_MANIFEST" | sort -u
    )
    ADAMW_EXPECTED=$(printf '%s\n' "${EXPECTED_PAIRS[@]}" | grep -c '^adamw:' || true)
    SPECTRAL_EXPECTED=$(printf '%s\n' "${EXPECTED_PAIRS[@]}" | grep -c '^spectral:' || true)
    if [[ $ADAMW_EXPECTED -eq 0 || $SPECTRAL_EXPECTED -eq 0 ]]; then
      echo "each summary cell must contain expected configurations for both arms" >&2
      exit 1
    fi
    EXPECTED_ARGS=()
    for PAIR in "${EXPECTED_PAIRS[@]}"; do
      EXPECTED_ARGS+=(--expected-arm-config-id "$PAIR")
    done
    output="$PROJECT/results/summary-${SUMMARY_PREFIX}${split}-u${updates}-s${seed}"
    "$PROJECT/uv" run --no-sync python -m numerai_competitive.summarize \
      --results "$PROJECT/results" --output "$output" --split "$split" \
      --updates "$updates" --seed "$seed" \
      "${EXPECTED_ARGS[@]}" \
      --search "$SEARCH" --features "$FEATURES" >> "$LOG" 2>&1
  done < <(awk -F $'\t' '{print $2 "\t" $3 "\t" $4}' "$ORIGINAL_MANIFEST" | sort -u)
  if [[ $SUMMARY_MODE == --selection ]]; then
    completion="stage complete and all exact selected cells summarized"
  else
    completion="stage complete and all cells summarized"
  fi
else
  completion="stage complete; downstream audit required"
fi
printf '%s %s (retries=%s)\n' "$(date -Is)" "$completion" "$RETRY" >> "$LOG"
