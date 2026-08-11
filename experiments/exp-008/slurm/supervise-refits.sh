#!/bin/bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 SUBMISSION_TSV SELECTION_JSON" >&2
  exit 2
fi
ORIGINAL_MANIFEST=$1
SELECTION=$2
PROJECT=${NUMERAI_PROJECT:-/mnt/nw/home/t.buckworth/numerai-competitive}
SEARCH=${NUMERAI_SEARCH_CONFIG:-$PROJECT/configs/search-v1.json}
QUEUE_USER=${NUMERAI_QUEUE_USER:-t.buckworth}
LOG="${ORIGINAL_MANIFEST%.tsv}-supervisor.log"
CURRENT_MANIFEST=$ORIGINAL_MANIFEST
RETRY=0

manifest_jobs_active() {
  declare -A wanted=()
  local queued_jobs
  while IFS=$'\t' read -r job _rest; do wanted["${job%%;*}"]=1; done < "$CURRENT_MANIFEST"
  if ! queued_jobs=$(squeue -u "$QUEUE_USER" -h -o '%i'); then
    echo "squeue failed; refusing to infer that submitted refits ended" >&2
    exit 1
  fi
  while read -r queued; do
    [[ -n $queued ]] || continue
    [[ -n ${wanted[$queued]+present} ]] && return 0
  done <<< "$queued_jobs"
  return 1
}

all_refits_complete() {
  while IFS=$'\t' read -r _job arm config_id updates seed extra; do
    if [[ -n ${extra:-} || -z ${seed:-} ]]; then return 1; fi
    directory="$PROJECT/results/final-refit-u${updates}-s${seed}-${arm}-c${config_id}"
    [[ -f "$directory/result.json" && -f "$directory/model.pt" ]] || return 1
  done < "$ORIGINAL_MANIFEST"
}

while true; do
  while manifest_jobs_active; do
    printf '%s refits still active (retry=%s)\n' "$(date -Is)" "$RETRY" >> "$LOG"
    sleep 60
  done
  all_refits_complete && break
  if [[ $RETRY -ge 10 ]]; then
    echo "more than ten refit checkpoint cycles required" >&2
    exit 1
  fi
  dependency=$(awk -F $'\t' '{value=$1} END {sub(/;.*/, "", value); print value}' \
    "$CURRENT_MANIFEST")
  [[ $dependency =~ ^[0-9]+$ ]] || { echo "refit manifest lacks dependency job" >&2; exit 1; }
  RETRY=$((RETRY + 1))
  retry_manifest="${ORIGINAL_MANIFEST%.tsv}.retry-${RETRY}.tsv"
  bash "$PROJECT/slurm/resume-checkpointed-refits.sh" \
    "$ORIGINAL_MANIFEST" "$dependency" > "${retry_manifest}.tmp"
  mv "${retry_manifest}.tmp" "$retry_manifest"
  CURRENT_MANIFEST=$retry_manifest
  printf '%s resubmitted %s checkpointed refits (retry=%s)\n' \
    "$(date -Is)" "$(wc -l < "$CURRENT_MANIFEST")" "$RETRY" >> "$LOG"
done

"$PROJECT/uv" run --no-sync python -m numerai_competitive.audit_refits \
  --manifest "$ORIGINAL_MANIFEST" --results "$PROJECT/results" --selection "$SELECTION" \
  --search "$SEARCH" \
  --features /mnt/nw/home/t.buckworth/numerai-v5.3-source/features.json \
  --output "$PROJECT/results/audit-final-refits-budgeted" >> "$LOG" 2>&1
printf '%s refits complete and audited (retries=%s)\n' "$(date -Is)" "$RETRY" >> "$LOG"
