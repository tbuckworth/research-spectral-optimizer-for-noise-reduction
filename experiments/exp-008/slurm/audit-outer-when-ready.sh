#!/bin/bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "usage: $0 MANIFEST SELECTION_JSON OUTER_SPLIT OUTPUT" >&2
  exit 2
fi
MANIFEST=$1
SELECTION=$2
OUTER_SPLIT=$3
OUTPUT=$4
PROJECT=/mnt/nw/home/t.buckworth/numerai-competitive
LOG="${MANIFEST%.tsv}-audit.log"

while IFS=$'\t' read -r _job split updates seed arm config_id extra; do
  if [[ -n ${extra:-} || -z ${config_id:-} ]]; then
    echo "invalid six-column outer manifest row" >&2
    exit 1
  fi
done < "$MANIFEST"

while true; do
  missing=0
  while IFS=$'\t' read -r _job split updates seed arm config_id _extra; do
    result="$PROJECT/results/stage-${split}-u${updates}-s${seed}-${arm}-c${config_id}/result.json"
    [[ -f "$result" ]] || missing=$((missing + 1))
  done < "$MANIFEST"
  [[ $missing -eq 0 ]] && break
  if ! tmux has-session -t numerai-outer1-supervisor 2>/dev/null; then
    sleep 2
    missing_after_exit=0
    while IFS=$'\t' read -r _job split updates seed arm config_id _extra; do
      result="$PROJECT/results/stage-${split}-u${updates}-s${seed}-${arm}-c${config_id}/result.json"
      [[ -f "$result" ]] || missing_after_exit=$((missing_after_exit + 1))
    done < "$MANIFEST"
    if [[ $missing_after_exit -ne 0 ]]; then
      echo "outer supervisor exited before all selected results existed" >&2
      exit 1
    fi
    break
  fi
  printf '%s waiting for %s selected outer results\n' "$(date -Is)" "$missing" >> "$LOG"
  sleep 60
done

"$PROJECT/uv" run --no-sync python -m numerai_competitive.audit_outer \
  --manifest "$MANIFEST" --results "$PROJECT/results" --selection "$SELECTION" \
  --search "$PROJECT/configs/search-v1.json" \
  --features /mnt/nw/home/t.buckworth/numerai-v5.3-source/features.json \
  --outer-split "$OUTER_SPLIT" --updates 100000 --seed 0 --seed 1 --seed 2 \
  --output "$OUTPUT" >> "$LOG" 2>&1
printf '%s selected outer audit complete\n' "$(date -Is)" >> "$LOG"
