#!/bin/bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: $0 DEPENDENCY_JOB_ID [OUTER_SPLIT]" >&2
  exit 2
fi
DEPENDENCY_JOB=$1
OUTER_SPLIT=${2:-outer_1}
if [[ ! $OUTER_SPLIT =~ ^outer_[123]$ ]]; then
  echo "outer split must be outer_1, outer_2 or outer_3" >&2
  exit 1
fi
OUTER_NUMBER=${OUTER_SPLIT#outer_}
INNER_COUNT=$((OUTER_NUMBER + 1))
if [[ $OUTER_NUMBER == 1 ]]; then
  SUPERVISOR_SESSION=numerai-f2-supervisor
  PROMOTE_SESSION=numerai-f2-promote
else
  SUPERVISOR_SESSION="numerai-outer${OUTER_NUMBER}-f2-supervisor"
  PROMOTE_SESSION="numerai-outer${OUTER_NUMBER}-f2-promote"
fi
PROJECT=${NUMERAI_PROJECT:-/mnt/nw/home/t.buckworth/numerai-competitive}
SELECTION="$PROJECT/results/selection-${OUTER_SPLIT}-f1-top4.json"
BASE_SELECTION="$PROJECT/results/selection-${OUTER_SPLIT}-f1-base-top4.json"
MANIFEST="$PROJECT/results/submission-${OUTER_SPLIT}-f2-u100000.tsv"
LOG="$PROJECT/results/promote-${OUTER_SPLIT}-f1.log"
SUMMARIES=()
MARKERS=()
SESSIONS=()
SPLITS=()
for INDEX in $(seq 1 "$INNER_COUNT"); do
  SPLITS+=("${OUTER_SPLIT}_inner_${INDEX}")
  SUMMARIES+=("$PROJECT/results/summary-${OUTER_SPLIT}_inner_${INDEX}-u20000-s0/scores.csv")
  MARKERS+=("$PROJECT/results/summary-${OUTER_SPLIT}_inner_${INDEX}-u20000-s0/summary-complete.json")
  if [[ $OUTER_NUMBER == 1 ]]; then
    SESSIONS+=("numerai-f1-inner${INDEX}")
  else
    SESSIONS+=("numerai-outer${OUTER_NUMBER}-f1-inner${INDEX}")
  fi
done

while true; do
  MISSING=0
  for INDEX in "${!MARKERS[@]}"; do
    if [[ ! -f ${MARKERS[$INDEX]} ]]; then
      MISSING=$((MISSING + 1))
      if ! tmux has-session -t "${SESSIONS[$INDEX]}" 2>/dev/null; then
        echo "an F1 monitor exited without an audited completion marker" >&2
        exit 1
      fi
    fi
  done
  [[ $MISSING -eq 0 ]] && break
  printf '%s waiting for all audited F1 summaries\n' "$(date -Is)" >> "$LOG"
  sleep 60
done

cd "$PROJECT"
if [[ -e "$SELECTION" || -e "$BASE_SELECTION" || -e "$MANIFEST" || -e "${MANIFEST}.tmp" ]] \
    || tmux has-session -t "$SUPERVISOR_SESSION" 2>/dev/null \
    || tmux has-session -t "$PROMOTE_SESSION" 2>/dev/null; then
  echo "F2 selection, manifest, temporary file or supervisor session already exists" >&2
  exit 1
fi
"$PROJECT/uv" run --no-sync python -m numerai_competitive.select_configs \
  --scores "${SUMMARIES[@]}" --top 4 --output "$BASE_SELECTION"
HIGH_RANK_SEARCH="$PROJECT/configs/search-v1-high-rank.json"
if [[ $OUTER_NUMBER == 1 ]]; then
  bash "$PROJECT/slurm/prepare-high-rank-search.sh" "$DEPENDENCY_JOB" "${SUMMARIES[@]}"
elif [[ ! -f $HIGH_RANK_SEARCH ]]; then
  echo "audited high-rank search from outer_1 is missing" >&2
  exit 1
fi
"$PROJECT/uv" run --no-sync python -m numerai_competitive.high_rank_selection \
  --selection "$BASE_SELECTION" --search "$HIGH_RANK_SEARCH" --output "$SELECTION"
export NUMERAI_SEARCH_CONFIG="$HIGH_RANK_SEARCH"
EXPECTED=$(python3 -c \
  'import json,sys; print(len(json.load(open(sys.argv[1]))["selected"]["paired_union"]))' \
  "$SELECTION")
if [[ $EXPECTED -lt 5 || $EXPECTED -gt 13 ]]; then
  echo "invalid paired-union size $EXPECTED" >&2
  exit 1
fi

TEMPORARY="${MANIFEST}.tmp"
: > "$TEMPORARY"
for SEED in 0 1 2; do
  bash "$PROJECT/slurm/submit-selected.sh" "$SELECTION" 100000 "$SEED" \
    "$DEPENDENCY_JOB" "${SPLITS[@]}" >> "$TEMPORARY"
done
mv "$TEMPORARY" "$MANIFEST"
FIRST_JOB=$(head -n 1 "$MANIFEST" | cut -f1 | cut -d';' -f1)
LAST_JOB=$(tail -n 1 "$MANIFEST" | cut -f1 | cut -d';' -f1)
EXPECTED_ROWS=$((EXPECTED * 2 * INNER_COUNT * 3))
if [[ $(wc -l < "$MANIFEST") -ne $EXPECTED_ROWS ]]; then
  echo "submission manifest row count differs from paired union x arms x folds x seeds" >&2
  exit 1
fi

tmux new-session -d -s "$SUPERVISOR_SESSION" \
  "NUMERAI_SEARCH_CONFIG='$HIGH_RANK_SEARCH' bash '$PROJECT/slurm/supervise-resumable-stage.sh' '$MANIFEST'"
tmux new-session -d -s "$PROMOTE_SESSION" \
  "NUMERAI_SEARCH_CONFIG='$HIGH_RANK_SEARCH' bash '$PROJECT/slurm/promote-f2-when-ready.sh' '$LAST_JOB' '$OUTER_SPLIT'"
printf '%s submitted F2 union=%s jobs=%s--%s dependency=%s\n' \
  "$(date -Is)" "$EXPECTED" "$FIRST_JOB" "$LAST_JOB" "$DEPENDENCY_JOB" >> "$LOG"
