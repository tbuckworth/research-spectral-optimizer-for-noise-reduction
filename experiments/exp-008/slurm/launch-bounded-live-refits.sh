#!/bin/bash
set -euo pipefail

if [[ $# -ne 1 || ! $1 =~ ^[0-9]+$ ]]; then
  echo "usage: $0 ENVIRONMENT_JOB_ID" >&2
  exit 2
fi
PROJECT=/mnt/nw/home/t.buckworth/numerai-live-candidates
RESULTS=$PROJECT/results
MANIFEST=$RESULTS/submission-bounded-live-refits.tsv
[[ -f $RESULTS/bounded-live-freeze.json ]] || { echo "bounded freeze missing" >&2; exit 1; }
[[ ! -e $MANIFEST ]] || { echo "bounded live manifest already exists" >&2; exit 1; }
BUILD=$(sbatch --parsable --dependency="afterok:$1" "$PROJECT/slurm/run-build-bounded-live-shard.sbatch")
printf 'job_id\tarm\tconfig_id\tseed\tdependency\n' > "${MANIFEST}.tmp"
JOBS=()
for SPEC in adamw:38 spectral:39; do
  IFS=: read -r ARM CONFIG_ID <<< "$SPEC"
  for SEED in 0 1 2; do
    JOB=$(sbatch --parsable --job-name="n8-live-${ARM:0:1}-${SEED}" \
      --dependency="afterok:${BUILD%%;*}" \
      --export="ALL,ARM=${ARM},CONFIG_ID=${CONFIG_ID},SEED=${SEED}" \
      "$PROJECT/slurm/run-bounded-live-refit.sbatch")
    JOBS+=("${JOB%%;*}")
    printf '%s\t%s\t%s\t%s\t%s\n' "$JOB" "$ARM" "$CONFIG_ID" "$SEED" "${BUILD%%;*}" \
      >> "${MANIFEST}.tmp"
  done
done
mv "${MANIFEST}.tmp" "$MANIFEST"
cat "$MANIFEST"
