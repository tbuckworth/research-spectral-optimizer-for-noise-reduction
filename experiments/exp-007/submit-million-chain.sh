#!/usr/bin/env bash
set -euo pipefail

# Slurm records a QOS time-limit termination as unsuccessful, so afterok would
# cancel the continuation chain.  Each allocation is independently resumable;
# afterany is therefore the intended dependency.
allocations="${1:-4}"
if ! [[ "$allocations" =~ ^[1-9][0-9]*$ ]]; then
  echo "usage: $0 [positive-allocation-count]" >&2
  exit 2
fi

cd "$(dirname "$0")"
job_id="$(sbatch --parsable --array=0-3 run-million.sbatch)"
printf '%s\n' "$job_id"

for ((i = 1; i < allocations; i++)); do
  job_id="$(sbatch --parsable --dependency="afterany:${job_id}" \
    --array=0-3 run-million.sbatch)"
  printf '%s\n' "$job_id"
done
