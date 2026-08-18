#!/bin/bash
#SBATCH --partition=compute
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:10:00
#SBATCH --job-name=n8-correct-launch
#SBATCH --output=/mnt/nw/home/t.buckworth/numerai-corrected-live/slurm-%x-%j.out

set -euo pipefail
PROJECT=/mnt/nw/home/t.buckworth/numerai-corrected-live/experiments/exp-008
cd "$PROJECT"
JOBS=()
for ARM_CONFIG in "adamw 38" "spectral 39"; do
  read -r ARM CONFIG_ID <<< "$ARM_CONFIG"
  for SEED in 0 1 2; do
    JOB=$(sbatch --parsable --job-name="n8-cor-${ARM:0:1}-${SEED}" \
      --export="ALL,ARM=$ARM,CONFIG_ID=$CONFIG_ID,SEED=$SEED" \
      slurm/run-corrected-refit.sbatch)
    JOBS+=("$JOB")
    printf '%s\t%s\t%s\n' "$JOB" "$ARM" "$SEED"
  done
done
DEPENDENCY=$(IFS=:; echo "${JOBS[*]}")
AUDIT=$(sbatch --parsable --dependency="afterok:${DEPENDENCY}" \
  slurm/audit-export-corrected-live.sbatch)
printf '%s\taudit-and-export\n' "$AUDIT"
