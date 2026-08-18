#!/bin/bash
#SBATCH --partition=compute
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --job-name=n8-correct-audit
#SBATCH --output=/mnt/nw/home/t.buckworth/numerai-corrected-live/slurm-%x-%j.out

set -euo pipefail
PROJECT=/mnt/nw/home/t.buckworth/numerai-corrected-live/experiments/exp-008
SHARD=/ephemeral/t.buckworth/numerai-v5.3-corrected-production-all
export TMPDIR=/ephemeral/t.buckworth/tmp
mkdir -p "$TMPDIR"
cd "$PROJECT"
"$PROJECT/.venv/bin/python" -m numerai_competitive.corrected_live audit \
  --freeze results/corrected-live-freeze.json --shard "$SHARD" \
  --results results --output results/corrected-live-refit-audit.json
"$PROJECT/.venv/bin/python" -m numerai_competitive.corrected_live export \
  --freeze results/corrected-live-freeze.json \
  --audit results/corrected-live-refit-audit.json \
  --output results/corrected-live-bundles
