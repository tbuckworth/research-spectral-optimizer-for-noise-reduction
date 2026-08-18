#!/bin/bash
#SBATCH --partition=compute
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=23:30:00
#SBATCH --signal=B:USR1@3600
#SBATCH --output=/mnt/nw/home/t.buckworth/numerai-corrected-live/slurm-%x-%j.out

set -euo pipefail
PROJECT=/mnt/nw/home/t.buckworth/numerai-corrected-live/experiments/exp-008
SHARD=/ephemeral/t.buckworth/numerai-v5.3-corrected-production-all
export TMPDIR=/ephemeral/t.buckworth/tmp
export CUBLAS_WORKSPACE_CONFIG=:4096:8
mkdir -p "$TMPDIR"
cd "$PROJECT"
"$PROJECT/.venv/bin/python" -c 'import torch; assert torch.cuda.is_available()'
exec "$PROJECT/.venv/bin/python" -m numerai_competitive.corrected_live refit \
  --search configs/search-v1.json --shard "$SHARD" \
  --freeze results/corrected-live-freeze.json \
  --output "results/corrected-live-refit-s${SEED}-${ARM}-c${CONFIG_ID}" \
  --arm "$ARM" --seed "$SEED" --device cuda
