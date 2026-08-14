#!/bin/bash
#SBATCH --partition=compute
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=23:30:00
#SBATCH --signal=B:USR1@3600
#SBATCH --output=/mnt/nw/home/t.buckworth/numerai-live-candidates/slurm-%x-%j.out

set -euo pipefail
PROJECT=/mnt/nw/home/t.buckworth/numerai-live-candidates
SHARD=/ephemeral/t.buckworth/numerai-v5.3-bounded-live-all
export PYTHONPATH="$PROJECT/src"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
cd "$PROJECT"
"$PROJECT/.venv/bin/python" -c 'import torch; assert torch.cuda.is_available()'
exec "$PROJECT/.venv/bin/python" -m numerai_competitive.bounded_live refit \
  --search configs/search-v1.json --shard "$SHARD" \
  --freeze results/bounded-live-freeze.json \
  --output "results/bounded-live-refit-s${SEED}-${ARM}-c${CONFIG_ID}" \
  --arm "$ARM" --seed "$SEED" --device cuda
