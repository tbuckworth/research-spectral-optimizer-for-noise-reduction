#!/bin/bash
#SBATCH --partition=compute
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=23:30:00
#SBATCH --signal=B:USR1@3600
#SBATCH --output=/mnt/nw/home/t.buckworth/numerai-competitive/slurm-%x-%j.out

set -euo pipefail
PROJECT=/mnt/nw/home/t.buckworth/numerai-competitive
SHARDS=/ephemeral/t.buckworth/numerai-v5.3
SEARCH_CONFIG=${NUMERAI_SEARCH_CONFIG:?high-rank probe requires an explicit search manifest}
export UV_CACHE_DIR=/ephemeral/t.buckworth/uv-cache
export CUBLAS_WORKSPACE_CONFIG=:4096:8
cd "$PROJECT"
"$PROJECT/uv" run --no-sync python -c \
  'import torch; assert torch.cuda.is_available(), "CUDA unavailable; refusing CPU fallback"'
exec "$PROJECT/.venv/bin/python" -m numerai_competitive.run_high_rank_probe \
  --search "$SEARCH_CONFIG" --shards "$SHARDS" \
  --output "$PROJECT/results/${TASK_NAME}" --config-id "$CONFIG_ID" \
  --split "$SPLIT_NAME" --updates "$UPDATES" --seed "$SEED"
