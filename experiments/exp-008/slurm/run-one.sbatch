#!/bin/bash
#SBATCH --partition=compute
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/mnt/nw/home/t.buckworth/numerai-competitive/slurm-%x-%j.out

set -euo pipefail
PROJECT=/mnt/nw/home/t.buckworth/numerai-competitive
SHARDS=/ephemeral/t.buckworth/numerai-v5.3
export UV_CACHE_DIR=/ephemeral/t.buckworth/uv-cache
cd "$PROJECT"
"$PROJECT/uv" sync --frozen
"$PROJECT/uv" run python -c \
  'import torch; assert torch.cuda.is_available(), "CUDA unavailable; refusing CPU fallback"'
"$PROJECT/uv" run python -m numerai_competitive.run_task \
  --search configs/search-v1.json --shards "$SHARDS" \
  --output "$PROJECT/results/${TASK_NAME}" --arm "$TASK_ARM" \
  --config-id "$CONFIG_ID" --split "$SPLIT_NAME" \
  --updates "$UPDATES" --seed "$SEED"
