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
SHARDS=/ephemeral/t.buckworth/numerai-v5.3-corrected-live
export UV_CACHE_DIR=/ephemeral/t.buckworth/uv-cache
export TMPDIR=/ephemeral/t.buckworth/tmp
export PYTHONPATH="$PROJECT/src"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
mkdir -p "$TMPDIR"
cd "$PROJECT"
"$PROJECT/.venv/bin/python" -c \
  'import torch; assert torch.cuda.is_available(), "CUDA unavailable"'
exec "$PROJECT/.venv/bin/python" -m numerai_competitive.run_task \
  --search configs/search-v1.json --shards "$SHARDS" \
  --output "$PROJECT/results/calibration-${SPLIT}-${ARM}" \
  --arm "$ARM" --config-id "$CONFIG_ID" --split "$SPLIT" \
  --updates "$MAX_UPDATES" --seed 0 \
  --validation-examples 2560000 5120000 10240000 20480000 40960000
