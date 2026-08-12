#!/bin/bash
#SBATCH --partition=compute
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=23:30:00
#SBATCH --signal=B:USR1@3600
#SBATCH --output=/mnt/nw/home/t.buckworth/numerai-competitive/slurm-%x-%j.out

set -euo pipefail
PROJECT=${NUMERAI_PROJECT:-/mnt/nw/home/t.buckworth/numerai-competitive}
SHARD=/ephemeral/t.buckworth/numerai-v5.3-production-${PRODUCTION_CODE_COMMIT:0:12}-${CANDIDATE_FEATURE_SET}
export UV_CACHE_DIR=/ephemeral/t.buckworth/uv-cache
export CUBLAS_WORKSPACE_CONFIG=:4096:8
cd "$PROJECT"
"$PROJECT/uv" run --no-sync python -c \
  'import torch; assert torch.cuda.is_available(), "CUDA unavailable; refusing CPU fallback"'
exec "$PROJECT/.venv/bin/python" -m numerai_competitive.production_refit \
  --search configs/search-v1.json --shard "$SHARD" --freeze results/freeze.json \
  --evaluation-marker results/official-validation/evaluation-complete.json \
  --output "$PROJECT/results/${TASK_NAME}" --seed "$SEED" --device cuda
