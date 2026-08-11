#!/bin/bash
#SBATCH --partition=compute
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=23:30:00
#SBATCH --output=/mnt/nw/home/t.buckworth/numerai-competitive/slurm-%x-%j.out

set -euo pipefail
PROJECT=/mnt/nw/home/t.buckworth/numerai-competitive
SHARD="/ephemeral/t.buckworth/numerai-v5.3-validation-all-${CODE_COMMIT:0:12}"
FREEZE="$PROJECT/results/freeze.json"
OUTPUT="$PROJECT/results/official-validation"
export UV_CACHE_DIR=/ephemeral/t.buckworth/uv-cache
cd "$PROJECT"
"$PROJECT/uv" run --no-sync python -c \
  'import torch; assert torch.cuda.is_available(), "CUDA unavailable; refusing CPU fallback"'
ADAMW_MODELS=()
SPECTRAL_MODELS=()
for SEED in 0 1 2; do
  ADAMW_MODELS+=(--adamw-model \
    "$PROJECT/results/final-refit-u100000-s${SEED}-adamw-c${ADAMW_CONFIG}/model.pt")
  SPECTRAL_MODELS+=(--spectral-model \
    "$PROJECT/results/final-refit-u100000-s${SEED}-spectral-c${SPECTRAL_CONFIG}/model.pt")
done
"$PROJECT/uv" run --no-sync python -m numerai_competitive.evaluate_frozen \
  --shard "$SHARD" --freeze "$FREEZE" "${ADAMW_MODELS[@]}" "${SPECTRAL_MODELS[@]}" \
  --output "$OUTPUT" --device cuda --batch-size 4096
