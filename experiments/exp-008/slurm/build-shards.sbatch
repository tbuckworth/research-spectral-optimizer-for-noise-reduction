#!/bin/bash
#SBATCH --partition=compute
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --job-name=numerai-shards
#SBATCH --output=/mnt/nw/home/t.buckworth/numerai-competitive/slurm-%x-%j.out

set -euo pipefail
PROJECT=/mnt/nw/home/t.buckworth/numerai-competitive
ROOT=/ephemeral/t.buckworth/numerai-v5.3
SOURCE=/mnt/nw/home/t.buckworth/numerai-v5.3-source
export UV_CACHE_DIR=/ephemeral/t.buckworth/uv-cache
cd "$PROJECT"
"$PROJECT/uv" sync --frozen
"$PROJECT/uv" run python -m numerai_competitive.build_shard \
  --source "$SOURCE" --destination "$ROOT/all" --feature-set all
"$PROJECT/uv" run python -m numerai_competitive.build_shard \
  --source "$SOURCE" --destination "$ROOT/medium" --feature-set medium
