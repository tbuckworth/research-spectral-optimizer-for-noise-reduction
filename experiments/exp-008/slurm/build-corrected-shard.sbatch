#!/bin/bash
#SBATCH --partition=compute
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --job-name=n8-correct-data
#SBATCH --output=/mnt/nw/home/t.buckworth/numerai-corrected-live/slurm-%x-%j.out

set -euo pipefail
PROJECT=/mnt/nw/home/t.buckworth/numerai-corrected-live/experiments/exp-008
SOURCE=/mnt/nw/home/t.buckworth/numerai-v5.3-source
DEST=/ephemeral/t.buckworth/numerai-v5.3-corrected-live/all
export UV_CACHE_DIR=/ephemeral/t.buckworth/uv-cache
export TMPDIR=/ephemeral/t.buckworth/tmp
mkdir -p "$TMPDIR" "$(dirname "$DEST")"
cd "$PROJECT"
if [[ ! -f "$DEST/manifest.json" ]]; then
  test ! -e "$DEST"
  "$PROJECT/.venv/bin/python" -m numerai_competitive.build_shard \
    --source "$SOURCE" --destination "$DEST" --feature-set all
fi
"$PROJECT/.venv/bin/python" -c \
  'from pathlib import Path; from numerai_competitive.data import TrainShard; TrainShard.open(Path("/ephemeral/t.buckworth/numerai-v5.3-corrected-live/all"))'
