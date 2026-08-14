#!/bin/bash
#SBATCH --partition=compute
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --job-name=n8-live-data
#SBATCH --output=/mnt/nw/home/t.buckworth/numerai-live-candidates/slurm-%x-%j.out

set -euo pipefail
PROJECT=/mnt/nw/home/t.buckworth/numerai-live-candidates
TRAIN=/ephemeral/t.buckworth/numerai-v5.3/all
DEST=/ephemeral/t.buckworth/numerai-v5.3-bounded-live-all
cd "$PROJECT"
[[ -f "$TRAIN/manifest.json" ]] || { echo "existing audited train shard missing" >&2; exit 1; }
if [[ ! -e "$DEST" ]]; then
  "$PROJECT/.venv/bin/python" -m numerai_competitive.bounded_live build-shard \
    --train "$TRAIN" --freeze results/bounded-live-freeze.json --destination "$DEST"
fi
"$PROJECT/.venv/bin/python" -c \
  'from pathlib import Path; from numerai_competitive.data import ProductionShard; ProductionShard.open(Path("/ephemeral/t.buckworth/numerai-v5.3-bounded-live-all"))'
