#!/bin/bash
#SBATCH --partition=compute
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --job-name=n8-correct-select
#SBATCH --output=/mnt/nw/home/t.buckworth/numerai-corrected-live/slurm-%x-%j.out

set -euo pipefail
PROJECT=/mnt/nw/home/t.buckworth/numerai-corrected-live/experiments/exp-008
TRAIN=/ephemeral/t.buckworth/numerai-v5.3-corrected-live/all
PRODUCTION=/ephemeral/t.buckworth/numerai-v5.3-corrected-production-all
FREEZE=$PROJECT/results/corrected-live-freeze.json
export TMPDIR=/ephemeral/t.buckworth/tmp
export PYTHONPATH="$PROJECT/src"
mkdir -p "$TMPDIR"
cd "$PROJECT"
COMMIT=$(git rev-parse HEAD)
"$PROJECT/.venv/bin/python" -m numerai_competitive.corrected_live select \
  --result results/calibration-outer_1_inner_1-adamw/result.json \
  --result results/calibration-outer_1_inner_2-adamw/result.json \
  --result results/calibration-outer_1_inner_1-spectral/result.json \
  --result results/calibration-outer_1_inner_2-spectral/result.json \
  --protocol corrected-live-protocol.md --code-commit "$COMMIT" --output "$FREEZE"
if [[ ! -e "$PRODUCTION" ]]; then
  "$PROJECT/.venv/bin/python" -m numerai_competitive.corrected_live build-shard \
    --train "$TRAIN" --freeze "$FREEZE" --destination "$PRODUCTION"
fi
