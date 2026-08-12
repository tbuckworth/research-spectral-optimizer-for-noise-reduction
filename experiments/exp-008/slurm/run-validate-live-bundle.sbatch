#!/bin/bash
#SBATCH --partition=compute
#SBATCH --cpus-per-task=1
#SBATCH --mem=4000M
#SBATCH --time=00:15:00
#SBATCH --output=/mnt/nw/home/t.buckworth/numerai-competitive/slurm-%x-%j.out

set -euo pipefail
PROJECT=${NUMERAI_PROJECT:-/mnt/nw/home/t.buckworth/numerai-competitive}
BUNDLE="$PROJECT/results/live-bundle"
export UV_CACHE_DIR=/ephemeral/t.buckworth/uv-cache
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
cd "$PROJECT"
"$PROJECT/uv" run --no-sync python -m numerai_competitive.validate_live \
  --callable "$BUNDLE/predictor.pkl" \
  --live "$BUNDLE/live-fixture/live.parquet" \
  --benchmark "$BUNDLE/live-fixture/live_benchmark_models.parquet" \
  --output "$BUNDLE/runtime-audit.json" --predictions "$BUNDLE/live_predictions.csv" \
  --max-bytes 4000000000 --max-seconds 600
