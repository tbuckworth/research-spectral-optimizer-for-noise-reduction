#!/bin/bash
#SBATCH --partition=compute
#SBATCH --cpus-per-task=1
#SBATCH --mem=4000M
#SBATCH --time=00:15:00
#SBATCH --output=/mnt/nw/home/t.buckworth/numerai-live-candidates/slurm-%x-%j.out

set -euo pipefail
PROJECT=/mnt/nw/home/t.buckworth/numerai-live-candidates
BUNDLE=$PROJECT/results/bounded-live-bundles
export PYTHONPATH="$PROJECT/src"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
cd "$PROJECT"
"$PROJECT/.venv/bin/python" -m numerai_competitive.validate_live \
  --callable "$BUNDLE/${ARM}-predictor.pkl" \
  --live "$BUNDLE/live-fixture/live.parquet" \
  --benchmark "$BUNDLE/live-fixture/live_benchmark_models.parquet" \
  --output "$BUNDLE/${ARM}-runtime-audit.json" \
  --predictions "$BUNDLE/${ARM}-live_predictions.csv" \
  --max-bytes 4000000000 --max-seconds 600
