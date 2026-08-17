#!/bin/bash
#SBATCH --partition=compute
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --job-name=n8-live-bundle
#SBATCH --output=/mnt/nw/home/t.buckworth/numerai-live-candidates/slurm-%x-%j.out

set -euo pipefail
PROJECT=/mnt/nw/home/t.buckworth/numerai-live-candidates
BUNDLE=$PROJECT/results/bounded-live-bundles
export PYTHONPATH="$PROJECT/src"
cd "$PROJECT"
"$PROJECT/.venv/bin/python" -m numerai_competitive.bounded_live export \
  --freeze results/bounded-live-freeze.json \
  --audit results/bounded-live-refit-audit.json --output "$BUNDLE" --batch-size 2048
"$PROJECT/.venv/bin/python" -m numerai_competitive.download_live \
  --destination "$BUNDLE/live-fixture" --freeze results/bounded-live-freeze.json
