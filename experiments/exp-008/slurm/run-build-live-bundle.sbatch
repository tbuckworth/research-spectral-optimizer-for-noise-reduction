#!/bin/bash
#SBATCH --partition=compute
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/mnt/nw/home/t.buckworth/numerai-competitive/slurm-%x-%j.out

set -euo pipefail
PROJECT=${NUMERAI_PROJECT:-/mnt/nw/home/t.buckworth/numerai-competitive}
FREEZE="$PROJECT/results/freeze.json"
PRODUCTION_AUDIT="$PROJECT/results/production-refit-audit.json"
BUNDLE="$PROJECT/results/live-bundle"
FIXTURE="$BUNDLE/live-fixture"
CALLABLE="$BUNDLE/predictor.pkl"
export UV_CACHE_DIR=/ephemeral/t.buckworth/uv-cache
cd "$PROJECT"
[[ -f results/official-validation/evaluation-complete.json ]] \
  || { echo "sealed evaluation is incomplete" >&2; exit 1; }
[[ ! -e $BUNDLE ]] || { echo "live bundle output already exists" >&2; exit 1; }
MODELS=()
for SEED in 0 1 2; do
  MODELS+=(--model \
    "$PROJECT/results/production-refit-s${SEED}-${CANDIDATE_ARM}-c${CANDIDATE_CONFIG}/model.pt")
done
"$PROJECT/uv" run --no-sync python -m numerai_competitive.live \
  "${MODELS[@]}" --freeze "$FREEZE" --production-audit "$PRODUCTION_AUDIT" \
  --output "$CALLABLE" --batch-size 4096
"$PROJECT/uv" run --no-sync python -m numerai_competitive.download_live \
  --destination "$FIXTURE" --freeze "$FREEZE"
