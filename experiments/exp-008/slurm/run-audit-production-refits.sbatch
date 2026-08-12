#!/bin/bash
#SBATCH --partition=compute
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --job-name=numerai-production-audit
#SBATCH --output=/mnt/nw/home/t.buckworth/numerai-competitive/slurm-%x-%j.out

set -euo pipefail
PROJECT=${NUMERAI_PROJECT:-/mnt/nw/home/t.buckworth/numerai-competitive}
SHARD=/ephemeral/t.buckworth/numerai-v5.3-production-${PRODUCTION_CODE_COMMIT:0:12}-${CANDIDATE_FEATURE_SET}
MODELS=()
for SEED in 0 1 2; do
  MODELS+=(--model "$PROJECT/results/production-refit-s${SEED}-${CANDIDATE_ARM}-c${CANDIDATE_CONFIG}/model.pt")
done
cd "$PROJECT"
"$PROJECT/uv" run --no-sync python -m numerai_competitive.audit_production_refits \
  --freeze results/freeze.json \
  --evaluation-marker results/official-validation/evaluation-complete.json \
  --shard "$SHARD" "${MODELS[@]}" --output results/production-refit-audit.json \
  --production-code-commit "$PRODUCTION_CODE_COMMIT" \
  --production-code-snapshot "$PROJECT/production-code-snapshot.json" --code-root "$PROJECT"
