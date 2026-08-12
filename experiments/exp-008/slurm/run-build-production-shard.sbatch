#!/bin/bash
#SBATCH --partition=compute
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --job-name=numerai-production-data
#SBATCH --output=/mnt/nw/home/t.buckworth/numerai-competitive/slurm-%x-%j.out

set -euo pipefail
PROJECT=/mnt/nw/home/t.buckworth/numerai-competitive
TRAIN=/ephemeral/t.buckworth/numerai-v5.3/${CANDIDATE_FEATURE_SET}
VALIDATION=/ephemeral/t.buckworth/numerai-v5.3-validation-all-${PROCEDURE_CODE_COMMIT:0:12}
DESTINATION=/ephemeral/t.buckworth/numerai-v5.3-production-${PRODUCTION_CODE_COMMIT:0:12}-${CANDIDATE_FEATURE_SET}
FREEZE=$PROJECT/results/freeze.json
EVALUATION=$PROJECT/results/official-validation/evaluation-complete.json
export UV_CACHE_DIR=/ephemeral/t.buckworth/uv-cache
cd "$PROJECT"
test ! -e "$DESTINATION"
"$PROJECT/uv" run --no-sync python -m numerai_competitive.production_data \
  --train "$TRAIN" --validation "$VALIDATION" --freeze "$FREEZE" \
  --evaluation-marker "$EVALUATION" --destination "$DESTINATION"
