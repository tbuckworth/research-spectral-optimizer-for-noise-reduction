#!/bin/bash
#SBATCH --partition=compute
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=23:30:00
#SBATCH --output=/mnt/nw/home/t.buckworth/numerai-competitive/slurm-%x-%j.out

set -euo pipefail
PROJECT=/mnt/nw/home/t.buckworth/numerai-competitive
SOURCE="$PROJECT/results/sealed-v5.3-source-${CODE_COMMIT:0:12}"
SHARD="/ephemeral/t.buckworth/numerai-v5.3-validation-all-${CODE_COMMIT:0:12}"
FREEZE="$PROJECT/results/freeze.json"
export UV_CACHE_DIR=/ephemeral/t.buckworth/uv-cache
cd "$PROJECT"

ADAMW_MODELS=()
SPECTRAL_MODELS=()
for SEED in 0 1 2; do
  ADAMW_MODELS+=(--adamw-model \
    "$PROJECT/results/final-refit-u100000-s${SEED}-adamw-c${ADAMW_CONFIG}/model.pt")
  SPECTRAL_MODELS+=(--spectral-model \
    "$PROJECT/results/final-refit-u100000-s${SEED}-spectral-c${SPECTRAL_CONFIG}/model.pt")
done
"$PROJECT/uv" run --no-sync python -m numerai_competitive.freeze \
  --search configs/search-v1.json --protocol fidelity-protocol.md \
  --candidate-plan results/candidate-plan.json --output "$FREEZE" \
  --code-commit "$CODE_COMMIT" --code-snapshot "$PROJECT/code-snapshot.json" \
  --code-root "$PROJECT" --adamw-config "$ADAMW_CONFIG" \
  --spectral-config "$SPECTRAL_CONFIG" "${ADAMW_MODELS[@]}" "${SPECTRAL_MODELS[@]}" \
  --updates 100000 --seed 0 --seed 1 --seed 2 --authorize-validation-reveal
"$PROJECT/uv" run --no-sync python -m numerai_competitive.download_validation \
  --destination "$SOURCE" --freeze "$FREEZE"
if [[ -e "$SHARD" ]]; then
  echo "sealed validation shard destination already exists" >&2
  exit 1
fi
"$PROJECT/uv" run --no-sync python -m numerai_competitive.build_shard \
  --source "$SOURCE" --destination "$SHARD" --feature-set all \
  --split validation --freeze-manifest "$FREEZE"
