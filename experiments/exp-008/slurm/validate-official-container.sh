#!/bin/bash
set -euo pipefail

if [[ $# -ne 6 ]]; then
  echo "usage: $0 OFFICIAL_REPO MODEL LIVE BENCHMARK EXPECTED_PREDICTIONS OUTPUT_DIR" >&2
  exit 2
fi
OFFICIAL_REPO=$(realpath "$1")
MODEL=$(realpath "$2")
LIVE=$(realpath "$3")
BENCHMARK=$(realpath "$4")
EXPECTED=$(realpath "$5")
OUTPUT=$(realpath -m "$6")
PROJECT=${NUMERAI_PROJECT:-$(cd "$(dirname "$0")/.." && pwd)}
if [[ -x "$PROJECT/uv" ]]; then
  UV_RUNNER="$PROJECT/uv"
else
  UV_RUNNER=$(command -v uv || true)
  [[ -n $UV_RUNNER ]] || { echo "uv executable not found" >&2; exit 1; }
fi
for path in "$MODEL" "$LIVE" "$BENCHMARK" "$EXPECTED"; do
  [[ -f $path ]] || { echo "required input missing: $path" >&2; exit 1; }
done
[[ -d "$OFFICIAL_REPO/.git" && -f "$OFFICIAL_REPO/py3.12/Dockerfile" ]] \
  || { echo "official numerai-predict checkout is invalid" >&2; exit 1; }
[[ -z $(git -C "$OFFICIAL_REPO" status --porcelain) ]] \
  || { echo "official runner checkout is dirty" >&2; exit 1; }
RUNNER_COMMIT=$(git -C "$OFFICIAL_REPO" rev-parse HEAD)
[[ $RUNNER_COMMIT =~ ^[0-9a-f]{40}$ ]] || { echo "invalid runner commit" >&2; exit 1; }
if [[ -e "$OUTPUT/official-container-audit.json" ]]; then
  echo "official container audit already exists" >&2
  exit 1
fi
mkdir -p "$OUTPUT"
IMAGE="numerai_predict_py_3_12:${RUNNER_COMMIT:0:12}"
docker build --platform=linux/amd64 --provenance=false \
  --build-arg "GIT_REF=${RUNNER_COMMIT:0:12}" \
  --tag "$IMAGE" --file "$OFFICIAL_REPO/py3.12/Dockerfile" "$OFFICIAL_REPO"
IMAGE_ID=$(docker image inspect --format '{{.Id}}' "$IMAGE")
STARTED=$(date +%s)
timeout 600 docker run --rm --network none --cpus=1 --memory=4000000000 \
  --volume "$MODEL:/input/model.pkl:ro" \
  --volume "$LIVE:/input/live.parquet:ro" \
  --volume "$BENCHMARK:/input/benchmarks.parquet:ro" \
  --volume "$OUTPUT:/output" \
  "$IMAGE" --debug --model /input/model.pkl --dataset /input/live.parquet \
  --benchmarks /input/benchmarks.parquet --output_dir /output
ELAPSED=$(( $(date +%s) - STARTED ))
mapfile -t GENERATED < <(find "$OUTPUT" -maxdepth 1 -type f -name 'live_predictions-*.csv')
[[ ${#GENERATED[@]} -eq 1 ]] \
  || { echo "official runner must emit exactly one prediction CSV" >&2; exit 1; }
(
  cd "$PROJECT"
  "$UV_RUNNER" run --no-sync python -m numerai_competitive.official_container \
    --expected "$EXPECTED" --official "${GENERATED[0]}" \
    --output "$OUTPUT/official-container-audit.json" --runner-commit "$RUNNER_COMMIT" \
    --image-id "$IMAGE_ID" --elapsed-seconds "$ELAPSED" --max-seconds 600
)
