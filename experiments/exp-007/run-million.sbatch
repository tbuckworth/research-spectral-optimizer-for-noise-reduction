#!/bin/bash
#SBATCH --job-name=spectral-million
#SBATCH --partition=compute
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
# The account QOS caps free-compute jobs at 12 hours.  The deterministic
# checkpoints below allow long arms to continue across allocations.
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm-%A_%a.out
#SBATCH --array=0-3
set -euo pipefail

source ~/venv/bin/activate
test -n "${CUDA_VISIBLE_DEVICES:-}"
nvidia-smi -L

PERSIST="$HOME/spectral-exp007"
SCRATCH_DATA="/ephemeral/$USER/spectral-million-data"
mkdir -p "$SCRATCH_DATA" "$PERSIST/out" "$PERSIST/logs"
# The free compute partition is one shared L40 worker. Stage the immutable shard
# once even when all four array tasks begin together.
flock "$SCRATCH_DATA.lock" rsync -a "$PERSIST/data/" "$SCRATCH_DATA/"
export NUMERAI_SHARD="$SCRATCH_DATA"
cd "$PERSIST"

common=(--steps 1000000 --eval-every 10000 --seed 20260805
        --run-tag million --full-valid-test --checkpoint-every 100000)
is_complete() {
  python -c 'import json, sys
with open(sys.argv[1]) as handle:
    payload = json.load(handle)
ok = payload.get("steps") == 1_000_000 and payload.get("curve") and payload["curve"][-1].get("step") == 1_000_000
raise SystemExit(0 if ok else 1)' "$1"
}
case "$SLURM_ARRAY_TASK_ID" in
  0) output="out/adamw-r0-seed20260805-million.json"
     test -f "$output" && is_complete "$output" && { echo "COMPLETE $output"; exit 0; }
     python -u src/run_overfit.py --arm adamw --rank 0 "${common[@]}" ;;
  1) output="out/spectral-top-r2048-seed20260805-million.json"
     test -f "$output" && is_complete "$output" && { echo "COMPLETE $output"; exit 0; }
     python -u src/run_overfit.py --arm spectral --rank 2048 --blockwise \
       --block-size 250000 --projection-mode top --relative-eig-tol 1e-12 "${common[@]}" ;;
  2) output="out/spectral-top-r3072-seed20260805-million.json"
     test -f "$output" && is_complete "$output" && { echo "COMPLETE $output"; exit 0; }
     python -u src/run_overfit.py --arm spectral --rank 3072 --blockwise \
       --block-size 250000 --projection-mode top --relative-eig-tol 1e-12 "${common[@]}" ;;
  3) output="out/spectral-top-r4096-seed20260805-million.json"
     test -f "$output" && is_complete "$output" && { echo "COMPLETE $output"; exit 0; }
     python -u src/run_overfit.py --arm spectral --rank 4096 --blockwise \
       --block-size 250000 --projection-mode top --relative-eig-tol 1e-12 "${common[@]}" ;;
  *) exit 2 ;;
esac
