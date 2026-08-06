#!/bin/bash
#SBATCH --job-name=spectral-million
#SBATCH --partition=compute
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=logs/slurm-%A_%a.out
#SBATCH --array=0-3
set -euo pipefail

source ~/venv/bin/activate
test -n "${CUDA_VISIBLE_DEVICES:-}"
nvidia-smi -L

PERSIST="$HOME/spectral-exp007"
SCRATCH="/ephemeral/$USER/spectral-million-$SLURM_JOB_ID"
mkdir -p "$SCRATCH/data" "$PERSIST/out" "$PERSIST/logs"
rsync -a "$PERSIST/data/" "$SCRATCH/data/"
export NUMERAI_SHARD="$SCRATCH/data"
cd "$PERSIST"

common=(--steps 1000000 --eval-every 10000 --seed 20260805
        --run-tag million --full-valid-test)
case "$SLURM_ARRAY_TASK_ID" in
  0) python -u src/run_overfit.py --arm adamw --rank 0 "${common[@]}" ;;
  1) python -u src/run_overfit.py --arm spectral --rank 2048 --blockwise \
       --block-size 250000 --projection-mode top --relative-eig-tol 1e-12 "${common[@]}" ;;
  2) python -u src/run_overfit.py --arm spectral --rank 3072 --blockwise \
       --block-size 250000 --projection-mode top --relative-eig-tol 1e-12 "${common[@]}" ;;
  3) python -u src/run_overfit.py --arm spectral --rank 4096 --blockwise \
       --block-size 250000 --projection-mode top --relative-eig-tol 1e-12 "${common[@]}" ;;
  *) exit 2 ;;
esac
