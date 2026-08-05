#!/bin/bash
#SBATCH --job-name=stable-spectral-test
#SBATCH --partition=compute
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/comparison-%j.out
set -euo pipefail
source ~/venv/bin/activate
export PYTHONPATH="$PWD/src:$PWD/upstream"
export STOP_AFTER_SWEEP=0
python -u src/run_corrected_training.py
