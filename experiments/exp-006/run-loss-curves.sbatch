#!/bin/bash
#SBATCH --job-name=spectral-loss-curves
#SBATCH --partition=compute
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=logs/loss-curves-%j.out
set -euo pipefail
source ~/venv/bin/activate
export PYTHONPATH="$PWD/src:$PWD/upstream"
python -u src/loss_curves.py
