#!/bin/bash
#SBATCH --job-name=stable-spectral-diag
#SBATCH --partition=compute
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/diagnostic-%j.out
set -euo pipefail
source ~/venv/bin/activate
export PYTHONPATH="$PWD/src:$PWD/upstream"
export NUMERAI_PROTOCOL="$PWD/upstream/protocol.json"
python -u src/diagnose_numerai.py
