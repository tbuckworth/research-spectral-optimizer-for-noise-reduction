#!/bin/bash
#SBATCH --job-name=exp001-gate
#SBATCH --partition=compute
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm-%j.out

source ~/venv/bin/activate
set -x
cd "$SLURM_SUBMIT_DIR"

# Shard lives on /ephemeral (fast local scratch); copy from NFS if this
# node's /ephemeral doesn't have it yet.
EPH=/ephemeral/t.buckworth/researcher-shards
mkdir -p "$EPH"
if [ ! -f "$EPH/shard_train.parquet" ]; then
  cp data/shard_train.parquet "$EPH/"
fi
ls -la "$EPH"

python -u src/cluster_job.py --shard "$EPH/shard_train.parquet" --outdir out
