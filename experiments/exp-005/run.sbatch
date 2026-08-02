#!/bin/bash
#SBATCH --job-name=exp005-seq
#SBATCH --partition=compute
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:50:00
#SBATCH --output=logs/slurm-%j.out

source ~/venv/bin/activate
set -x
cd "$SLURM_SUBMIT_DIR"

EPH=/ephemeral/t.buckworth/researcher-shards
mkdir -p "$EPH"
if [ ! -f "$EPH/shard_train.parquet" ]; then
  cp ../exp-001/data/shard_train.parquet "$EPH/"
fi
if [ ! -f "$EPH/shard_verdict.parquet" ]; then
  cp ../exp-004/data/shard_verdict.parquet "$EPH/"
fi
ls -la "$EPH"

python -u src/seq_job.py \
  --train-shard "$EPH/shard_train.parquet" \
  --verdict-shard "$EPH/shard_verdict.parquet" \
  --ref ref --outdir out
