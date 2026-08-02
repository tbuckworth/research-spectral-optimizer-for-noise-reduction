#!/bin/bash
#SBATCH --partition=compute
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --job-name=audit-rerun-exp004
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/slurm-%j.out

set -x
cd /mnt/nw/home/t.buckworth/researcher-runs/2026-07-31-spectral-optimizer-for-noise-reduction-on-financial-timeseri/audit-rerun-exp-004
source ~/venv/bin/activate
EPH=/ephemeral/t.buckworth/researcher-shards
mkdir -p "$EPH"
RUNROOT=/mnt/nw/home/t.buckworth/researcher-runs/2026-07-31-spectral-optimizer-for-noise-reduction-on-financial-timeseri
if [ ! -f "$EPH/shard_train.parquet" ]; then
  cp "$RUNROOT/exp-001/data/shard_train.parquet" "$EPH/"
fi
if [ ! -f "$EPH/shard_verdict.parquet" ]; then
  cp "$RUNROOT/exp-004/data/shard_verdict.parquet" "$EPH/"
fi
cp data/shard_verdict_audit.parquet "$EPH/"
ls -la "$EPH"
python -u src/rerun_job.py \
  --train-shard "$EPH/shard_train.parquet" \
  --verdict-shard "$EPH/shard_verdict.parquet" \
  --verdict-shard2 "$EPH/shard_verdict_audit.parquet" \
  --ref ref --outdir out
