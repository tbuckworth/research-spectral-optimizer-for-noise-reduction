#!/bin/bash
#SBATCH --job-name=exp003-sweep
#SBATCH --partition=compute
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=logs/slurm-%j.out

source ~/venv/bin/activate
set -x
cd "$SLURM_SUBMIT_DIR"

# The tuning-block eval shard is rsynced from the local machine after this
# job is submitted (submission is early to overlap queue wait). Wait for the
# .done sentinel, up to 45 min, then give up with a clear error.
DONE=data/shard_tuning.done
for i in $(seq 1 90); do
  [ -f "$DONE" ] && break
  echo "waiting for $DONE ($i/90)"; sleep 30
done
if [ ! -f "$DONE" ]; then
  echo "ERROR: tuning shard never arrived; aborting" >&2
  exit 1
fi

EPH=/ephemeral/t.buckworth/researcher-shards
mkdir -p "$EPH"
if [ ! -f "$EPH/shard_train.parquet" ]; then
  cp ../exp-001/data/shard_train.parquet "$EPH/"
fi
if [ ! -f "$EPH/shard_tuning.parquet" ]; then
  cp data/shard_tuning.parquet "$EPH/"
fi
ls -la "$EPH"

python -u src/sweep.py \
  --train-shard "$EPH/shard_train.parquet" \
  --tune-shard "$EPH/shard_tuning.parquet" \
  --ref ref --outdir out
