#!/bin/bash
#SBATCH --partition=compute
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --job-name=numerai-env
#SBATCH --output=/mnt/nw/home/t.buckworth/numerai-competitive/slurm-%x-%j.out

set -euo pipefail
PROJECT=/mnt/nw/home/t.buckworth/numerai-competitive
export UV_CACHE_DIR=/ephemeral/t.buckworth/uv-cache
cd "$PROJECT"
"$PROJECT/uv" sync --frozen
"$PROJECT/uv" run python -c \
  'import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))'
