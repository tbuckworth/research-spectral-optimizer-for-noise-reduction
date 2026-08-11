#!/bin/bash
#SBATCH --partition=compute
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --job-name=numerai-target-smoke
#SBATCH --output=/mnt/nw/home/t.buckworth/numerai-competitive/slurm-%x-%j.out

set -euo pipefail
PROJECT=/mnt/nw/home/t.buckworth/numerai-competitive
SHARDS=/ephemeral/t.buckworth/numerai-v5.3
OUTPUT="$PROJECT/results/smoke-main-target-adamw"
export UV_CACHE_DIR=/ephemeral/t.buckworth/uv-cache
export CUBLAS_WORKSPACE_CONFIG=:4096:8

cd "$PROJECT"
test ! -e "$OUTPUT"
"$PROJECT/uv" run --no-sync python - <<'PY'
import json
from pathlib import Path

import numpy as np
import torch

root = Path("/ephemeral/t.buckworth/numerai-v5.3")
for feature_set in ("medium", "all"):
    manifest = json.loads((root / feature_set / "manifest.json").read_text())
    assert manifest["split"] == "train"
    assert manifest["targets"][0] == "target"
    assert manifest["benchmarks"] == ["v53_lgbm_ender20", "v53_lgbm_ender60"]
    targets = np.load(root / feature_set / "targets_f32.npy", mmap_mode="r")
    assert targets.shape[1] == len(manifest["targets"]) == 5
    assert np.isfinite(targets[:, 0]).all()
assert torch.cuda.is_available(), "CUDA unavailable; refusing CPU fallback"
print("main-target shard checks passed", torch.cuda.get_device_name(0))
PY

"$PROJECT/.venv/bin/python" -m numerai_competitive.run_task \
  --search configs/search-v1.json --shards "$SHARDS" --output "$OUTPUT" \
  --arm adamw --config-id 0 --split outer_1_inner_1 --updates 50 --seed 20260811

"$PROJECT/uv" run --no-sync python - <<'PY'
import json
import math
from pathlib import Path

result = json.loads(Path(
    "/mnt/nw/home/t.buckworth/numerai-competitive/results/"
    "smoke-main-target-adamw/result.json"
).read_text())
assert result["status"] == "complete"
assert result["config"]["target"] == "target"
assert result["config"]["benchmark"] == "v53_lgbm_ender60"
assert result["split"]["train_eras"][-1] == "0140"
assert result["split"]["purged_eras"] == [f"{era:04d}" for era in range(141, 157)]
assert result["split"]["valid_eras"][0] == "0157"
assert result["updates"] == 50
assert math.isfinite(result["validation"]["corr"]["mean"])
print("main-target training/scoring smoke passed", result["validation"]["corr"]["mean"])
PY
