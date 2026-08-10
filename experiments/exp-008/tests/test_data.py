import json
from pathlib import Path

import numpy as np
import pytest

from numerai_competitive.data import TrainShard


def test_loader_rejects_validation_manifest(tmp_path: Path):
    (tmp_path / "manifest.json").write_text(json.dumps({"split": "validation", "data_version": "v5.3"}))
    with pytest.raises(ValueError, match="train shard"):
        TrainShard.open(tmp_path)


def test_loader_rejects_mismatched_rows(tmp_path: Path):
    (tmp_path / "manifest.json").write_text(json.dumps({
        "split": "train", "data_version": "v5.3", "rows": 2,
        "targets": ["target_cyrusd_20"], "benchmarks": ["v53_lgbm_ender20"]
    }))
    for name, shape, dtype in [
        ("X_u8.npy", (1, 1), np.uint8), ("targets_f32.npy", (1, 1), np.float32),
        ("era_i16.npy", (1,), np.int16), ("benchmarks_f32.npy", (1, 1), np.float32)
    ]:
        np.save(tmp_path / name, np.zeros(shape, dtype=dtype))
    with pytest.raises(ValueError, match="row counts"):
        TrainShard.open(tmp_path)
