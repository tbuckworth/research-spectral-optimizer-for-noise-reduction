import json
from pathlib import Path

import numpy as np
import pytest

from numerai_competitive.data import (
    TrainShard,
    ValidationShard,
    _feature_codes,
    _validate_freeze_manifest,
)


def test_loader_rejects_validation_manifest(tmp_path: Path):
    (tmp_path / "manifest.json").write_text(json.dumps({"split": "validation", "data_version": "v5.3"}))
    with pytest.raises(ValueError, match="train shard"):
        TrainShard.open(tmp_path)
    (tmp_path / "manifest.json").write_text(json.dumps({"split": "train", "data_version": "v5.3"}))
    with pytest.raises(ValueError, match="validation shard"):
        ValidationShard.open(tmp_path)


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


def test_feature_codes_accept_official_integer_and_quarter_scales():
    expected = np.arange(5, dtype=np.uint8)
    np.testing.assert_array_equal(_feature_codes(expected.astype(np.int8), 0), expected)
    np.testing.assert_array_equal(_feature_codes(expected.astype(float) / 4, 0), expected)
    with pytest.raises(ValueError, match="outside"):
        _feature_codes(np.array([0, 5], dtype=np.int8), 0)


def test_validation_freeze_requires_complete_explicit_authorization(tmp_path: Path):
    path = tmp_path / "freeze.json"
    base = {
        "status": "frozen", "protocol": "exp-008", "code_commit": "abc",
        "search_sha256": "def", "validation_reveal_authorized": True,
        "selected": {
            "adamw": {"config_id": 1, "updates": 100_000, "seeds": [0, 1, 2]},
            "spectral": {"config_id": 2, "updates": 100_000, "seeds": [0, 1, 2]},
        },
    }
    path.write_text(json.dumps(base))
    assert _validate_freeze_manifest(path)["status"] == "frozen"
    base["validation_reveal_authorized"] = False
    path.write_text(json.dumps(base))
    with pytest.raises(ValueError, match="remains sealed"):
        _validate_freeze_manifest(path)
