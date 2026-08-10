import json

import pytest
import torch

from numerai_competitive.freeze import create_freeze


def _model(path, arm, config_id, seed):
    torch.save({
        "signature": f"{arm}-{seed}", "data_version": "v5.3",
        "train_config": {
            "arm": arm, "search_config_id": config_id, "seed": seed, "updates": 100,
            "target": "target_cyrusd_20", "feature_set": "medium",
        },
        "train_split": {
            "name": "all_train_refit", "train_eras": [f"{i:04d}" for i in range(1, 575)],
            "valid_eras": [], "purged_eras": [],
        },
    }, path)


def test_freeze_verifies_complete_model_provenance(tmp_path):
    search = tmp_path / "search.json"
    search.write_text(json.dumps({
        "protocol": "test", "configs": [
            {"arm": "adamw", "config_id": 1, "feature_set": "medium"},
            {"arm": "spectral", "config_id": 2, "feature_set": "medium"},
        ],
    }))
    protocol = tmp_path / "protocol.md"
    protocol.write_text("frozen")
    candidate = tmp_path / "candidate.json"
    candidate.write_text(json.dumps({
        "status": "frozen_train_only_selection",
        "selected": {"arm": "spectral", "benchmark": "v53_lgbm_ender20",
                     "model_weight": 0.75, "benchmark_weight": 0.25},
    }))
    adamw, spectral = [], []
    for seed in [0, 1]:
        adamw.append(tmp_path / f"adamw-{seed}.pt")
        spectral.append(tmp_path / f"spectral-{seed}.pt")
        _model(adamw[-1], "adamw", 1, seed)
        _model(spectral[-1], "spectral", 2, seed)
    output = tmp_path / "freeze.json"
    manifest = create_freeze(search, protocol, candidate, output, "abc", 1, 2, adamw, spectral,
                             100, [0, 1], True)
    assert output.is_file() and manifest["status"] == "frozen"
    assert manifest["selected"]["spectral"]["model_signatures"] == ["spectral-0", "spectral-1"]
    assert manifest["candidate_transform"]["model_weight"] == 0.75
    with pytest.raises(ValueError, match="authorization"):
        create_freeze(search, protocol, candidate, output, "abc", 1, 2, adamw, spectral,
                      100, [0, 1], False)
