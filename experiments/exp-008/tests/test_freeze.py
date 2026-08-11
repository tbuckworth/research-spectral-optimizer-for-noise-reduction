import hashlib
import json

import pytest
import torch

from numerai_competitive.data import sha256
from numerai_competitive.freeze import create_freeze


def _code_snapshot(tmp_path, commit):
    root = tmp_path / "code"
    names = ["pyproject.toml", "uv.lock", "fidelity-protocol.md",
             "src/numerai_competitive/code_snapshot.py",
             "src/numerai_competitive/freeze.py"]
    files = {}
    for name in names:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name)
        files[name] = sha256(path)
    digest = hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode())
    snapshot = tmp_path / "code-snapshot.json"
    snapshot.write_text(json.dumps({
        "status": "complete", "code_commit": commit, "source_prefix": "x",
        "files": files, "file_count": len(files), "file_map_sha256": digest.hexdigest(),
    }))
    return root, snapshot


def _model(path, arm, config_id, seed, updates=5000):
    torch.save({
        "signature": f"{arm}-{seed}", "data_version": "v5.3",
        "train_config": {
            "arm": arm, "search_config_id": config_id, "seed": seed, "updates": updates,
            "target": "target", "benchmark": "v53_lgbm_ender60", "feature_set": "medium",
        },
        "train_split": {
            "name": "sealed_validation_refit_60d",
            "train_eras": [f"{i:04d}" for i in range(1, 559)],
            "valid_eras": [], "purged_eras": [f"{i:04d}" for i in range(559, 575)],
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
        "selected": {"arm": "spectral", "benchmark": "v53_lgbm_ender60",
                     "model_weight": 0.75, "benchmark_weight": 0.25},
    }))
    adamw, spectral = [], []
    for seed in [0, 1]:
        adamw.append(tmp_path / f"adamw-{seed}.pt")
        spectral.append(tmp_path / f"spectral-{seed}.pt")
        _model(adamw[-1], "adamw", 1, seed)
        _model(spectral[-1], "spectral", 2, seed)
    output = tmp_path / "freeze.json"
    commit = "a" * 40
    code_root, snapshot = _code_snapshot(tmp_path, commit)
    manifest = create_freeze(search, protocol, candidate, output, commit, snapshot, code_root,
                             1, 2, adamw, spectral, 5000, [0, 1], True)
    assert output.is_file() and manifest["status"] == "frozen"
    assert manifest["selected"]["spectral"]["model_signatures"] == ["spectral-0", "spectral-1"]
    assert manifest["candidate_transform"]["model_weight"] == 0.75
    with pytest.raises(ValueError, match="authorization"):
        create_freeze(search, protocol, candidate, output, commit, snapshot, code_root,
                      1, 2, adamw, spectral, 5000, [0, 1], False)


def test_freeze_preserves_arm_specific_update_budgets(tmp_path):
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
        "selected": {"arm": "adamw", "benchmark": "v53_lgbm_ender60",
                     "model_weight": 1.0, "benchmark_weight": 0.0},
    }))
    adamw, spectral = tmp_path / "adamw.pt", tmp_path / "spectral.pt"
    _model(adamw, "adamw", 1, 0, 5000)
    _model(spectral, "spectral", 2, 0, 20000)
    commit = "b" * 40
    code_root, snapshot = _code_snapshot(tmp_path, commit)
    manifest = create_freeze(
        search, protocol, candidate, tmp_path / "freeze.json", commit, snapshot, code_root,
        1, 2, [adamw], [spectral], {"adamw": 5000, "spectral": 20000}, [0], True,
    )
    assert manifest["selected"]["adamw"]["updates"] == 5000
    assert manifest["selected"]["spectral"]["updates"] == 20000
