import hashlib
import json

import numpy as np
import torch

from numerai_competitive.audit_production_refits import audit_production_refits
from numerai_competitive.data import sha256


def test_audit_production_refits_keeps_live_models_separate_from_validation_models(tmp_path):
    code_root = tmp_path / "code"
    files = {}
    for name in (
        "pyproject.toml", "uv.lock", "fidelity-protocol.md",
        "src/numerai_competitive/code_snapshot.py", "src/numerai_competitive/freeze.py",
    ):
        path = code_root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name)
        files[name] = sha256(path)
    digest = hashlib.sha256(
        json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    code_commit = "a" * 40
    snapshot = tmp_path / "production-code-snapshot.json"
    snapshot.write_text(json.dumps({
        "status": "complete", "code_commit": code_commit, "files": files,
        "file_count": len(files), "file_map_sha256": digest,
    }))
    freeze = tmp_path / "freeze.json"
    freeze.write_text(json.dumps({
        "status": "frozen", "code_commit": "b" * 40,
        "primary_benchmark": "v53_lgbm_ender60",
        "candidate_transform": {"arm": "spectral"},
        "selected": {"spectral": {
            "config_id": 7, "updates": 5000, "seeds": [0, 1], "feature_set": "medium",
        }},
    }))
    evaluation = tmp_path / "evaluation-complete.json"
    evaluation.write_text(json.dumps({"status": "complete"}))
    shard = tmp_path / "production"
    shard.mkdir()
    np.save(shard / "X_u8.npy", np.zeros((4, 2), np.uint8))
    np.save(shard / "targets_f32.npy", np.zeros((4, 1), np.float32))
    np.save(shard / "era_i16.npy", np.asarray([1, 1, 2, 3], np.int16))
    np.save(shard / "benchmarks_f32.npy", np.zeros((4, 1), np.float32))
    identity = "f" * 64
    (shard / "manifest.json").write_text(json.dumps({
        "split": "production_train", "data_version": "v5.3", "rows": 4,
        "feature_set": "medium", "feature_names": ["a", "b"],
        "targets": ["target"], "benchmarks": ["v53_lgbm_ender60"],
        "freeze_manifest_sha256": sha256(freeze),
        "sealed_evaluation_sha256": sha256(evaluation),
        "resolved_validation_rows": 1, "training_data_sha256": identity, "era_max": 3,
    }))
    models = []
    for seed in [0, 1]:
        path = tmp_path / f"model-{seed}.pt"
        torch.save({
            "signature": f"production-{seed}", "training_data_sha256": identity,
            "train_config": {
                "arm": "spectral", "search_config_id": 7, "seed": seed,
                "updates": 5000, "target": "target", "benchmark": "v53_lgbm_ender60",
                "feature_set": "medium",
            },
            "train_split": {
                "name": "production_live_refit_resolved", "train_eras": ["0001", "0002", "0003"],
                "valid_eras": [], "purged_eras": [],
            },
        }, path)
        models.append(path)
    output = tmp_path / "audit.json"
    report = audit_production_refits(
        freeze, evaluation, shard, models, output, code_commit, snapshot, code_root,
    )
    assert report["status"] == "audit_complete"
    assert report["training_era_max"] == 3
    assert report["model_signatures"] == ["production-0", "production-1"]
    assert output.is_file()
