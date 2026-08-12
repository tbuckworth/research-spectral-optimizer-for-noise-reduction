import json

import numpy as np

from numerai_competitive.data import ProductionShard, sha256
from numerai_competitive.production_data import build_production_shard


def _arrays(root, split, features, eras, targets, freeze_hash=None):
    root.mkdir()
    rows = len(eras)
    all_features = ["a", "b", "c"]
    X = np.arange(rows * len(all_features), dtype=np.uint8).reshape(rows, -1) % 5
    np.save(root / "X_u8.npy", X[:, [all_features.index(name) for name in features]])
    np.save(root / "targets_f32.npy", np.asarray(targets, dtype=np.float32)[:, None])
    np.save(root / "era_i16.npy", np.asarray(eras, dtype=np.int16))
    np.save(root / "benchmarks_f32.npy", np.linspace(0, 1, rows, dtype=np.float32)[:, None])
    manifest = {
        "split": split, "data_version": "v5.3", "feature_set": "medium" if len(features) == 2 else "all",
        "feature_names": features, "targets": ["target"], "benchmarks": ["v53_lgbm_ender60"],
        "rows": rows,
    }
    if freeze_hash:
        manifest["freeze_manifest_sha256"] = freeze_hash
    (root / "manifest.json").write_text(json.dumps(manifest))


def test_production_shard_uses_only_resolved_validation_and_frozen_features(tmp_path):
    freeze = tmp_path / "freeze.json"
    freeze.write_text(json.dumps({
        "status": "frozen", "candidate_transform": {"arm": "adamw"},
        "selected": {"adamw": {"feature_set": "medium"}},
    }))
    train, validation = tmp_path / "train", tmp_path / "validation"
    _arrays(train, "train", ["a", "c"], [1, 1, 2], [0.1, 0.2, 0.3])
    _arrays(validation, "validation", ["a", "b", "c"], [3, 3, 4],
            [0.4, float("nan"), 0.6], sha256(freeze))
    evaluation = tmp_path / "official-validation"
    evaluation.mkdir()
    report = evaluation / "official-validation-report.json"
    report.write_text(json.dumps({"freeze_manifest_sha256": sha256(freeze)}))
    marker = evaluation / "evaluation-complete.json"
    marker.write_text(json.dumps({
        "status": "complete", "artifacts": {report.name: sha256(report)},
    }))
    destination = tmp_path / "production"
    manifest = build_production_shard(train, validation, freeze, marker, destination)
    shard = ProductionShard.open(destination)
    assert manifest["frozen_train_rows"] == 3
    assert manifest["resolved_validation_rows"] == 2
    assert shard.eras.tolist() == [1, 1, 2, 3, 4]
    assert shard.targets[:, 0].tolist() == np.asarray([0.1, 0.2, 0.3, 0.4, 0.6], np.float32).tolist()
    assert shard.X.shape == (5, 2)
    assert shard.manifest["training_data_sha256"]
