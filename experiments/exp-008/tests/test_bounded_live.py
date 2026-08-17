import json

import numpy as np

from numerai_competitive.bounded_live import build_train_only_shard, create_freeze
from numerai_competitive.data import sha256


def test_bounded_freeze_requires_selection_to_match_outer_audit(tmp_path):
    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps({
        "status": "one_day_development_selection_frozen",
        "outer_reselection_allowed": False,
        "selected": {"adamw": [38], "spectral": [39], "paired_union": [38, 39]},
        "selected_updates": {"adamw": [20_000], "spectral": [20_000]},
    }))
    audit = tmp_path / "audit.json"
    audit.write_text(json.dumps({"status": "audit_complete",
                                 "selected": {"adamw": 38, "spectral": 39}}))
    search = tmp_path / "search.json"
    search.write_text(json.dumps({"configs": [
        {"arm": "adamw", "config_id": 38, "feature_set": "all"},
        {"arm": "spectral", "config_id": 39, "feature_set": "all"},
    ]}))
    protocol = tmp_path / "protocol.md"
    protocol.write_text("frozen protocol")
    output = tmp_path / "freeze.json"
    value = create_freeze(selection, audit, search, protocol, "a" * 40, output)
    assert value["status"] == "bounded_live_frozen"
    assert value["selected"]["adamw"] == {
        "config_id": 38, "updates": 20_000, "seeds": [0, 1, 2], "feature_set": "all",
    }
    assert value["upload_authorized"] is False and value["staking_authorized"] is False


def test_train_only_shard_hardlinks_frozen_train_arrays(tmp_path):
    train = tmp_path / "train"
    train.mkdir()
    np.save(train / "X_u8.npy", np.ones((4, 2), dtype=np.uint8))
    np.save(train / "targets_f32.npy", np.ones((4, 1), dtype=np.float32))
    np.save(train / "era_i16.npy", np.array([1, 1, 2, 2], dtype=np.int16))
    np.save(train / "benchmarks_f32.npy", np.zeros((4, 1), dtype=np.float32))
    (train / "manifest.json").write_text(json.dumps({
        "split": "train", "data_version": "v5.3", "feature_set": "all",
        "feature_names": ["a", "b"], "targets": ["target"], "benchmarks": ["bench"],
        "rows": 4, "eras": 2, "era_min": 1, "era_max": 2,
    }))
    freeze = tmp_path / "freeze.json"
    freeze.write_text(json.dumps({"status": "bounded_live_frozen", "target": "target"}))
    destination = tmp_path / "production"
    value = build_train_only_shard(train, freeze, destination)
    assert value["split"] == "production_train" and value["resolved_validation_rows"] == 0
    assert value["bounded_live_freeze_sha256"] == sha256(freeze)
    assert (destination / "X_u8.npy").stat().st_ino == (train / "X_u8.npy").stat().st_ino
