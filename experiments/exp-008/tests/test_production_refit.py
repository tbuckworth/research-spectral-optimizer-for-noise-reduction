import json

import numpy as np
import torch

from numerai_competitive.data import sha256
from numerai_competitive.production_refit import run_production_refit


def test_production_refit_uses_frozen_config_and_resolved_data_identity(tmp_path):
    search = tmp_path / "search.json"
    search.write_text(json.dumps({"configs": [{
        "activation": "silu", "arm": "adamw", "batch_mode": "row", "batch_size": 2,
        "clip_grad_norm": 1.0, "config_id": 3, "depth": 2, "dropout": 0.0,
        "feature_set": "medium", "learning_rate": 0.001, "loss": "mse",
        "normalization": "none", "residual": True, "schedule": "constant",
        "target": "target", "warmup_fraction": 0.0, "weight_decay": 0.0, "width": 4,
    }]}))
    freeze = tmp_path / "freeze.json"
    freeze.write_text(json.dumps({
        "status": "frozen", "candidate_transform": {"arm": "adamw"},
        "selected": {"adamw": {
            "config_id": 3, "updates": 2, "seeds": [0], "feature_set": "medium",
        }},
    }))
    evaluation = tmp_path / "evaluation-complete.json"
    evaluation.write_text(json.dumps({"status": "complete"}))
    shard = tmp_path / "production"
    shard.mkdir()
    rng = np.random.default_rng(1)
    np.save(shard / "X_u8.npy", rng.integers(0, 5, (8, 2), dtype=np.uint8))
    np.save(shard / "targets_f32.npy", rng.random((8, 1), dtype=np.float32))
    np.save(shard / "era_i16.npy", np.repeat([1, 2, 3, 4], 2).astype(np.int16))
    np.save(shard / "benchmarks_f32.npy", rng.random((8, 1), dtype=np.float32))
    identity = "e" * 64
    (shard / "manifest.json").write_text(json.dumps({
        "split": "production_train", "data_version": "v5.3", "rows": 8,
        "feature_set": "medium", "feature_names": ["a", "b"],
        "targets": ["target"], "benchmarks": ["v53_lgbm_ender60"],
        "freeze_manifest_sha256": sha256(freeze),
        "sealed_evaluation_sha256": sha256(evaluation),
        "resolved_validation_rows": 2, "training_data_sha256": identity,
    }))
    output = tmp_path / "output"
    result = run_production_refit(search, shard, freeze, evaluation, output, seed=0, device="cpu")
    assert result["status"] == "complete" and result["validation"] is None
    assert result["split"]["name"] == "production_live_refit_resolved"
    assert result["training_data_sha256"] == identity
    artifact = torch.load(output / "model.pt", weights_only=False)
    assert artifact["training_data_sha256"] == identity
    assert artifact["train_config"]["search_config_id"] == 3
