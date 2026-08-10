import hashlib
import json

import numpy as np
import pytest

from numerai_competitive.materialize import materialize_config
from numerai_competitive.summarize import _verify_search_identity, collect_stage, write_summary


def _result(root, arm, config_id):
    path = root / f"stage-fold-u10-s0-{arm}-c{config_id}"
    path.mkdir()
    np.savez_compressed(
        path / "validation_predictions.npz", row_index=np.arange(4), era=np.repeat([1, 2], 2),
        target=np.linspace(0, 1, 4), benchmark=np.linspace(1, 0, 4),
        prediction=np.linspace(0.1, 0.9, 4), per_era_corr=np.array([0.1, 0.2]),
        per_era_bmc=np.array([0.01, 0.02]),
    )
    path.joinpath("result.json").write_text(json.dumps({
        "status": "complete", "signature": f"{arm}-{config_id}",
        "split": {"name": "fold"}, "updates": 10, "examples": 40,
        "config": {"arm": arm, "seed": 0, "search_config_id": config_id},
        "prediction_file": "validation_predictions.npz",
        "validation": {"rows": 4, "corr": {"mean": 0.01 + config_id, "sharpe": 1.0},
                       "bmc": {"mean": 0.002}},
        "parameter_count": 10, "peak_cuda_memory_bytes": 2**30,
        "logs": [{"elapsed_seconds": 3.0}],
    }))


def test_collect_and_plot_complete_paired_stage(tmp_path):
    for config_id in range(2):
        for arm in ("adamw", "spectral"):
            _result(tmp_path, arm, config_id)
    frame = collect_stage(tmp_path, split="fold", updates=10, seed=0, expected_configs=2)
    write_summary(frame, tmp_path / "summary")
    assert len(frame) == 4
    assert frame["result_sha256"].str.len().eq(64).all()
    assert (tmp_path / "summary" / "paired-corr.png").is_file()


def test_collect_rejects_incomplete_stage(tmp_path):
    _result(tmp_path, "adamw", 0)
    with pytest.raises(ValueError, match="incomplete"):
        collect_stage(tmp_path, split="fold", updates=10, seed=0, expected_configs=1)


def _draw(arm="adamw"):
    draw = {
        "arm": arm, "config_id": 7, "feature_set": "medium", "width": 8,
        "depth": 2, "residual": False, "normalization": "none", "activation": "relu",
        "dropout": 0.0, "target": "target_cyrusd_20", "batch_mode": "rows",
        "batch_size": 4, "learning_rate": 1e-3, "weight_decay": 0.0, "loss": "mse",
        "schedule": "constant", "warmup_fraction": 0.0, "clip_grad_norm": 0.0,
    }
    if arm == "spectral":
        draw |= {"rank": 2, "decay": 0.99, "filter_strength": 1.0,
                 "filter_warmup": 1, "filter_update_every": 1, "filter_mode": "learned"}
    return draw


def _legacy_result(draw):
    config = materialize_config(draw, input_dim=3, updates=10, seed=0)
    config.pop("search_config_id")
    split = {"name": "fold", "train_eras": ["0001"], "purged_eras": [],
             "valid_eras": ["0002"]}
    signature = hashlib.sha256(json.dumps(
        {"config": config, "split": split}, sort_keys=True,
    ).encode()).hexdigest()
    return {"config": config, "split": split, "signature": signature}


def test_legacy_result_identity_is_verified_against_frozen_draw_and_signature():
    draw = _draw()
    result = _legacy_result(draw)
    assert _verify_search_identity(
        result, arm="adamw", config_id=7, updates=10, seed=0,
        search_draws=[draw], feature_dimensions={"medium": 3},
    ) == "legacy_frozen_config_match"


def test_legacy_result_identity_rejects_changed_search_hyperparameter():
    draw = _draw("spectral")
    result = _legacy_result(draw)
    result["config"]["learning_rate"] = 2e-3
    with pytest.raises(ValueError, match="frozen search draw"):
        _verify_search_identity(
            result, arm="spectral", config_id=7, updates=10, seed=0,
            search_draws=[draw], feature_dimensions={"medium": 3},
        )
