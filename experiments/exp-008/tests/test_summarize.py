import json

import numpy as np
import pytest

from numerai_competitive.summarize import collect_stage, write_summary


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
