import json

import numpy as np
import pandas as pd
import torch

from numerai_competitive.live import export_callable
from numerai_competitive.model import MLPConfig, ResidualMLP
from numerai_competitive.validate_live import validate


def test_target_free_live_resource_validator(tmp_path):
    config = MLPConfig(input_dim=3, width=5, depth=2)
    artifact = tmp_path / "model.pt"
    torch.save({
        "model_config": config.__dict__, "model": ResidualMLP(config).state_dict(),
        "feature_names": ["a", "b", "c"], "data_version": "v5.3",
    }, artifact)
    callable_path = export_callable([artifact], tmp_path / "predictor.pkl", batch_size=2)
    index = pd.Index(["id1", "id2", "id3"], name="id")
    pd.DataFrame(
        np.array([[0, 2, 4], [4, 1, 2], [1, 3, 0]], dtype=np.uint8),
        index=index, columns=["a", "b", "c"],
    ).to_parquet(tmp_path / "live.parquet")
    report = validate(callable_path, tmp_path / "live.parquet", tmp_path / "report.json")
    assert report["status"] == "pass" and report["rows"] == 3
    assert report["model_count"] == 1 and report["artifact_bytes"] > 0
    assert json.loads((tmp_path / "report.json").read_text())["status"] == "pass"
