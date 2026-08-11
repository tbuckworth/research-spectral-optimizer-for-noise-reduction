import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import torch

from numerai_competitive.data import sha256
from numerai_competitive.live import export_callable
from numerai_competitive.model import MLPConfig, ResidualMLP
from numerai_competitive.predict_live import predict
from numerai_competitive.validate_live import validate


def test_target_free_live_resource_validator(tmp_path):
    config = MLPConfig(input_dim=3, width=5, depth=2)
    artifact = tmp_path / "model.pt"
    torch.save({
        "signature": "seed-0",
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
    csv_path = predict(callable_path, tmp_path / "live.parquet", tmp_path / "live.csv")
    csv = pd.read_csv(csv_path, index_col="id")
    assert csv.index.tolist() == index.tolist()
    assert list(csv.columns) == ["prediction"]
    assert not list(tmp_path.glob("*.tmp"))


def test_final_callable_export_verifies_freeze_signatures_and_hashes(tmp_path):
    config = MLPConfig(input_dim=2, width=4, depth=2)
    model = tmp_path / "model.pt"
    torch.save({
        "signature": "adamw-seed-0", "model_config": config.__dict__,
        "model": ResidualMLP(config).state_dict(), "feature_names": ["a", "b"],
        "data_version": "v5.3",
    }, model)
    freeze = tmp_path / "freeze.json"
    freeze.write_text(json.dumps({
        "status": "frozen",
        "candidate_transform": {"arm": "adamw", "model_weight": 1.0,
                                "benchmark": "v53_lgbm_ender20"},
        "selected": {"adamw": {"model_signatures": ["adamw-seed-0"],
                                "model_sha256": [sha256(model)]}},
    }))
    output = export_callable([model], tmp_path / "final.pkl", freeze_manifest=freeze)
    assert output.is_file()
    payload = json.loads(freeze.read_text())
    payload["selected"]["adamw"]["model_signatures"] = ["wrong"]
    freeze.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="signatures/order"):
        export_callable([model], tmp_path / "bad.pkl", freeze_manifest=freeze)


def test_export_is_self_contained_without_project_package_import(tmp_path):
    config = MLPConfig(input_dim=2, width=4, depth=2)
    model = tmp_path / "model.pt"
    torch.save({
        "signature": "seed-0", "model_config": config.__dict__,
        "model": ResidualMLP(config).state_dict(), "feature_names": ["a", "b"],
        "data_version": "v5.3",
    }, model)
    output = export_callable([model], tmp_path / "self-contained.pkl")
    script = """
import cloudpickle
import importlib.abc
import sys

class BlockProject(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "numerai_competitive" or fullname.startswith("numerai_competitive."):
            raise ModuleNotFoundError(fullname)
        return None

for name in list(sys.modules):
    if name == "numerai_competitive" or name.startswith("numerai_competitive."):
        del sys.modules[name]
sys.meta_path.insert(0, BlockProject())
with open(sys.argv[1], "rb") as handle:
    predictor = cloudpickle.load(handle)
assert len(predictor.models) == 1
"""
    completed = subprocess.run(
        [sys.executable, "-I", "-c", script, output], check=False,
        capture_output=True, text=True,
    )
    assert completed.returncode == 0, completed.stderr
