import json
import shutil

import pandas as pd
import pytest

from numerai_competitive.download_live import DATASETS, download_live


class FakeAPI:
    def __init__(self, files, rounds=(1400, 1400)):
        self.files = files
        self.rounds = iter(rounds)
        self.api_touched = False

    def get_current_round(self):
        self.api_touched = True
        return next(self.rounds)

    def list_datasets(self):
        return list(self.files)

    def download_dataset(self, name, destination):
        shutil.copyfile(self.files[name], destination)


def _freeze(path, valid=True):
    path.write_text(json.dumps({
        "status": "frozen" if valid else "draft", "protocol": "test",
        "code_commit": "abc", "search_sha256": "def",
        "selected": {
            "adamw": {"config_id": 1, "updates": 100, "seeds": [0]},
            "spectral": {"config_id": 2, "updates": 100, "seeds": [0]},
        },
        "validation_reveal_authorized": valid,
    }))


def _files(tmp_path):
    index = pd.Index(["id1", "id2"], name="id")
    live = tmp_path / "source-live.parquet"
    benchmark = tmp_path / "source-benchmark.parquet"
    pd.DataFrame({"feature_a": [0, 4]}, index=index).to_parquet(live)
    pd.DataFrame({"v53_lgbm_ender60": [0.2, 0.8]}, index=index).to_parquet(benchmark)
    return {
        "v5.3/live.parquet": live,
        "v5.3/live_benchmark_models.parquet": benchmark,
    }


def test_live_download_requires_freeze_before_api_access(tmp_path):
    freeze = tmp_path / "freeze.json"
    _freeze(freeze, False)
    api = FakeAPI({})
    with pytest.raises(ValueError, match="sealed"):
        download_live(tmp_path / "live", freeze, api)
    assert not api.api_touched


def test_live_download_records_one_consistent_target_free_round(tmp_path):
    freeze = tmp_path / "freeze.json"
    _freeze(freeze)
    report = download_live(tmp_path / "live", freeze, FakeAPI(_files(tmp_path)))
    assert report["status"] == "complete" and report["round"] == 1400
    assert report["rows"] == 2 and set(report["artifacts"]) == set(DATASETS.values())


def test_live_download_accepts_null_target_schema_columns(tmp_path):
    freeze = tmp_path / "freeze.json"
    _freeze(freeze)
    files = _files(tmp_path)
    live_path = files["v5.3/live.parquet"]
    live = pd.read_parquet(live_path)
    live["target"] = float("nan")
    live.to_parquet(live_path)
    assert download_live(tmp_path / "live", freeze, FakeAPI(files))["status"] == "complete"


def test_live_download_refuses_revealed_target_values(tmp_path):
    freeze = tmp_path / "freeze.json"
    _freeze(freeze)
    files = _files(tmp_path)
    live_path = files["v5.3/live.parquet"]
    live = pd.read_parquet(live_path)
    live["target"] = [float("nan"), 0.5]
    live.to_parquet(live_path)
    with pytest.raises(ValueError, match="revealed target values"):
        download_live(tmp_path / "live", freeze, FakeAPI(files))


def test_live_download_accepts_unstakeable_bounded_pair_freeze(tmp_path):
    freeze = tmp_path / "freeze.json"
    freeze.write_text(json.dumps({
        "status": "bounded_live_frozen", "code_commit": "a" * 40,
        "selected": {"adamw": {}, "spectral": {}},
        "upload_authorized": False, "staking_authorized": False,
    }))
    report = download_live(tmp_path / "live", freeze, FakeAPI(_files(tmp_path)))
    assert report["status"] == "complete" and report["freeze_code_commit"] == "a" * 40


def test_live_download_refuses_round_transition_and_removes_temporary_files(tmp_path):
    freeze = tmp_path / "freeze.json"
    _freeze(freeze)
    destination = tmp_path / "live"
    with pytest.raises(RuntimeError, match="changed"):
        download_live(destination, freeze, FakeAPI(_files(tmp_path), rounds=(1400, 1401)))
    assert not list(destination.glob("*.download"))
    assert not (destination / "download-complete.json").exists()
