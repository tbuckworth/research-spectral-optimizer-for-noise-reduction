import hashlib
import json
from pathlib import Path

import pytest

import numerai_competitive.download_validation as module


class FakeAPI:
    def __init__(self, payloads):
        self.payloads = payloads
        self.list_called = False
        self.downloaded = []

    def list_datasets(self):
        self.list_called = True
        return list(self.payloads)

    def download_dataset(self, name, destination):
        self.downloaded.append(name)
        Path(destination).write_bytes(self.payloads[name])


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


def test_download_requires_freeze_before_api_access(tmp_path):
    freeze = tmp_path / "freeze.json"
    _freeze(freeze, valid=False)
    api = FakeAPI({})
    with pytest.raises(ValueError, match="sealed"):
        module.download_validation(tmp_path / "data", freeze, api)
    assert not api.list_called


def test_download_verifies_pinned_artifacts_and_records_freeze(tmp_path, monkeypatch):
    payloads = {name: name.encode() for name in module.DATASETS}
    hashes = {name: hashlib.sha256(payload).hexdigest() for name, payload in payloads.items()}
    monkeypatch.setattr(module, "DATASETS", {
        name: (module.DATASETS[name][0], hashes[name]) for name in module.DATASETS
    })
    freeze = tmp_path / "freeze.json"
    _freeze(freeze)
    api = FakeAPI(payloads)
    report = module.download_validation(tmp_path / "data", freeze, api)
    assert report["status"] == "complete" and len(report["artifacts"]) == 3
    assert set(api.downloaded) == set(payloads)
    assert (tmp_path / "data" / "download-complete.json").is_file()


def test_download_refuses_existing_wrong_artifact(tmp_path, monkeypatch):
    payloads = {name: name.encode() for name in module.DATASETS}
    hashes = {name: hashlib.sha256(payload).hexdigest() for name, payload in payloads.items()}
    monkeypatch.setattr(module, "DATASETS", {
        name: (module.DATASETS[name][0], hashes[name]) for name in module.DATASETS
    })
    freeze = tmp_path / "freeze.json"
    _freeze(freeze)
    destination = tmp_path / "data"
    destination.mkdir()
    (destination / "validation.parquet").write_bytes(b"wrong")
    with pytest.raises(ValueError, match="differs"):
        module.download_validation(destination, freeze, FakeAPI(payloads))
