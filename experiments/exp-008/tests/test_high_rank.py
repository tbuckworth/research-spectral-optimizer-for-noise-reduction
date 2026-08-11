import json

import pytest

from numerai_competitive.data import sha256
from numerai_competitive.high_rank import create_extension


def test_high_rank_extension_clones_only_selected_spectral_draw(tmp_path):
    search = tmp_path / "search.json"
    search.write_text(json.dumps({
        "primary_target": "target_cyrusd_20", "primary_metric": "standalone_exact_corr",
        "configs": [
            {"arm": "spectral", "config_id": 4, "rank": 256, "width": 512},
            {"arm": "spectral", "config_id": 5, "rank": 16, "width": 2048},
        ],
    }))
    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps({"selected": {"spectral": [4]}}))
    output = tmp_path / "extension.json"
    report = create_extension(search, selection, [1024, 512], output)
    assert report["ranks"] == [512, 1024]
    assert [config["width"] for config in report["configs"]] == [512, 512]
    assert [config["config_id"] for config in report["configs"]] == [10512, 11024]
    assert report["source_search_sha256"] == sha256(search)
    assert json.loads(output.read_text()) == report


def test_high_rank_extension_rejects_non_extension_rank(tmp_path):
    search = tmp_path / "search.json"
    search.write_text(json.dumps({"configs": [
        {"arm": "spectral", "config_id": 1, "rank": 256},
    ]}))
    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps({"selected": {"spectral": [1]}}))
    with pytest.raises(ValueError, match="above the selected rank"):
        create_extension(search, selection, [256], tmp_path / "out.json")
