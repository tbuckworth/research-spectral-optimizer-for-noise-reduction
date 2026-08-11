import json

import pytest

from numerai_competitive.data import sha256
from numerai_competitive.high_rank import (
    create_extension,
    memory_estimate,
    parameter_count,
    required_probe_updates,
)


def test_high_rank_extension_clones_only_selected_spectral_draw(tmp_path):
    search = tmp_path / "search.json"
    search.write_text(json.dumps({
        "primary_target": "target_cyrusd_20", "primary_metric": "standalone_exact_corr",
        "configs": [
            {"arm": "spectral", "config_id": 4, "rank": 256, "width": 512,
             "depth": 2, "feature_set": "medium", "normalization": "none",
             "filter_update_every": 2, "filter_warmup": 100},
            {"arm": "spectral", "config_id": 5, "rank": 16, "width": 2048,
             "depth": 4, "feature_set": "all", "normalization": "layer",
             "filter_update_every": 1, "filter_warmup": 100},
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
    assert all(config["memory_screen"]["probe_required"] for config in report["configs"])
    assert [config["required_probe_updates"] for config in report["configs"]] == [1025, 2049]
    assert json.loads(output.read_text()) == report


def test_high_rank_extension_rejects_non_extension_rank(tmp_path):
    search = tmp_path / "search.json"
    search.write_text(json.dumps({"configs": [
        {"arm": "spectral", "config_id": 1, "rank": 256, "width": 512,
         "depth": 2, "feature_set": "medium", "normalization": "none",
         "filter_update_every": 1, "filter_warmup": 100},
    ]}))
    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps({"selected": {"spectral": [1]}}))
    with pytest.raises(ValueError, match="above the selected rank"):
        create_extension(search, selection, [256], tmp_path / "out.json")


def test_memory_screen_uses_exact_model_parameter_count():
    config = {"width": 2, "depth": 2, "feature_set": "medium",
              "normalization": "layer"}
    # input: 780*2+2; hidden: LN(4)+2*2+2; output LN(4)+2+1
    assert parameter_count(config) == 1579
    small = memory_estimate(config, 512)
    assert small["basis_bytes"] == 1579 * 512 * 4
    huge = dict(config, width=2048, depth=4, feature_set="all")
    assert not memory_estimate(huge, 1024)["analytically_feasible_48gb"]


def test_probe_budget_accounts_for_covariance_cadence_and_warmup():
    config = {"filter_update_every": 10, "filter_warmup": 5000}
    assert required_probe_updates(config, 256) == 5001
    config["filter_warmup"] = 100
    assert required_probe_updates(config, 256) == 2561
