import json

import pytest

from numerai_competitive.confirm_walkforward import confirm


def _write(path, value):
    path.write_text(json.dumps(value))
    return path


def test_confirms_same_preselected_configs_without_reselection(tmp_path):
    selection = _write(tmp_path / "selection.json", {
        "status": "development_budget_sensitivity_selection",
        "selected": {"adamw": [1], "spectral": [2]},
        "selected_updates": {"adamw": [5000], "spectral": [100000]},
    })
    audits = [
        _write(tmp_path / f"outer-{number}.json", {
            "status": "audit_complete", "split": {"name": f"outer_{number}"},
            "selected": {"adamw": 1, "spectral": 2},
            "updates": {"adamw": 5000, "spectral": 100000},
        }) for number in range(1, 4)
    ]
    report = confirm(selection, audits, tmp_path / "confirmed.json")
    assert report["status"] == "fixed_config_walkforward_confirmed"
    assert report["reselected_on_outer_outcomes"] is False
    assert report["selected"] == {"adamw": [1], "spectral": [2]}


def test_rejects_outer_result_from_different_config(tmp_path):
    selection = _write(tmp_path / "selection.json", {
        "selected": {"adamw": [1], "spectral": [2]},
        "selected_updates": {"adamw": [5000], "spectral": [100000]},
    })
    audits = [
        _write(tmp_path / f"outer-{number}.json", {
            "status": "audit_complete", "split": {"name": f"outer_{number}"},
            "selected": {"adamw": 9 if number == 2 else 1, "spectral": 2},
            "updates": {"adamw": 5000, "spectral": 100000},
        }) for number in range(1, 4)
    ]
    with pytest.raises(ValueError, match="outer_2"):
        confirm(selection, audits, tmp_path / "confirmed.json")
