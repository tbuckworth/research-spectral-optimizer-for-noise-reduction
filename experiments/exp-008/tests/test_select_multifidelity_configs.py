from pathlib import Path

import pandas as pd
import pytest

from numerai_competitive.select_multifidelity_configs import (
    build_selection,
    select_multifidelity,
)


def _scores(updates: int, adamw: list[float], spectral: list[float]) -> pd.DataFrame:
    return pd.DataFrame([
        {"arm": arm, "config_id": config_id, "corr_mean": score,
         "split": "fold_1", "seed": 0, "updates": updates}
        for arm, values in (("adamw", adamw), ("spectral", spectral))
        for config_id, score in enumerate(values)
    ])


def test_preserves_winners_from_each_fidelity():
    early = _scores(5_000, [0.9, 0.1, 0.0], [0.0, 0.8, 0.1])
    late = _scores(20_000, [0.0, 0.1, 0.9], [0.8, 0.0, 0.1])
    value = select_multifidelity([early, late], 1)
    assert value["fidelity_selections"] == {
        "5000": {"adamw": [0], "spectral": [1], "paired_union": [0, 1]},
        "20000": {"adamw": [2], "spectral": [0], "paired_union": [0, 2]},
    }
    assert value["selected"] == {
        "adamw": [0, 2], "spectral": [0, 1], "paired_union": [0, 1, 2],
    }


def test_rejects_mixed_or_duplicate_fidelities():
    early = _scores(5_000, [0.1], [0.1])
    mixed = pd.concat([early, _scores(20_000, [0.2], [0.2])], ignore_index=True)
    with pytest.raises(ValueError, match="exactly one fidelity"):
        select_multifidelity([mixed], 1)
    with pytest.raises(ValueError, match="duplicate fidelity"):
        select_multifidelity([early, early], 1)


def test_build_selection_hashes_every_group(tmp_path: Path):
    paths = []
    for updates in (5_000, 20_000):
        path = tmp_path / f"scores-{updates}.csv"
        _scores(updates, [0.2, 0.1], [0.1, 0.2]).to_csv(path, index=False)
        paths.append(path)
    output = tmp_path / "selection.json"
    report = build_selection([[paths[0]], [paths[1]]], 1, output)
    assert report["status"] == "multifidelity_promotion"
    assert set(report["score_sha256"]) == {str(path) for path in paths}
    assert output.is_file()
