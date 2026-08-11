import json

import pandas as pd
import pytest

from numerai_competitive.high_rank_source import select_source


def _config(config_id, width):
    return {"arm": "spectral", "config_id": config_id, "rank": 128, "width": width,
            "depth": 2, "feature_set": "medium", "normalization": "none"}


def test_selects_best_development_config_feasible_at_requested_rank(tmp_path):
    search = tmp_path / "search.json"
    search.write_text(json.dumps({"configs": [_config(1, 2048), _config(2, 512),
                                                _config(3, 768)]}))
    rows = []
    for config_id, values in ((1, [0.05, 0.06]), (2, [0.03, 0.04]),
                              (3, [0.04, 0.04])):
        for split, value in zip(("fold1", "fold2"), values):
            rows.append({"arm": "spectral", "config_id": config_id,
                         "corr_mean": value, "split": split, "seed": 0,
                         "updates": 100_000})
    scores = tmp_path / "scores.csv"
    pd.DataFrame(rows).to_csv(scores, index=False)
    report = select_source(search, [scores], 1024, tmp_path / "selection.json")
    assert 1 not in report["feasible_config_ids"]
    assert report["selected"] == {"spectral": [3]}
    assert report["winner_score"]["corr_mean"] == pytest.approx(0.04)


def test_rejects_unequal_development_coverage(tmp_path):
    search = tmp_path / "search.json"
    search.write_text(json.dumps({"configs": [_config(1, 512), _config(2, 512)]}))
    scores = tmp_path / "scores.csv"
    pd.DataFrame([
        {"arm": "spectral", "config_id": 1, "corr_mean": 0.1,
         "split": "a", "seed": 0, "updates": 100},
        {"arm": "spectral", "config_id": 2, "corr_mean": 0.2,
         "split": "a", "seed": 0, "updates": 100},
        {"arm": "spectral", "config_id": 2, "corr_mean": 0.3,
         "split": "b", "seed": 0, "updates": 100},
    ]).to_csv(scores, index=False)
    with pytest.raises(ValueError, match="unequal"):
        select_source(search, [scores], 512, tmp_path / "selection.json")
