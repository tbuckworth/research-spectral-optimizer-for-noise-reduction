import pandas as pd
import pytest

from numerai_competitive.select_configs import select_configs


def test_selects_each_arm_by_mean_then_worst_fold():
    rows = []
    values = {
        "adamw": {0: [0.2, 0.0], 1: [0.1, 0.1], 2: [0.05, 0.04]},
        "spectral": {0: [0.0, 0.0], 1: [0.3, 0.1], 2: [0.2, 0.2]},
    }
    for arm, configs in values.items():
        for config_id, fold_scores in configs.items():
            rows.extend({"arm": arm, "config_id": config_id, "corr_mean": score}
                        for score in fold_scores)
    assert select_configs(pd.DataFrame(rows), 2) == {
        "adamw": [1, 0], "spectral": [2, 1],
    }


def test_rejects_unequal_fold_coverage():
    frame = pd.DataFrame([
        {"arm": "adamw", "config_id": 0, "corr_mean": 0.1},
        {"arm": "adamw", "config_id": 1, "corr_mean": 0.1},
        {"arm": "adamw", "config_id": 1, "corr_mean": 0.2},
        {"arm": "spectral", "config_id": 0, "corr_mean": 0.1},
    ])
    with pytest.raises(ValueError, match="unequal fold coverage"):
        select_configs(frame, 1)
