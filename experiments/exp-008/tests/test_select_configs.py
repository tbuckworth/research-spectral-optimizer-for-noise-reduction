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
            rows.extend({"arm": arm, "config_id": config_id, "corr_mean": score,
                         "split": f"fold_{fold}", "seed": 0, "updates": 20_000}
                        for fold, score in enumerate(fold_scores))
    assert select_configs(pd.DataFrame(rows), 2) == {
        "adamw": [1, 0], "spectral": [2, 1], "paired_union": [0, 1, 2],
    }


def test_rejects_unequal_fold_coverage():
    frame = pd.DataFrame([
        {"arm": "adamw", "config_id": 0, "corr_mean": 0.1, "split": "a", "seed": 0,
         "updates": 20_000},
        {"arm": "adamw", "config_id": 1, "corr_mean": 0.1, "split": "a", "seed": 0,
         "updates": 20_000},
        {"arm": "adamw", "config_id": 1, "corr_mean": 0.2, "split": "b", "seed": 0,
         "updates": 20_000},
        {"arm": "spectral", "config_id": 0, "corr_mean": 0.1, "split": "a", "seed": 0,
         "updates": 20_000},
        {"arm": "spectral", "config_id": 1, "corr_mean": 0.1, "split": "a", "seed": 0,
         "updates": 20_000},
    ])
    with pytest.raises(ValueError, match="unequal fold/seed coverage"):
        select_configs(frame, 1)


def test_rejects_mixed_fidelities_and_unpaired_config_ids():
    base = pd.DataFrame([
        {"arm": "adamw", "config_id": 0, "corr_mean": 0.1, "split": "a", "seed": 0,
         "updates": 5_000},
        {"arm": "spectral", "config_id": 0, "corr_mean": 0.1, "split": "a", "seed": 0,
         "updates": 20_000},
    ])
    with pytest.raises(ValueError, match="mix fidelity"):
        select_configs(base, 1)
    unpaired = pd.concat([base.iloc[[0]], base.iloc[[0]].assign(arm="spectral", config_id=1)])
    with pytest.raises(ValueError, match="config-ID coverage differs"):
        select_configs(unpaired, 1)


def test_allows_predeclared_asymmetric_spectral_rank_candidates():
    frame = pd.DataFrame([
        {"arm": "adamw", "config_id": 0, "corr_mean": 0.1, "split": "a", "seed": 0,
         "updates": 20_000},
        {"arm": "spectral", "config_id": 0, "corr_mean": 0.1, "split": "a", "seed": 0,
         "updates": 20_000},
        {"arm": "spectral", "config_id": 1000512, "corr_mean": 0.2, "split": "a",
         "seed": 0, "updates": 20_000},
    ])
    assert select_configs(frame, 1, allow_asymmetric=True) == {
        "adamw": [0], "spectral": [1000512], "paired_union": [0, 1000512],
    }
