import pandas as pd
import pytest

from numerai_competitive.select_successive_finalists import select_finalists


def _ordinary() -> pd.DataFrame:
    return pd.DataFrame([
        {"arm": arm, "config_id": config_id, "corr_mean": score, "split": "inner_1",
         "seed": 0, "updates": 100000}
        for arm, values in {
            "adamw": [(1, 0.03), (2, 0.02)],
            "spectral": [(1, 0.01), (2, 0.04)],
        }.items() for config_id, score in values
    ])


def _high() -> pd.DataFrame:
    return pd.DataFrame([
        {"arm": "spectral", "config_id": config_id, "corr_mean": score - penalty,
         "split": split, "seed": 0, "updates": 20000}
        for config_id, score in ((101, 0.03), (102, 0.05), (103, 0.04))
        for split, penalty in (("inner_1", 0.0), ("inner_2", 0.01))
    ])


def test_selects_arm_winners_source_control_and_top_high_ranks():
    value = select_finalists(
        _ordinary(), _high(), high_rank_ids=[101, 102, 103],
        high_rank_source_config_id=7, high_rank_top=2,
    )
    assert value["ordinary_winners"] == {"adamw": [1], "spectral": [2]}
    assert value["ordinary_confirmation_paired_union"] == [1, 2, 7]
    assert value["high_rank_spectral"] == [102, 103]


def test_rejects_unequal_high_rank_fold_coverage():
    frame = _high().query("not (config_id == 103 and split == 'inner_2')")
    with pytest.raises(ValueError, match="equal fold coverage"):
        select_finalists(
            _ordinary(), frame, high_rank_ids=[101, 102, 103],
            high_rank_source_config_id=7, high_rank_top=2,
        )


def test_rejects_multifold_ordinary_scout():
    frame = pd.concat([_ordinary(), _ordinary().assign(split="inner_2")])
    with pytest.raises(ValueError, match="one fold"):
        select_finalists(
            frame, _high(), high_rank_ids=[101, 102, 103],
            high_rank_source_config_id=7, high_rank_top=2,
        )
