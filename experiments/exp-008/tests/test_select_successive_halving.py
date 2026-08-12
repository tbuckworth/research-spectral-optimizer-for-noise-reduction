from pathlib import Path

import pandas as pd
import pytest

from numerai_competitive.select_successive_halving import (
    build_plan,
    select_successive_halving,
)


def _scores(budget: int, offset: float = 0.0) -> pd.DataFrame:
    rows = []
    for split, penalty in (("a", 0.0), ("b", 0.01)):
        for arm, arm_offset in (("adamw", 0.0), ("spectral", 0.005)):
            for config_id in range(5):
                rows.append({
                    "arm": arm, "config_id": config_id, "updates": budget,
                    "split": split, "seed": 0,
                    "corr_mean": offset + arm_offset + config_id / 100 - penalty,
                })
    return pd.DataFrame(rows)


def test_selects_budget_specific_confirmations_and_small_paired_scout():
    value = select_successive_halving(
        [_scores(5000), _scores(20000, 0.02)], confirmation_top=2,
        long_scout_top=1, high_rank_source_config_id=1,
    )
    assert value["confirmation_selections"]["5000"] == {
        "adamw": [4, 3], "spectral": [4, 3], "paired_union": [3, 4],
    }
    assert value["confirmation_selections"]["20000"] == {
        "adamw": [1, 3, 4], "spectral": [1, 3, 4], "paired_union": [1, 3, 4],
    }
    assert value["long_scout_paired_union"] == [1, 4]
    assert value["high_rank_source_config_id"] == 1


def test_rejects_duplicate_or_unsupported_fidelities():
    with pytest.raises(ValueError, match="duplicate"):
        select_successive_halving(
            [_scores(5000), _scores(5000)], confirmation_top=2,
            long_scout_top=1, high_rank_source_config_id=1,
        )
    with pytest.raises(ValueError, match="5k or 20k"):
        select_successive_halving(
            [_scores(5000), _scores(100000)], confirmation_top=2,
            long_scout_top=1, high_rank_source_config_id=1,
        )


def test_build_plan_hashes_every_score_file(tmp_path: Path):
    paths = []
    for budget in (5000, 20000):
        group = []
        for index in range(2):
            path = tmp_path / f"{budget}-{index}.csv"
            _scores(budget).query("split == @split_name", local_dict={
                "split_name": "a" if index == 0 else "b",
            }).to_csv(path, index=False)
            group.append(path)
        paths.append(group)
    output = tmp_path / "plan.json"
    report = build_plan(
        paths, confirmation_top=2, long_scout_top=1,
        high_rank_source_config_id=2, output=output,
    )
    assert report["status"] == "successive_halving_plan_frozen"
    assert set(report["score_sha256"]) == {str(path) for group in paths for path in group}
    assert output.is_file()
