import pandas as pd
import pytest

from numerai_competitive.assemble_successive_scores import assemble, expected_candidates

PLAN = {
    "confirmation_selections": {
        "5000": {"paired_union": [1, 2]},
        "20000": {"paired_union": [2, 3]},
    },
}
FINALISTS = {
    "ordinary_confirmation_paired_union": [3, 7],
    "high_rank_spectral": [101, 102],
}


def _scores() -> pd.DataFrame:
    rows = []
    for budget, arms in expected_candidates(PLAN, FINALISTS).items():
        for arm, ids in arms.items():
            for config_id in ids:
                for split in ("a", "b"):
                    for seed in range(3):
                        rows.append({
                            "arm": arm, "config_id": config_id, "updates": budget,
                            "split": split, "seed": seed, "corr_mean": config_id / 1000,
                        })
    rows.append({
        "arm": "spectral", "config_id": 999, "updates": 20000,
        "split": "a", "seed": 0, "corr_mean": 1.0,
    })
    return pd.DataFrame(rows)


def test_assembles_only_frozen_candidates_with_equal_coverage():
    result = assemble(
        _scores(), PLAN, FINALISTS, expected_splits={"a", "b"},
        expected_seeds={0, 1, 2},
    )
    assert 999 not in set(result["config_id"])
    assert len(result) == 96
    assert set(result.query("updates == 100000 and arm == 'spectral'")["config_id"]) == {
        3, 7, 101, 102,
    }


def test_rejects_missing_or_duplicate_exact_cells():
    scores = _scores()
    missing = scores.drop(scores.query(
        "arm == 'adamw' and config_id == 1 and updates == 5000 and split == 'a' and seed == 0"
    ).index)
    with pytest.raises(ValueError, match="coverage differs"):
        assemble(missing, PLAN, FINALISTS, expected_splits={"a", "b"},
                 expected_seeds={0, 1, 2})
    duplicated = pd.concat([scores, scores.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        assemble(duplicated, PLAN, FINALISTS, expected_splits={"a", "b"},
                 expected_seeds={0, 1, 2})
