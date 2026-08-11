import pandas as pd
import pytest

from numerai_competitive.select_budgeted_configs import restrict_candidates, select_budgeted_configs


def _scores() -> pd.DataFrame:
    rows = []
    values = {
        ("adamw", 1, 5_000): (0.04, 0.03),
        ("adamw", 1, 100_000): (0.01, 0.00),
        ("spectral", 1, 5_000): (0.03, 0.02),
        ("spectral", 1, 100_000): (0.05, 0.04),
    }
    for (arm, config_id, updates), pair in values.items():
        for seed, value in enumerate(pair):
            rows.append({
                "arm": arm, "config_id": config_id, "updates": updates,
                "split": "outer_1_inner_1", "seed": seed, "corr_mean": value,
            })
    return pd.DataFrame(rows)


def test_selects_budget_independently_by_arm() -> None:
    assert select_budgeted_configs(_scores(), 1) == {
        "adamw": [{"config_id": 1, "updates": 5_000}],
        "spectral": [{"config_id": 1, "updates": 100_000}],
    }


def test_rejects_unequal_seed_coverage() -> None:
    frame = _scores().drop(index=0)
    with pytest.raises(ValueError, match="unequal split/seed coverage"):
        select_budgeted_configs(frame, 1)


def test_rejects_unpaired_candidate_budget_coverage() -> None:
    frame = _scores()
    frame = frame[~(
        frame["arm"].eq("spectral") & frame["updates"].eq(100_000)
    )]
    with pytest.raises(ValueError, match="config/budget coverage differs"):
        select_budgeted_configs(frame, 1)


def test_can_select_spectral_specific_candidates_when_explicitly_allowed() -> None:
    frame = _scores()
    extra = frame[
        frame["arm"].eq("spectral") & frame["updates"].eq(100_000)
    ].copy()
    extra["config_id"] = 99
    extra["corr_mean"] = 0.08
    combined = pd.concat([frame, extra], ignore_index=True)
    assert select_budgeted_configs(combined, 1, allow_asymmetric=True) == {
        "adamw": [{"config_id": 1, "updates": 5_000}],
        "spectral": [{"config_id": 99, "updates": 100_000}],
    }


def test_rejects_duplicate_cell() -> None:
    frame = pd.concat([_scores(), _scores().iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        select_budgeted_configs(frame, 1)


def test_restricts_larger_stage_to_preregistered_union() -> None:
    frame = _scores()
    extra = frame.copy()
    extra["config_id"] = 2
    combined = pd.concat([frame, extra], ignore_index=True)
    selected = restrict_candidates(combined, [1])
    assert set(selected["config_id"]) == {1}
    assert len(selected) == len(frame)


@pytest.mark.parametrize("config_ids", [[], [1, 1], [-1]])
def test_rejects_invalid_candidate_filter(config_ids: list[int]) -> None:
    with pytest.raises(ValueError, match="config IDs"):
        restrict_candidates(_scores(), config_ids)


def test_rejects_missing_requested_candidate() -> None:
    with pytest.raises(ValueError, match="missing from scores"):
        restrict_candidates(_scores(), [1, 2])
