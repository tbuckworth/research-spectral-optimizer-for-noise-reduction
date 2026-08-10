import numpy as np
import pandas as pd
import pytest

from numerai_competitive.evaluate_frozen import _prediction_correlation, _rank_within_era


def test_rank_within_era_preserves_order_ties_and_boundaries():
    values = np.array([3.0, 1.0, 1.0, 9.0, 8.0])
    eras = np.array([1, 1, 1, 2, 2])
    ranked = _rank_within_era(values, eras)
    np.testing.assert_allclose(ranked, [5 / 6, 1 / 3, 1 / 3, 3 / 4, 1 / 4])
    assert np.all((ranked > 0) & (ranked < 1))


def test_rank_within_era_matches_pandas_reference():
    rng = np.random.default_rng(7)
    values = rng.normal(size=30)
    eras = np.repeat([3, 7, 9], 10)
    expected = pd.Series(values).groupby(eras).rank(method="average")
    expected = (expected - 0.5) / 10
    np.testing.assert_allclose(_rank_within_era(values, eras), expected)


def test_prediction_correlation_reports_per_era_not_only_pooled_value():
    index = pd.Index([f"row_{i}" for i in range(8)])
    eras = pd.Series(["1"] * 4 + ["2"] * 4, index=index)
    benchmark = pd.Series([0.1, 0.2, 0.3, 0.4] * 2, index=index)
    candidate = np.array([0.1, 0.2, 0.3, 0.4, 0.4, 0.3, 0.2, 0.1])
    result = _prediction_correlation(candidate, benchmark, eras)
    assert result["eras"] == 2
    assert result["per_era_mean"] == pytest.approx(0.0, abs=1e-12)
    assert result["per_era_min"] == pytest.approx(-1.0)
    assert result["per_era_max"] == pytest.approx(1.0)
