import numpy as np
import pandas as pd

from numerai_competitive.evaluate_frozen import _rank_within_era


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
