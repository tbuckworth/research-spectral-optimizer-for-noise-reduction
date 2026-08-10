import numpy as np
import pandas as pd
import pytest

from numerai_competitive.inference import moving_block_bootstrap


def test_constant_paired_effect_has_exact_interval_and_is_deterministic():
    index = pd.Index([f"{i:04d}" for i in range(40)])
    adamw = pd.Series(np.linspace(-0.02, 0.03, 40), index=index)
    spectral = adamw + 0.004
    first = moving_block_bootstrap(spectral, adamw, samples=200, seed=5)
    second = moving_block_bootstrap(spectral, adamw, samples=200, seed=5)
    assert first == second
    assert first.estimate == pytest.approx(0.004)
    assert first.ci_low == pytest.approx(0.004)
    assert first.ci_high == pytest.approx(0.004)
    assert first.probability_positive == 1


def test_rejects_misalignment_missing_values_and_short_series():
    left = pd.Series([0.1] * 8, index=range(8))
    with pytest.raises(ValueError, match="aligned"):
        moving_block_bootstrap(left, left.rename(index={0: 9}))
    with pytest.raises(ValueError, match="non-finite"):
        moving_block_bootstrap(left, left.mask(left.index == 3))
    with pytest.raises(ValueError, match="block_length"):
        moving_block_bootstrap(left.iloc[:4], left.iloc[:4])
