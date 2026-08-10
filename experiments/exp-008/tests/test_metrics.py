from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from numerai_tools.scoring import correlation_contribution, numerai_corr

from numerai_competitive.metrics import (
    per_era_corr,
    per_era_correlation_contribution,
    summarize_era_scores,
)


@pytest.fixture
def panel():
    ids = pd.Index([f"id{i:02d}" for i in range(16)], name="id")
    eras = pd.Series(["era1"] * 8 + ["era2"] * 8, index=ids, name="era")
    pred = pd.Series([0.1, 0.1, 0.4, 0.7, 0.7, 0.2, 0.9, 0.4] * 2,
                     index=ids, name="model")
    target = pd.Series([0.0, 0.25, 0.5, 0.75, 1.0, 0.25, 0.75, 0.5] * 2,
                       index=ids, name="target_cyrusd_20")
    benchmark = pd.Series([0.8, 0.2, 0.2, 0.6, 0.4, 0.9, 0.1, 0.6] * 2,
                          index=ids, name="v53_lgbm_ender20")
    return pred, target, benchmark, eras


def test_tied_per_era_corr_exactly_matches_tools(panel):
    pred, target, _, eras = panel
    actual = per_era_corr(pred, target, eras)["model"]
    expected = pd.Series({era: numerai_corr(pred[eras.eq(era)].to_frame(),
                                             target[eras.eq(era)],
                                             max_filtered_index_ratio=0.0)["model"]
                          for era in pd.unique(eras)})
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)


def test_tied_contribution_exactly_matches_tools(panel):
    pred, target, benchmark, eras = panel
    actual = per_era_correlation_contribution(pred, benchmark, target, eras)["model"]
    expected = pd.Series({era: correlation_contribution(pred[eras.eq(era)].to_frame(),
                                                          benchmark[eras.eq(era)],
                                                          target[eras.eq(era)])["model"]
                          for era in pd.unique(eras)})
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)


@pytest.mark.parametrize("kind", ["prediction", "target", "benchmark"])
def test_shuffled_ids_rejected(panel, kind):
    pred, target, benchmark, eras = panel
    shuffled = eras.index[::-1]
    if kind == "prediction": pred = pred.reindex(shuffled)
    if kind == "target": target = target.reindex(shuffled)
    if kind == "benchmark": benchmark = benchmark.reindex(shuffled)
    with pytest.raises(ValueError, match="exactly match"):
        per_era_correlation_contribution(pred, benchmark, target, eras)


@pytest.mark.parametrize("kind", ["prediction", "target", "benchmark", "era"])
def test_missing_or_duplicate_ids_rejected(panel, kind):
    pred, target, benchmark, eras = panel
    if kind == "era":
        eras = eras.copy(); eras.iloc[-1] = np.nan
    else:
        if kind == "prediction": pred = pred.iloc[:-1]
        if kind == "target": target = target.iloc[:-1]
        if kind == "benchmark": benchmark = benchmark.iloc[:-1]
    with pytest.raises(ValueError):
        per_era_correlation_contribution(pred, benchmark, target, eras)


def test_summary_population_std_and_drawdown():
    scores = pd.Series([0.1, -0.3, 0.2, -0.1], name="corr")
    summary = summarize_era_scores(scores).loc["corr"]
    assert summary["mean"] == pytest.approx(np.mean(scores))
    assert summary["std"] == pytest.approx(np.std(scores))
    assert summary["max_drawdown"] == pytest.approx(0.3)
    assert summary["cumulative"] == pytest.approx(-0.1)
