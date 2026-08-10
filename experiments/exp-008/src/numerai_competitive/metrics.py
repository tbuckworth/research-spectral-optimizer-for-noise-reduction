"""Strict, era-wise wrappers around the pinned ``numerai_tools`` scorers."""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
from numerai_tools.scoring import correlation_contribution, numerai_corr

PREDICTION_COLUMN = "prediction"


def _as_frame(predictions: pd.Series | pd.DataFrame) -> pd.DataFrame:
    if isinstance(predictions, pd.DataFrame):
        if predictions.columns.has_duplicates or predictions.shape[1] == 0:
            raise ValueError("predictions must have unique, non-empty columns")
        return predictions
    if not isinstance(predictions, pd.Series):
        raise TypeError("predictions must be a pandas Series or DataFrame")
    return predictions.rename(predictions.name or PREDICTION_COLUMN).to_frame()


def validate_exact_alignment(eras: pd.Series, objects: Mapping[str, pd.Series | pd.DataFrame]) -> None:
    if not isinstance(eras, pd.Series):
        raise TypeError("eras must be a pandas Series")
    if eras.index.has_duplicates:
        raise ValueError("eras index contains duplicate IDs")
    if eras.isna().any() or len(eras) == 0:
        raise ValueError("eras must be non-empty and contain no missing values")
    for name, obj in objects.items():
        if not isinstance(obj, (pd.Series, pd.DataFrame)):
            raise TypeError(f"{name} must be a pandas Series or DataFrame")
        if obj.index.has_duplicates:
            raise ValueError(f"{name} index contains duplicate IDs")
        if not obj.index.equals(eras.index):
            raise ValueError(f"{name} IDs/order do not exactly match eras")
        missing = obj.isna().any() if isinstance(obj, pd.Series) else obj.isna().any().any()
        if bool(missing):
            raise ValueError(f"{name} contains missing values")


def _by_era(predictions, targets, eras, benchmark=None) -> pd.DataFrame:
    frame = _as_frame(predictions)
    objects = {"predictions": frame, "targets": targets}
    if benchmark is not None:
        objects["benchmark"] = benchmark
    validate_exact_alignment(eras, objects)
    rows, labels = [], []
    for era in pd.unique(eras):
        ids = eras.index[eras.eq(era)]
        pred, target = frame.loc[ids].sort_index(), targets.loc[ids].sort_index()
        score = (
            numerai_corr(pred, target, max_filtered_index_ratio=0.0)
            if benchmark is None
            else correlation_contribution(pred, benchmark.loc[ids].sort_index(), target)
        )
        rows.append(score)
        labels.append(era)
    result = pd.DataFrame(rows, index=pd.Index(labels, name=eras.name or "era"))
    result.columns = frame.columns
    return result


def per_era_corr(predictions, targets: pd.Series, eras: pd.Series) -> pd.DataFrame:
    return _by_era(predictions, targets, eras)


def per_era_correlation_contribution(
    predictions, benchmark: pd.Series, targets: pd.Series, eras: pd.Series
) -> pd.DataFrame:
    return _by_era(predictions, targets, eras, benchmark=benchmark)


per_era_numerai_corr = per_era_corr


def summarize_era_scores(scores: pd.Series | pd.DataFrame) -> pd.DataFrame:
    frame = _as_frame(scores)
    if frame.empty or frame.isna().any().any():
        raise ValueError("scores must be non-empty and contain no missing values")
    values = frame.astype(float)
    mean, std = values.mean(), values.std(ddof=0)
    cumulative = values.cumsum()
    drawdown = cumulative.cummax() - cumulative
    return pd.DataFrame({"mean": mean, "std": std, "sharpe": mean / std,
                         "max_drawdown": drawdown.max(), "cumulative": cumulative.iloc[-1]})
