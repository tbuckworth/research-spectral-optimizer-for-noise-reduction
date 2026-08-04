#!/usr/bin/env python3
"""numerai_corr eval plumbing (exp-f02; reused by later runs).

`numerai_corr` matches the parent run's implementation
(parent experiments/exp-001/src/data_prep.py) — rank+gaussianize preds,
center target at 0.5, accentuate tails (**1.5), Pearson — with one addition:
a zero-variance guard returning 0.0 (a constant predictor has no rank
information; the parent's bare np.corrcoef would return NaN there). The
guard is what makes the zero-predictor null well-defined (sub-check e).
"""

import numpy as np
from scipy import stats


def numerai_corr(preds: np.ndarray, target: np.ndarray) -> float:
    """Official numerai_corr; returns 0.0 for zero-variance preds/target."""
    preds = np.asarray(preds, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if np.std(preds) < 1e-12 or np.std(target) < 1e-12:
        return 0.0
    ranked = stats.rankdata(preds, method="average") / (len(preds) + 1)
    gauss = stats.norm.ppf(ranked)
    p15 = np.sign(gauss) * np.abs(gauss) ** 1.5
    t = target - 0.5
    t15 = np.sign(t) * np.abs(t) ** 1.5
    return float(np.corrcoef(p15, t15)[0, 1])


def per_era_corr(preds: np.ndarray, target: np.ndarray, eras: np.ndarray):
    """Per-era numerai_corr. Returns (dict era -> corr, mean over eras)."""
    preds = np.asarray(preds)
    target = np.asarray(target)
    eras = np.asarray(eras)
    out = {}
    for era in np.unique(eras):
        m = eras == era
        out[int(era)] = numerai_corr(preds[m], target[m])
    mean = float(np.mean(list(out.values()))) if out else float("nan")
    return out, mean
