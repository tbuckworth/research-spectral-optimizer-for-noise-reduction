"""Predeclared paired time-series inference for optimizer comparisons."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PairedBootstrap:
    estimate: float
    ci_low: float
    ci_high: float
    probability_positive: float
    eras: int
    block_length: int
    samples: int
    seed: int

    def to_dict(self) -> dict:
        return asdict(self)


def _aligned_difference(spectral: pd.Series, adamw: pd.Series) -> np.ndarray:
    if spectral.index.has_duplicates or adamw.index.has_duplicates:
        raise ValueError("paired era scores must have unique indices")
    if not spectral.index.equals(adamw.index):
        raise ValueError("spectral and AdamW era indices are not exactly aligned")
    difference = spectral.to_numpy(float) - adamw.to_numpy(float)
    if not np.isfinite(difference).all():
        raise ValueError("paired era differences contain non-finite values")
    return difference


def moving_block_bootstrap(
    spectral: pd.Series,
    adamw: pd.Series,
    *,
    block_length: int = 8,
    samples: int = 10_000,
    seed: int = 20260810,
) -> PairedBootstrap:
    """Circular moving-block bootstrap of mean spectral-minus-AdamW CORR."""
    difference = _aligned_difference(spectral, adamw)
    n = len(difference)
    if n < block_length or block_length < 1 or samples < 1:
        raise ValueError("need n >= block_length >= 1 and positive samples")
    rng = np.random.default_rng(seed)
    blocks = int(np.ceil(n / block_length))
    offsets = np.arange(block_length)
    boot_means = np.empty(samples, dtype=np.float64)
    for sample in range(samples):
        starts = rng.integers(0, n, size=blocks)
        indices = ((starts[:, None] + offsets) % n).reshape(-1)[:n]
        boot_means[sample] = difference[indices].mean()
    return PairedBootstrap(
        estimate=float(difference.mean()),
        ci_low=float(np.quantile(boot_means, 0.025)),
        ci_high=float(np.quantile(boot_means, 0.975)),
        probability_positive=float(np.mean(boot_means > 0)),
        eras=n,
        block_length=block_length,
        samples=samples,
        seed=seed,
    )
