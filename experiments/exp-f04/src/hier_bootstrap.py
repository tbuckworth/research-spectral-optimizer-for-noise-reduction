"""Hierarchical era-block x seed moving-block bootstrap (A6) - DELIVERABLE.

This module is the inference machinery the verdict analysis will import.

Estimand: the seed-averaged fold-mean paired per-era difference
    theta = mean_over_eras( mean_over_seeds( D[era, seed] ) )
where D is an (n_eras, n_seeds) matrix of per-era paired differences
(e.g. B-A numerai_corr, same seed and data order in both arms).

Resampling scheme (A6, hierarchical):
  - CIRCULAR MOVING-BLOCK bootstrap over eras: block length L computed from
    the lag-1 autocorrelation of the seed-averaged difference series of the
    window itself (recomputed per fold window - never inherited).
  - Nested SEED-level resampling: for each bootstrap replicate, the seed
    axis is independently resampled with replacement (n_seeds draws), so
    between-seed variance is folded into the CI rather than averaged away.
  - Bootstrap RNG is passed in explicitly (fixed and stated); stability of
    conclusions must be checked across >= 2 RNG seeds before any claim.

Block-length rule (pre-registered): AR(1) plug-in with the n^(1/3)
Hall-Horowitz-Jing rate,
    L = round( (sqrt(6)*|rho|/(1-rho^2))^(2/3) * n^(1/3) ),
clipped to [2, n//3]; if rho <= 0, L = 2. Sanity anchor: at the parent's
measured rho = 0.247, n = 255 this gives L = 5 (parent used L = 4).

D8 pooled secondary estimand: the same function applied to the
concatenation of the three folds' era-level paired differences (330 eras)
under ONE hierarchical MBB, block length from the pooled series' own ACF.
Pre-registered as SECONDARY: it cannot overturn the per-fold decision rule.
"""

from __future__ import annotations

import numpy as np


def lag1_acf(x: np.ndarray) -> float:
    """Lag-1 autocorrelation of a 1-D series (sample correlation of x_t, x_{t+1})."""
    x = np.asarray(x, dtype=float)
    if len(x) < 3 or np.std(x[:-1]) == 0 or np.std(x[1:]) == 0:
        return 0.0
    return float(np.corrcoef(x[:-1], x[1:])[0, 1])


def block_length(rho: float, n: int) -> int:
    """Pre-registered AR(1) plug-in block length at the n^(1/3) rate."""
    if rho <= 0.0:
        return 2
    rho = min(rho, 0.99)
    raw = (np.sqrt(6.0) * rho / (1.0 - rho * rho)) ** (2.0 / 3.0) * n ** (1.0 / 3.0)
    return int(np.clip(round(raw), 2, max(2, n // 3)))


def hier_mbb(
    D: np.ndarray,
    n_boot: int = 2000,
    rng: np.random.Generator | None = None,
    L: int | None = None,
    alpha: float = 0.05,
) -> dict:
    """Hierarchical era-block x seed bootstrap CI for the seed-averaged mean.

    Parameters
    ----------
    D : (n_eras, n_seeds) per-era per-seed paired differences.
    n_boot : bootstrap replicates.
    rng : numpy Generator (bootstrap RNG - fixed and stated by the caller).
    L : block length override; default = block_length(lag1_acf(seed-avg), n).
    alpha : 1 - CI level.

    Returns dict with point estimate, CI, half-width, realized rho and L.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    D = np.asarray(D, dtype=float)
    n, S = D.shape
    seed_avg = D.mean(axis=1)
    rho = lag1_acf(seed_avg)
    if L is None:
        L = block_length(rho, n)
    nblocks = int(np.ceil(n / L))
    starts = rng.integers(0, n, size=(n_boot, nblocks))
    idx = ((starts[..., None] + np.arange(L)) % n).reshape(n_boot, -1)[:, :n]
    seed_idx = rng.integers(0, S, size=(n_boot, S))
    A = D[idx]  # (n_boot, n, S)
    A = np.take_along_axis(A, seed_idx[:, None, :], axis=2)
    stats = A.mean(axis=(1, 2))
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "mean": float(D.mean()),
        "lo": float(lo),
        "hi": float(hi),
        "half_width": float((hi - lo) / 2.0),
        "rho_lag1": float(rho),
        "block_length": int(L),
        "n_eras": int(n),
        "n_seeds": int(S),
        "n_boot": int(n_boot),
        "excludes_zero": bool(lo > 0.0 or hi < 0.0),
    }


def seed_vs_era_variance_ratio(D: np.ndarray) -> dict:
    """A6 diagnostic: between-seed variance of the fold mean vs era-blocked
    variance of the seed-averaged series (both as variance of the fold-mean
    estimate). If seed share < ~10%, the era-block CI may serve as headline
    with the hierarchical CI alongside."""
    D = np.asarray(D, dtype=float)
    n, S = D.shape
    seed_means = D.mean(axis=0)
    var_seed = float(seed_means.var(ddof=1) / S)  # contribution to fold-mean var
    seed_avg = D.mean(axis=1)
    rho = lag1_acf(seed_avg)
    L = block_length(rho, n)
    nb = n // L
    blocks = seed_avg[: nb * L].reshape(nb, L).mean(axis=1)
    var_era = float(blocks.var(ddof=1) / nb)  # block-mean based fold-mean var
    return {
        "var_fold_mean_from_seeds": var_seed,
        "var_fold_mean_from_era_blocks": var_era,
        "seed_share": float(var_seed / (var_seed + var_era)) if (var_seed + var_era) > 0 else 0.0,
    }


def pooled_hier_mbb(D_folds: list[np.ndarray], n_boot: int = 2000,
                    rng: np.random.Generator | None = None) -> dict:
    """D8 pooled SECONDARY estimand: concatenated era-level paired differences
    across folds under one hierarchical MBB (block length from the pooled
    series' own lag-1 ACF). Cannot overturn the per-fold decision rule."""
    D = np.concatenate([np.asarray(d, dtype=float) for d in D_folds], axis=0)
    out = hier_mbb(D, n_boot=n_boot, rng=rng)
    out["estimand"] = ("D8 pooled secondary: concatenated era-level paired "
                       "differences, one hierarchical MBB; SECONDARY - cannot "
                       "overturn the per-fold rule")
    return out
