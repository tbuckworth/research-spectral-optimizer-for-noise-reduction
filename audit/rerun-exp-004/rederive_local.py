#!/usr/bin/env python3
"""AUDIT: independent re-derivation of exp-004/exp-005 headline numbers from
primary per-era CSV artifacts. Own bootstrap implementation, own RNG seed.
Reads only; writes nothing into experiment dirs."""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

RUN = Path("/media/titus/big/researcher-output/"
           "2026-07-31-spectral-optimizer-for-noise-reduction-on-financial-timeseri")
SEEDS = [0, 1, 2]
L = 4
NB = 20000
rng = np.random.default_rng(987654321)  # different RNG + seed from producer


def mbb_ci(x, n_boot=NB, L=L, rg=rng):
    """Own circular moving-block bootstrap (independent implementation)."""
    n = len(x)
    nblocks = int(np.ceil(n / L))
    starts = rg.integers(0, n, size=(n_boot, nblocks))
    idx = ((starts[..., None] + np.arange(L)) % n).reshape(n_boot, -1)[:, :n]
    means = x[idx].mean(axis=1)
    return x.mean(), np.percentile(means, 2.5), np.percentile(means, 97.5)


def arm_series(df, arm):
    return df[[f"{arm}_s{s}" for s in SEEDS]].mean(axis=1).to_numpy()


def audit_exp(name, csv, arms, proto):
    print(f"\n===== {name} =====")
    df = pd.read_csv(csv, dtype={"era": str})
    eras = [str(e).zfill(4) for e in df["era"]]
    verdict_block = [str(e) for e in proto["verdict_block"]]
    print(f"eras in CSV: {len(df)}; all in verdict block: "
          f"{set(eras) <= set(verdict_block)}; "
          f"covers whole block: {set(eras) == set(verdict_block)}")
    for a in arms:
        s = arm_series(df, a)
        per_seed = [df[f"{a}_s{x}"].mean() for x in SEEDS]
        print(f"  {a:11s} mean nc {s.mean():+.5f}  per-seed "
              + " ".join(f"{v:+.5f}" for v in per_seed)
              + f"  Sharpe {s.mean()/s.std(ddof=1):+.3f}")
    if "zero_pred" in df.columns:
        print(f"  zero_pred   mean nc {df['zero_pred'].mean():+.5f}")
    d = arm_series(df, "filter_on") - arm_series(df, "filter_off")
    m, lo, hi = mbb_ci(d)
    print(f"  HEADLINE on-off: {m:+.5f}  95% MBB CI [{lo:+.5f}, {hi:+.5f}]")
    t = stats.ttest_1samp(d, 0)
    w = stats.wilcoxon(d)
    print(f"    t-test p={t.pvalue:.2e}; wilcoxon p={w.pvalue:.2e}; "
          f"frac eras negative diff: {(d < 0).mean():.3f}")
    if "c4_random" in arms:
        d2 = arm_series(df, "filter_on") - arm_series(df, "c4_random")
        m2, lo2, hi2 = mbb_ci(d2)
        print(f"  C4: on-random: {m2:+.5f}  CI [{lo2:+.5f}, {hi2:+.5f}]")
        d3 = arm_series(df, "c4_random") - arm_series(df, "filter_off")
        m3, lo3, hi3 = mbb_ci(d3)
        print(f"  random-off:    {m3:+.5f}  CI [{lo3:+.5f}, {hi3:+.5f}]")
    for s in SEEDS:
        ds = (df[f"filter_on_s{s}"] - df[f"filter_off_s{s}"]).to_numpy()
        ms, los, his = mbb_ci(ds, n_boot=5000)
        print(f"  per-seed {s}: {ms:+.5f} CI [{los:+.5f}, {his:+.5f}]")
    # F10 Sharpe diff via joint block resample
    on, off = arm_series(df, "filter_on"), arm_series(df, "filter_off")
    n = len(on)
    nblocks = int(np.ceil(n / L))
    starts = rng.integers(0, n, size=(8000, nblocks))
    idx = ((starts[..., None] + np.arange(L)) % n).reshape(8000, -1)[:, :n]
    o, f = on[idx], off[idx]
    sh = o.mean(axis=1) / o.std(axis=1, ddof=1) - f.mean(axis=1) / f.std(axis=1, ddof=1)
    pt = on.mean() / on.std(ddof=1) - off.mean() / off.std(ddof=1)
    print(f"  F10 Sharpe diff: {pt:+.3f} CI "
          f"[{np.percentile(sh, 2.5):+.3f}, {np.percentile(sh, 97.5):+.3f}]")
    return df


proto = json.load(open(RUN / "experiments/exp-004/ref/protocol.json"))
df4 = audit_exp("exp-004 (MLP)",
                RUN / "experiments/exp-004/out/per_era_numerai_corr.csv",
                ["filter_off", "filter_on", "c4_random", "gaf"], proto)
df5 = audit_exp("exp-005 (GRU)",
                RUN / "experiments/exp-005/out/per_era_numerai_corr.csv",
                ["filter_off", "filter_on", "c4_random"], proto)

d = arm_series(df4, "gaf") - arm_series(df4, "filter_off")
m, lo, hi = mbb_ci(d)
print(f"\nexp-004 gaf-off: {m:+.5f} CI [{lo:+.5f}, {hi:+.5f}]")

# F3 re-derivation from exp-003 sweep artifacts
best = pd.read_csv(RUN / "experiments/exp-003/out/per_era_best_seeds.csv")
print("\n===== exp-003 F3 re-derivation =====")
print("columns:", list(best.columns))
seed_cols = [c for c in best.columns if c != "era"]
for c in seed_cols:
    print(f"  {c}: mean {best[c].mean():+.5f}")

# lag-1 acf of realized exp-004 diff series (F5 sanity)
d4 = arm_series(df4, "filter_on") - arm_series(df4, "filter_off")
ac = np.corrcoef(d4[:-1], d4[1:])[0, 1]
print(f"\nexp-004 realized diff-series lag-1 acf: {ac:+.3f} (L=4 block used)")
