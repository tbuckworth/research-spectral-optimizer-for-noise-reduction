#!/usr/bin/env python3
"""AUDIT: analyze the fresh-seed re-execution of exp-004 (job 6616).
Seeds {10,11,12}; evaluated on (a) the producer's verdict shard and
(b) the audit's independently-built verdict shard. Own MBB implementation."""
import numpy as np
import pandas as pd
from pathlib import Path

HERE = Path(__file__).parent
SEEDS = [10, 11, 12]
L = 4
rng = np.random.default_rng(1357911)


def mbb_ci(x, n_boot=20000, L=L, rg=rng):
    n = len(x)
    nblocks = int(np.ceil(n / L))
    starts = rg.integers(0, n, size=(n_boot, nblocks))
    idx = ((starts[..., None] + np.arange(L)) % n).reshape(n_boot, -1)[:, :n]
    means = x[idx].mean(axis=1)
    return x.mean(), np.percentile(means, 2.5), np.percentile(means, 97.5)


def arm(df, a):
    return df[[f"{a}_s{s}" for s in SEEDS]].mean(axis=1).to_numpy()


for label, fname in [("PRODUCER shard", "per_era_numerai_corr.csv"),
                     ("AUDIT shard (own subsample)",
                      "per_era_numerai_corr_AUDITSHARD.csv")]:
    df = pd.read_csv(HERE / "out" / fname, dtype={"era": str})
    print(f"\n===== re-run seeds {SEEDS} on {label} ({len(df)} eras) =====")
    for a in ["filter_off", "filter_on", "c4_random", "gaf"]:
        cols = [f"{a}_s{s}" for s in SEEDS]
        if not all(c in df.columns for c in cols):
            continue
        s = arm(df, a)
        print(f"  {a:11s} mean nc {s.mean():+.5f}  per-seed "
              + " ".join(f"{df[c].mean():+.5f}" for c in cols))
    d = arm(df, "filter_on") - arm(df, "filter_off")
    m, lo, hi = mbb_ci(d)
    cat = ("hurts" if hi < 0 and m <= -0.00398 else
           "helps" if lo > 0 and m >= 0.00398 else
           "doesnt_help" if lo > -0.00398 and hi < 0.00398 else "no_verdict")
    print(f"  HEADLINE on-off: {m:+.5f} CI [{lo:+.5f}, {hi:+.5f}] "
          f"-> {cat} (vs F3=0.00398)")
    if all(f"c4_random_s{s}" in df.columns for s in SEEDS):
        d2 = arm(df, "filter_on") - arm(df, "c4_random")
        m2, lo2, hi2 = mbb_ci(d2)
        print(f"  C4 on-random:   {m2:+.5f} CI [{lo2:+.5f}, {hi2:+.5f}]")

# combined 6-seed evidence: original seeds 0-2 + rerun 10-12 (producer shard)
orig = pd.read_csv(HERE.parent.parent /
                   "experiments/exp-004/out/per_era_numerai_corr.csv",
                   dtype={"era": str})
rerun = pd.read_csv(HERE / "out" / "per_era_numerai_corr.csv",
                    dtype={"era": str})
assert (orig["era"].values == rerun["era"].values).all()
on6 = pd.concat([orig[[f"filter_on_s{s}" for s in [0, 1, 2]]],
                 rerun[[f"filter_on_s{s}" for s in SEEDS]]], axis=1).mean(axis=1)
off6 = pd.concat([orig[[f"filter_off_s{s}" for s in [0, 1, 2]]],
                  rerun[[f"filter_off_s{s}" for s in SEEDS]]], axis=1).mean(axis=1)
d6 = (on6 - off6).to_numpy()
m, lo, hi = mbb_ci(d6)
print(f"\n6-SEED pooled headline (seeds 0-2 + 10-12, producer shard): "
      f"{m:+.5f} CI [{lo:+.5f}, {hi:+.5f}]")
