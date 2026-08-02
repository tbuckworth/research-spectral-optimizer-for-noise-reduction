#!/usr/bin/env python3
"""AUDIT: self-constructed verdict-era evaluation shard.
Same frozen verdict era range (0971-1225; the pre-registered block), but an
INDEPENDENT row subsample (different RNG, different seed) built by audit code
directly from the raw v5.0_validation.parquet — not the producer's shard.
Also cross-checks the producer's shard row-values against the raw parquet."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

RUN = Path("/media/titus/big/researcher-output/"
           "2026-07-31-spectral-optimizer-for-noise-reduction-on-financial-timeseri")
OUT = RUN / "audit" / "rerun-exp-004"
ROWS_PER_ERA = 2560

proto = json.load(open(RUN / "experiments/exp-004/ref/protocol.json"))
verdict = [str(e) for e in proto["verdict_block"]]
feats = json.load(open(RUN / "data/v5.0_features.json"))
medium = feats["feature_sets"]["medium"]
cols = ["era", "data_type", "target"] + medium

df = pq.read_table(RUN / "data/v5.0_validation.parquet", columns=cols,
                   filters=[("era", "in", set(verdict))]).to_pandas()
df = df[df["data_type"] == "validation"].drop(columns=["data_type"])
loaded = set(df["era"].astype(str).unique())
assert loaded == set(verdict), "era mismatch vs frozen verdict block"

# --- cross-check the producer's shard against raw data (leak/fabrication) --
prod = pd.read_parquet(RUN / "data/shard_verdict.parquet")
print(f"producer shard: {len(prod)} rows, {prod['era'].nunique()} eras")
assert set(prod["era"].astype(str).unique()) == set(verdict)
# per-era row counts within raw availability + target distribution match
raw_counts = df.groupby(df["era"].astype(str)).size()
prod_counts = prod.groupby(prod["era"].astype(str)).size()
assert (prod_counts <= np.minimum(raw_counts, ROWS_PER_ERA)).all()
# spot-check 3 eras: every producer row must exist in raw (merge on all cols)
rng0 = np.random.default_rng(11)
for era in rng0.choice(verdict, 3, replace=False):
    a = prod[prod["era"].astype(str) == era].reset_index(drop=True)
    b = df[df["era"].astype(str) == era].reset_index(drop=True)
    a2 = a.astype({c: "int8" for c in medium})
    b2 = b.astype({c: "int8" for c in medium})
    merged = a2.merge(b2, on=list(a2.columns), how="left", indicator=True)
    frac = (merged["_merge"] == "both").mean()
    print(f"  era {era}: producer rows found in raw: {frac:.4f}")
    assert frac > 0.999, f"producer shard rows NOT in raw data for era {era}"
print("producer shard cross-check vs raw parquet: OK")

# --- build audit shard: independent subsample -----------------------------
rng = np.random.default_rng(424242)  # audit's own seed
parts = []
for era, sub in df.groupby("era", observed=True):
    if len(sub) > ROWS_PER_ERA:
        sub = sub.iloc[rng.choice(len(sub), ROWS_PER_ERA, replace=False)]
    parts.append(sub)
shard = pd.concat(parts).reset_index(drop=True)
shard["target"] = shard["target"].astype("float32")
shard.dropna(subset=["target"], inplace=True)
out = OUT / "shard_verdict_audit.parquet"
shard.to_parquet(out, index=False)
print(f"audit shard: {len(shard)} rows, {shard['era'].nunique()} eras -> {out}")
