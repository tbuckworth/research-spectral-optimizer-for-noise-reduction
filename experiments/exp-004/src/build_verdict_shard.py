#!/usr/bin/env python3
"""Build the VERDICT-block evaluation shard for exp-004 (local).

F1 (binding): loads ONLY verdict-block eras (0971-1225) from
v5.0_validation.parquet via a parquet-level era filter. THIS IS THE FIRST
AND ONLY UNBLINDING OF THE VERDICT BLOCK. No config choice depends on it:
all arms/configs are frozen in code (src/verdict_job.py) before any
evaluation run touches this shard. This script computes no performance
statistic of any kind — it only subsamples rows.

Replicates exp-001/src/data_prep.py + exp-003/src/build_tuning_shard.py
shard conventions exactly: 705 medium features, <=2560 rows/era (rng seed
0), int8 features + float32 target + era, NaN targets dropped.

Output: <run-dir>/data/shard_verdict.parquet (kept out of the experiment
dir per the >50MB rule).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

SRC = Path(__file__).parent
EXP = SRC.parent
DATA = EXP.parent.parent / "data"
ROWS_PER_ERA = 2560  # matches exp-001 train shard / exp-003 tuning shard

proto = json.load(open(EXP / "ref" / "protocol.json"))
verdict = [str(e) for e in proto["verdict_block"]]
tuning = set(proto["tuning_block"])
train = set(proto["shard_train_eras"])
embargo = set(proto["embargoed_between_blocks"])
assert not set(verdict) & (tuning | train | embargo)

feats = json.load(open(DATA / "v5.0_features.json"))
medium = feats["feature_sets"]["medium"]
cols = ["era", "data_type", "target"] + medium
print(f"reading verdict-block eras {verdict[0]}..{verdict[-1]} "
      f"({len(verdict)} eras), {len(medium)} medium features")

df = pq.read_table(DATA / "v5.0_validation.parquet", columns=cols,
                   filters=[("era", "in", set(verdict))]).to_pandas()
print(f"read {len(df)} rows, mem {df.memory_usage(deep=True).sum()/1e9:.2f} GB")
df = df[df["data_type"] == "validation"].drop(columns=["data_type"])

# F1 hard assert: nothing outside the verdict block was loaded
loaded = set(df["era"].astype(str).unique())
assert loaded <= set(verdict), \
    f"non-verdict eras loaded: {sorted(loaded - set(verdict))[:5]}"
assert not loaded & (tuning | train | embargo), \
    "F1 VIOLATION: non-verdict-block eras loaded"

rng = np.random.default_rng(0)
parts = []
for era, sub in df.groupby("era", observed=True):
    if len(sub) > ROWS_PER_ERA:
        sub = sub.iloc[rng.choice(len(sub), ROWS_PER_ERA, replace=False)]
    parts.append(sub)
shard = pd.concat(parts).reset_index(drop=True)
shard["target"] = shard["target"].astype("float32")
shard.dropna(subset=["target"], inplace=True)
out = DATA / "shard_verdict.parquet"
shard.to_parquet(out, index=False)
sz = out.stat().st_size / 1e6
print(f"shard: {len(shard)} rows, {len(shard['era'].unique())} eras, "
      f"{sz:.0f} MB -> {out}")
