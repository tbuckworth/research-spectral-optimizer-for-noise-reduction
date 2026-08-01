#!/usr/bin/env python3
"""Build the TUNING-block evaluation shard for exp-003 (local).

F1 (binding): loads ONLY tuning-block eras (0579-0966) from
v5.0_validation.parquet via a parquet-level era filter. The verdict block
(0971-1225) is never read. Replicates exp-001/src/data_prep.py shard
conventions exactly: 705 medium features, <=2560 rows/era (rng seed 0),
int8 features + float32 target + era, NaN targets dropped.

Output: <run-dir>/data/shard_tuning.parquet (kept out of the experiment
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
ROWS_PER_ERA = 2560  # matches exp-001 train shard

proto = json.load(open(EXP / "ref" / "protocol.json"))
tuning = [str(e) for e in proto["tuning_block"]]
verdict = set(proto["verdict_block"])
assert not set(tuning) & verdict

feats = json.load(open(DATA / "v5.0_features.json"))
medium = feats["feature_sets"]["medium"]
cols = ["era", "data_type", "target"] + medium
print(f"reading tuning-block eras {tuning[0]}..{tuning[-1]} "
      f"({len(tuning)} eras), {len(medium)} medium features")

df = pq.read_table(DATA / "v5.0_validation.parquet", columns=cols,
                   filters=[("era", "in", set(tuning))]).to_pandas()
print(f"read {len(df)} rows, mem {df.memory_usage(deep=True).sum()/1e9:.2f} GB")
df = df[df["data_type"] == "validation"].drop(columns=["data_type"])

# F1 hard assert: nothing outside the tuning block was loaded
loaded = set(df["era"].astype(str).unique())
assert loaded <= set(tuning), f"non-tuning eras loaded: {sorted(loaded - set(tuning))[:5]}"
assert not loaded & verdict, "F1 VIOLATION: verdict-block eras loaded"

rng = np.random.default_rng(0)
parts = []
for era, sub in df.groupby("era", observed=True):
    if len(sub) > ROWS_PER_ERA:
        sub = sub.iloc[rng.choice(len(sub), ROWS_PER_ERA, replace=False)]
    parts.append(sub)
shard = pd.concat(parts).reset_index(drop=True)
shard["target"] = shard["target"].astype("float32")
shard.dropna(subset=["target"], inplace=True)
out = DATA / "shard_tuning.parquet"
shard.to_parquet(out, index=False)
sz = out.stat().st_size / 1e6
print(f"shard: {len(shard)} rows, {len(shard['era'].unique())} eras, "
      f"{sz:.0f} MB -> {out}")
