#!/usr/bin/env python3
"""Sub-component #9: Numerai v5 data + era-purged protocol (local).

Implements the binding amendments:
  F1: validation-side eras split into a TUNING block and a temporally later,
      embargoed FINAL-VERDICT block. Boundaries recorded to JSON for reuse.
  F5: embargo >= ceil(target_horizon_days / era_spacing_days) eras.
      Numerai v5: main target `target` (= cyrus_20d) horizon 20 business
      days; era spacing 5 business days (weekly eras) -> embargo = 4 eras.
  F9: corr sanity band + leakage gate recalibrated from CURRENT v5
      example-model validation stats (per-era numerai_corr of
      validation_example_preds), not 2021 forum numbers.

Also builds the subsampled training shard for the cluster debug job
(#7/#1): last N_TRAIN_ERAS train eras, medium feature set (705), up to
ROWS_PER_ERA rows/era, int8 features + float32 target + era.

Outputs (all small except the shard):
  out/protocol.json        - F1 block boundaries, F5 embargo, era lists
  out/f9_recalibration.json- example-model per-era corr stats + bands
  out/example_per_era_corr.csv
  <run-dir>/data/shard_train.parquet - cluster shard
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import stats

SRC = Path(__file__).parent
OUT = SRC.parent / "out"
DATA = SRC.parent.parent.parent / "data"
OUT.mkdir(exist_ok=True)

# --- pre-registered constants (recorded in protocol.json) ---
TARGET_HORIZON_DAYS = 20   # main target `target` == cyrus 20D
ERA_SPACING_DAYS = 5       # v5 eras are weekly (5 business days)
EMBARGO = int(np.ceil(TARGET_HORIZON_DAYS / ERA_SPACING_DAYS))  # = 4
TUNING_FRACTION = 0.6      # of usable validation eras (temporally first 60%)
MIN_TARGET_COVERAGE = 0.95 # era usable if >=95% of rows have target
N_TRAIN_ERAS = 150         # last N train eras go into the cluster shard
ROWS_PER_ERA = 2560        # per-era row subsample in the shard


def numerai_corr(preds: np.ndarray, target: np.ndarray) -> float:
    """Official numerai_corr: rank+gaussianize preds, center target,
    accentuate tails (**1.5), Pearson."""
    ranked = stats.rankdata(preds, method="average") / (len(preds) + 1)
    gauss = stats.norm.ppf(ranked)
    p15 = np.sign(gauss) * np.abs(gauss) ** 1.5
    t = target - 0.5
    t15 = np.sign(t) * np.abs(t) ** 1.5
    return float(np.corrcoef(p15, t15)[0, 1])


def main():
    # ---------- validation eras & usability ----------
    vf = pq.ParquetFile(DATA / "v5.0_validation.parquet")
    print(f"validation.parquet: {vf.metadata.num_rows} rows, "
          f"{vf.metadata.num_columns} cols")
    vdf = vf.read(columns=["id", "era", "data_type", "target"]).to_pandas()
    if "id" not in vdf.columns:
        vdf = vdf.reset_index()
    print(f"loaded validation meta: {len(vdf)} rows, "
          f"mem {vdf.memory_usage(deep=True).sum()/1e9:.2f} GB")
    vdf = vdf[vdf["data_type"] == "validation"]
    cov = vdf.groupby("era")["target"].apply(lambda s: s.notna().mean())
    usable_eras = sorted(cov[cov >= MIN_TARGET_COVERAGE].index)
    all_val_eras = sorted(vdf["era"].unique())
    print(f"validation eras: {len(all_val_eras)} total "
          f"({all_val_eras[0]}..{all_val_eras[-1]}), "
          f"{len(usable_eras)} usable (target coverage >= {MIN_TARGET_COVERAGE})")

    # ---------- train eras ----------
    tf = pq.ParquetFile(DATA / "v5.0_train.parquet")
    tera = tf.read(columns=["era"]).to_pandas()["era"]
    train_eras = sorted(tera.unique())
    last_train_era = int(train_eras[-1])

    # ---------- F5 purge at train/validation boundary ----------
    # first usable validation era must leave EMBARGO eras after last train era
    boundary_ok = [e for e in usable_eras
                   if int(e) > last_train_era + EMBARGO]
    dropped_boundary = [e for e in usable_eras if e not in boundary_ok]

    # ---------- F1 tuning / verdict blocks with embargo ----------
    n = len(boundary_ok)
    n_tune = int(np.floor(n * TUNING_FRACTION))
    tuning_block = boundary_ok[:n_tune]
    # embargo: skip eras within EMBARGO of the last tuning era
    last_tune = int(tuning_block[-1])
    verdict_block = [e for e in boundary_ok[n_tune:]
                     if int(e) > last_tune + EMBARGO]
    embargoed = [e for e in boundary_ok[n_tune:] if e not in verdict_block]
    print(f"F1 blocks: tuning {len(tuning_block)} eras "
          f"({tuning_block[0]}..{tuning_block[-1]}), "
          f"embargo drops {len(embargoed)} eras {embargoed}, "
          f"verdict {len(verdict_block)} eras "
          f"({verdict_block[0]}..{verdict_block[-1]})")
    total_usable = len(tuning_block) + len(verdict_block)
    print(f"total usable validation-side eras across blocks: {total_usable} "
          f"(PASS needs >= 100)")

    # shard train eras: last N_TRAIN_ERAS, no purge needed on the train side
    # (they precede validation; the embargo is enforced on the validation side)
    shard_train_eras = train_eras[-N_TRAIN_ERAS:]

    protocol = {
        "dataset": "numerai v5.0",
        "main_target": "target (cyrus_20d)",
        "target_horizon_days": TARGET_HORIZON_DAYS,
        "era_spacing_days": ERA_SPACING_DAYS,
        "embargo_eras": EMBARGO,
        "embargo_rule": "ceil(target_horizon_days / era_spacing_days)",
        "min_target_coverage": MIN_TARGET_COVERAGE,
        "train_eras_total": len(train_eras),
        "last_train_era": last_train_era,
        "shard_train_eras": [str(e) for e in shard_train_eras],
        "validation_eras_total": len(all_val_eras),
        "validation_eras_usable": len(usable_eras),
        "dropped_at_train_boundary": [str(e) for e in dropped_boundary],
        "tuning_block": [str(e) for e in tuning_block],
        "tuning_block_range": [str(tuning_block[0]), str(tuning_block[-1])],
        "embargoed_between_blocks": [str(e) for e in embargoed],
        "verdict_block": [str(e) for e in verdict_block],
        "verdict_block_range": [str(verdict_block[0]), str(verdict_block[-1])],
        "total_usable_eras": total_usable,
    }
    with open(OUT / "protocol.json", "w") as f:
        json.dump(protocol, f, indent=2)
    print("wrote out/protocol.json")

    # ---------- F9: recalibrate sanity band from example model ----------
    ep = pq.ParquetFile(DATA / "v5.0_validation_example_preds.parquet")
    epdf = ep.read().to_pandas()
    print(f"example preds: {len(epdf)} rows, cols {list(epdf.columns)}")
    pred_col = [c for c in epdf.columns if c not in ("id", "era")][0]
    if "id" not in epdf.columns:
        epdf = epdf.reset_index()
    merged = vdf[["id", "era", "target"]].merge(
        epdf[["id", pred_col]], on="id", how="inner")
    merged = merged.dropna(subset=["target", pred_col])
    per_era = []
    for era in usable_eras:
        sub = merged[merged["era"] == era]
        if len(sub) < 100:
            continue
        nc = numerai_corr(sub[pred_col].to_numpy(), sub["target"].to_numpy())
        sp = stats.spearmanr(sub[pred_col], sub["target"]).statistic
        per_era.append({"era": str(era), "numerai_corr": nc, "spearman": sp,
                        "n": len(sub)})
    pe = pd.DataFrame(per_era)
    pe.to_csv(OUT / "example_per_era_corr.csv", index=False)
    m, s = pe["numerai_corr"].mean(), pe["numerai_corr"].std()
    sharpe = m / s if s > 0 else float("nan")
    lag1 = pe["numerai_corr"].autocorr(lag=1)
    ms, ss = pe["spearman"].mean(), pe["spearman"].std()
    print(f"F9 example model over {len(pe)} usable val eras: "
          f"numerai_corr mean {m:.4f} sd {s:.4f} sharpe {sharpe:.2f}; "
          f"spearman mean {ms:.4f} sd {ss:.4f}; lag1 autocorr {lag1:.3f}")

    # blocks separately
    def blk(eras):
        sub = pe[pe["era"].isin([str(e) for e in eras])]
        return {"n_eras": len(sub),
                "numerai_corr_mean": float(sub["numerai_corr"].mean()),
                "numerai_corr_sd": float(sub["numerai_corr"].std()),
                "spearman_mean": float(sub["spearman"].mean())}

    f9 = {
        "example_model_file": "v5.0/validation_example_preds.parquet",
        "n_eras": len(pe),
        "numerai_corr_mean": float(m),
        "numerai_corr_sd": float(s),
        "numerai_corr_sharpe": float(sharpe),
        "spearman_mean": float(ms),
        "spearman_sd": float(ss),
        "lag1_autocorr_per_era_corr": float(lag1),
        "tuning_block_stats": blk(tuning_block),
        "verdict_block_stats": blk(verdict_block),
        # pre-registered recalibrated bands (rules fixed here, numbers from data)
        "sanity_band_rule": "working-model mean per-era numerai_corr in "
                            "[0.25x, 1.5x] of example-model mean",
        "sanity_band": [0.25 * float(m), 1.5 * float(m)],
        "leakage_gate_rule": "mean per-era numerai_corr > 2x example mean "
                             "OR mean per-era spearman > 0.06 => leakage suspect",
        "leakage_gate_numerai_corr": 2.0 * float(m),
        "leakage_gate_spearman": 0.06,
    }
    with open(OUT / "f9_recalibration.json", "w") as f:
        json.dump(f9, f, indent=2)
    print("wrote out/f9_recalibration.json")

    # ---------- build cluster shard ----------
    feats = json.load(open(DATA / "v5.0_features.json"))
    medium = feats["feature_sets"]["medium"]
    cols = ["era", "target"] + medium
    print(f"building shard: {len(shard_train_eras)} eras x <= {ROWS_PER_ERA} "
          f"rows, {len(medium)} medium features")
    tdf = pq.read_table(DATA / "v5.0_train.parquet", columns=cols,
                        filters=[("era", "in", set(shard_train_eras))]
                        ).to_pandas()
    print(f"read {len(tdf)} rows, mem "
          f"{tdf.memory_usage(deep=True).sum()/1e9:.2f} GB")
    rng = np.random.default_rng(0)
    parts = []
    for era, sub in tdf.groupby("era", observed=True):
        if len(sub) > ROWS_PER_ERA:
            sub = sub.iloc[rng.choice(len(sub), ROWS_PER_ERA, replace=False)]
        parts.append(sub)
    shard = pd.concat(parts).reset_index(drop=True)
    shard["target"] = shard["target"].astype("float32")
    shard.dropna(subset=["target"], inplace=True)
    shard_path = DATA / "shard_train.parquet"
    shard.to_parquet(shard_path, index=False)
    sz = shard_path.stat().st_size / 1e6
    print(f"shard: {len(shard)} rows, {len(shard['era'].unique())} eras, "
          f"{sz:.0f} MB -> {shard_path}")


if __name__ == "__main__":
    main()
