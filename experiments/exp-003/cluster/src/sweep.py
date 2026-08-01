#!/usr/bin/env python3
"""exp-003 sub-component #8: tuned-AdamW baseline sanity sweep (cluster GPU).

Plain-AdamW hyperparameter sweep for the MLP arm (model/loss identical to
exp-001's cluster_job.py MLP arm: MLP 705-256-64-1, MSE on centered target),
evaluated by mean per-era Spearman corr (selection metric, per plan #8) and
mean per-era numerai_corr (F9 band metric) on TUNING-block eras ONLY.

Binding amendments implemented here:
  F1:  train = train-side shard (eras 0425-0574); eval = tuning-block shard
       (eras 0579-0966). The verdict block (0971-1225) is never loaded; an
       assert against ref/protocol.json enforces it on both shards.
  F12: search grid centered on the prior-repo priors (lr=1e-3, wd=0,
       dropout=0 -- the Adam/AdamW center used throughout ~/pyg/optimizers).
       Trials x time recorded for the matched-compute budget argument.
       Under-exploration signatures (best-on-grid-boundary, non-monotone
       sweep shape) checked downstream in analyze.py from the saved table.
  F9:  best trial judged against ref/f9_recalibration.json band (printed
       here; verdict finalized in results.md).
  F5:  per-era corr vectors saved as CSVs for the lag-1 autocorr / block
       bootstrap analysis (local).

Also: zero-predictor control (seeded random preds, no information), and
3 seeds of the best config to estimate per-era seed noise for the #3 power
simulation.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats

DEV = "cuda" if torch.cuda.is_available() else "cpu"
STEPS = 2000          # = exp-001 FULL_RUN_STEPS (one "full training run")
BATCH = 1024
EVAL_CHUNK = 65536
MIN_ERA_ROWS = 100

# F12: grid centered on prior-repo priors lr=1e-3, wd=0.0, dropout=0.0
LRS = [3e-4, 1e-3, 3e-3]
WDS = [0.0, 1e-3]
DROPOUTS = [0.0, 0.2]
SWEEP_SEED = 0
EXTRA_SEEDS = [1, 2]  # best config re-trained with these for seed noise


def numerai_corr_centered(preds: np.ndarray, y_centered: np.ndarray) -> float:
    """Official numerai_corr; y_centered = target - 0.5 (as stored)."""
    ranked = stats.rankdata(preds, method="average") / (len(preds) + 1)
    gauss = stats.norm.ppf(ranked)
    p15 = np.sign(gauss) * np.abs(gauss) ** 1.5
    t15 = np.sign(y_centered) * np.abs(y_centered) ** 1.5
    return float(np.corrcoef(p15, t15)[0, 1])


def load_shard(path, allowed_eras, name):
    df = pd.read_parquet(path)
    eras = sorted(df["era"].astype(str).unique())
    bad = [e for e in eras if e not in allowed_eras]
    assert not bad, f"F1 VIOLATION: {name} contains disallowed eras {bad[:5]}"
    feat_cols = [c for c in df.columns if c.startswith("feature")]
    X = torch.tensor(df[feat_cols].to_numpy(dtype=np.float32), device=DEV)
    X = (X - 2.0) / 2.0
    y = torch.tensor(df["target"].to_numpy(dtype=np.float32), device=DEV) - 0.5
    era_codes, era_uniques = pd.factorize(df["era"].astype(str), sort=True)
    print(f"{name}: {X.shape[0]} rows, {X.shape[1]} features, "
          f"{len(era_uniques)} eras ({era_uniques[0]}..{era_uniques[-1]})",
          flush=True)
    return X, y, np.asarray(era_codes), list(map(str, era_uniques))


def make_mlp(d, dropout, seed):
    torch.manual_seed(seed)
    layers = [nn.Linear(d, 256), nn.ReLU()]
    if dropout > 0:
        layers.append(nn.Dropout(dropout))
    layers += [nn.Linear(256, 64), nn.ReLU()]
    if dropout > 0:
        layers.append(nn.Dropout(dropout))
    layers.append(nn.Linear(64, 1))
    return nn.Sequential(*layers).to(DEV)


def train_one(Xtr, ytr, lr, wd, dropout, seed):
    model = make_mlp(Xtr.shape[1], dropout, seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    g = torch.Generator(device="cpu").manual_seed(seed)
    n = Xtr.shape[0]
    model.train()
    t0 = time.time()
    for step in range(STEPS):
        idx = torch.randint(0, n, (BATCH,), generator=g).to(DEV)
        opt.zero_grad(set_to_none=True)
        loss = F.mse_loss(model(Xtr[idx]).squeeze(-1), ytr[idx])
        loss.backward()
        opt.step()
        if step % 400 == 0:
            print(f"    step {step:4d} loss {loss.item():.5f}", flush=True)
    if DEV == "cuda":
        torch.cuda.synchronize()
    return model, time.time() - t0


@torch.no_grad()
def predict(model, X):
    model.eval()
    out = []
    for i in range(0, X.shape[0], EVAL_CHUNK):
        out.append(model(X[i:i + EVAL_CHUNK]).squeeze(-1).cpu())
    return torch.cat(out).numpy()


def per_era_metrics(preds, y_np, era_codes, era_names):
    rows = []
    for code, name in enumerate(era_names):
        m = era_codes == code
        if m.sum() < MIN_ERA_ROWS:
            continue
        sp = stats.spearmanr(preds[m], y_np[m]).statistic
        nc = numerai_corr_centered(preds[m], y_np[m])
        rows.append({"era": name, "spearman": float(sp),
                     "numerai_corr": float(nc), "n": int(m.sum())})
    return pd.DataFrame(rows)


def summarize(pe):
    return {"mean_spearman": float(pe["spearman"].mean()),
            "sd_spearman": float(pe["spearman"].std()),
            "mean_numerai_corr": float(pe["numerai_corr"].mean()),
            "sd_numerai_corr": float(pe["numerai_corr"].std()),
            "n_eras": int(len(pe))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-shard", required=True)
    ap.add_argument("--tune-shard", required=True)
    ap.add_argument("--ref", required=True, help="dir with protocol.json etc.")
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"device: {DEV}, torch {torch.__version__}", flush=True)
    if DEV == "cuda":
        print(f"gpu: {torch.cuda.get_device_name(0)}", flush=True)

    proto = json.load(open(Path(args.ref) / "protocol.json"))
    f9 = json.load(open(Path(args.ref) / "f9_recalibration.json"))
    train_allowed = set(proto["shard_train_eras"])
    tune_allowed = set(proto["tuning_block"])
    # F1 hard assert: tuning block must not intersect verdict block
    assert not tune_allowed & set(proto["verdict_block"])

    t_start = time.time()
    Xtr, ytr, _, _ = load_shard(args.train_shard, train_allowed, "train shard")
    Xtu, ytu, era_tu, eras_tu = load_shard(args.tune_shard, tune_allowed,
                                           "tuning shard")
    ytu_np = ytu.cpu().numpy()

    # ---------- zero-predictor control (no information) ----------
    rng = np.random.default_rng(12345)
    zp = rng.standard_normal(Xtu.shape[0])
    pe_zero = per_era_metrics(zp, ytu_np, era_tu, eras_tu)
    pe_zero.to_csv(outdir / "per_era_zero_predictor.csv", index=False)
    zs = summarize(pe_zero)
    print(f"\nzero-predictor: mean spearman {zs['mean_spearman']:+.5f}, "
          f"mean numerai_corr {zs['mean_numerai_corr']:+.5f} "
          f"over {zs['n_eras']} eras", flush=True)

    # ---------- sweep ----------
    trials = []
    per_era_sp = pd.DataFrame({"era": [r["era"] for _, r in
                                       pe_zero.iterrows()]})
    per_era_nc = per_era_sp.copy()
    tid = 0
    for lr in LRS:
        for wd in WDS:
            for do in DROPOUTS:
                tag = f"t{tid:02d}_lr{lr:g}_wd{wd:g}_do{do:g}"
                print(f"\n### trial {tag}", flush=True)
                model, ttrain = train_one(Xtr, ytr, lr, wd, do, SWEEP_SEED)
                te0 = time.time()
                preds = predict(model, Xtu)
                pe = per_era_metrics(preds, ytu_np, era_tu, eras_tu)
                teval = time.time() - te0
                s = summarize(pe)
                trials.append({"trial": tid, "tag": tag, "lr": lr, "wd": wd,
                               "dropout": do, "seed": SWEEP_SEED,
                               "train_seconds": ttrain,
                               "eval_seconds": teval, **s})
                per_era_sp[tag] = pe["spearman"].to_numpy()
                per_era_nc[tag] = pe["numerai_corr"].to_numpy()
                print(f"  {tag}: mean spearman {s['mean_spearman']:+.5f}, "
                      f"mean numerai_corr {s['mean_numerai_corr']:+.5f}, "
                      f"train {ttrain:.1f}s eval {teval:.1f}s", flush=True)
                del model
                tid += 1

    tdf = pd.DataFrame(trials)
    best = tdf.loc[tdf["mean_spearman"].idxmax()]
    print(f"\nBEST trial (by mean per-era spearman): {best['tag']} "
          f"spearman {best['mean_spearman']:+.5f} "
          f"numerai_corr {best['mean_numerai_corr']:+.5f}", flush=True)

    # ---------- best config: extra seeds for seed-noise estimate ----------
    seed_frames = {SWEEP_SEED: {"spearman": per_era_sp[best["tag"]],
                                "numerai_corr": per_era_nc[best["tag"]]}}
    seed_times = []
    for sd in EXTRA_SEEDS:
        print(f"\n### best config re-train, seed {sd}", flush=True)
        model, ttrain = train_one(Xtr, ytr, best["lr"], best["wd"],
                                  best["dropout"], sd)
        preds = predict(model, Xtu)
        pe = per_era_metrics(preds, ytu_np, era_tu, eras_tu)
        s = summarize(pe)
        seed_times.append(ttrain)
        print(f"  seed {sd}: mean spearman {s['mean_spearman']:+.5f}, "
              f"mean numerai_corr {s['mean_numerai_corr']:+.5f} "
              f"({ttrain:.1f}s)", flush=True)
        seed_frames[sd] = {"spearman": pe["spearman"].to_numpy(),
                           "numerai_corr": pe["numerai_corr"].to_numpy()}
        del model

    bs = pd.DataFrame({"era": per_era_sp["era"]})
    for sd, d in seed_frames.items():
        bs[f"spearman_seed{sd}"] = np.asarray(d["spearman"])
        bs[f"numerai_corr_seed{sd}"] = np.asarray(d["numerai_corr"])
    bs.to_csv(outdir / "per_era_best_seeds.csv", index=False)
    per_era_sp.to_csv(outdir / "per_era_spearman_all_trials.csv", index=False)
    per_era_nc.to_csv(outdir / "per_era_numerai_all_trials.csv", index=False)

    # ---------- F9 band / leakage gate (informational print) ----------
    band = f9["sanity_band"]
    gate_nc, gate_sp = f9["leakage_gate_numerai_corr"], f9["leakage_gate_spearman"]
    m_nc, m_sp = best["mean_numerai_corr"], best["mean_spearman"]
    print(f"\nF9 band [{band[0]:.4f}, {band[1]:.4f}] (numerai_corr): "
          f"best {m_nc:+.5f} -> {'INSIDE' if band[0] <= m_nc <= band[1] else 'OUTSIDE'}",
          flush=True)
    leak = (m_nc > gate_nc) or (m_sp > gate_sp)
    print(f"leakage gate (nc>{gate_nc:.4f} or sp>{gate_sp:.4f}): "
          f"{'FIRED' if leak else 'not fired'}", flush=True)

    out = {"device": DEV, "torch": torch.__version__,
           "steps": STEPS, "batch": BATCH, "sweep_seed": SWEEP_SEED,
           "grid": {"lr": LRS, "wd": WDS, "dropout": DROPOUTS},
           "f12_search_center": {"lr": 1e-3, "wd": 0.0, "dropout": 0.0,
                                 "source": "~/pyg/optimizers (Adam/AdamW center)"},
           "trials": trials,
           "best": {k: (float(v) if isinstance(v, (int, float, np.floating))
                        else v) for k, v in best.items()},
           "best_extra_seed_train_seconds": seed_times,
           "zero_predictor": zs,
           "f9_band": band, "leakage_gate_fired": bool(leak),
           "total_job_seconds": time.time() - t_start}
    with open(outdir / "sweep_results.json", "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\nTotal sweep time: {(time.time()-t_start)/60:.1f} min; "
          f"wrote sweep_results.json + per-era CSVs", flush=True)


if __name__ == "__main__":
    main()
