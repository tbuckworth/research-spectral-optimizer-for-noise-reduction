#!/usr/bin/env python3
"""exp-004: seeded main comparison + mechanism controls (the VERDICT job).

ONE Slurm GPU job. ALL ARM CONFIGS ARE FROZEN IN THIS FILE, IN CODE, BEFORE
ANY EVALUATION TOUCHES THE VERDICT SHARD (F1: first and only unblinding).

Frozen inputs (binding, reused verbatim — see plan.md):
  F1: train = shard_train (eras 0425-0574); eval = shard_verdict
      (eras 0971-1225, 255 eras). Hard asserts against ref/protocol.json.
  F2: threshold mode = hard, mp_factor = 2.0 (SpectralConsensusFilter,
      exact B x B MP-threshold filter from spectral_optimizer.py — F11).
  F6: within-era batch composition, B = 1024.
  Baseline config t07 (exp-003): lr 1e-3, wd 1e-3, dropout 0.2,
      2000 steps @ B=1024. MLP 705-256-64-1 (197k params).
  F12 spectral-arm transferred priors (recorded): the prior repo
      (~/pyg/optimizers) center = filter defaults mp_factor=2.0,
      row-normalized similarity, hard threshold; base-optimizer center
      lr=1e-3/wd=0/do=0 was already superseded by the plan's binding rule
      that ALL arms share the SAME tuned base config t07. One spectral
      setting only (~1 matched-compute trial).

Arms (same train shard, same base config, same seeds {0,1,2}, identical
data order per seed via a per-seed within-era Sampler):
  1. filter_off : plain AdamW t07.
  2. filter_on  : SpectralConsensusFilter (hard, mp_factor 2.0) on AdamW t07.
  3. c4_random  : C4 norm-matched random-subspace control. Matching rule
     (pre-registered): at step t, project the uniform weight vector onto a
     RANDOM k(t)-dim sample-subspace (random B x k(t) orthonormal basis,
     dedicated RNG seed+777), where k(t) is the filter_on run's realized k
     at step t for the SAME seed; the resulting update is rescaled so that
     ||update|| / ||mean_grad|| equals the filter_on run's realized
     consensus_ratio at step t. This matches both the k trajectory and the
     kept-norm(-ratio) trajectory exactly; only WHICH subspace is kept
     differs (MP-eigenselected vs random).
  4. gaf        : GAF-style simple-agreement ablation (pre-registered rule):
     coordinate-wise sign agreement across the B per-sample gradients;
     keep coordinate j iff |sum_i sign(G_ij)| >= 2*sqrt(B) (agreement
     beyond 2 sigma of the random-sign null); update = mean gradient
     masked to kept coordinates. No eigendecomposition.

Per-sample gradients: torch.func vmap with randomness="different" (t07 has
dropout; the filter's own _setup_vmap predates dropout). The filtering math
itself is the verbatim SpectralConsensusFilter._spectral_filter (F11).

Diagnostics every LOG_EVERY=50 steps in all filtered arms: k, consensus
ratio, kept-energy fraction (filter_on, from a full eigenspectrum), C3
cosine (filtered vs mean gradient), GAF fraction of coordinates kept.
Full per-step k/ratio trajectories saved for the C4 matching + audit.

Evaluation: after training, each (arm, seed) model predicts ONCE on the
verdict shard; per-era numerai_corr and spearman saved to CSVs. Plus a
seeded zero-predictor sanity column. Nothing else touches the verdict shard.
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
from torch.func import vmap, grad_and_value, functional_call
from scipy import stats

import sys
sys.path.insert(0, str(Path(__file__).parent))
from spectral_optimizer import SpectralConsensusFilter  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------- FROZEN CONFIG (do not modify after submission) ----------
STEPS = 2000
BATCH = 1024
LR = 1e-3          # t07
WD = 1e-3          # t07
DROPOUT = 0.2      # t07
SEEDS = [0, 1, 2]
MP_FACTOR = 2.0    # F2
LOG_EVERY = 50
EVAL_CHUNK = 65536
MIN_ERA_ROWS = 100
GAF_SIGMA = 2.0    # keep coord iff |sum sign| >= GAF_SIGMA * sqrt(B)
ARM_ORDER = ["filter_off", "filter_on", "c4_random", "gaf"]
# --------------------------------------------------------------------------


def numerai_corr_centered(preds: np.ndarray, y_centered: np.ndarray) -> float:
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


class Sampler:
    """Within-era batch sampler (F6), seeded per training seed so every arm
    at the same seed sees the IDENTICAL batch sequence."""

    def __init__(self, era_codes, B, seed):
        self.B = B
        self.g = torch.Generator(device="cpu").manual_seed(seed)
        era_cpu = torch.tensor(era_codes)
        self.era_ids = era_cpu.unique()
        self.era_rows = {int(e): (era_cpu == e).nonzero().squeeze(-1)
                         for e in self.era_ids}
        self.big_eras = [e for e, r in self.era_rows.items() if len(r) >= B]

    def batch(self):
        e = self.big_eras[int(torch.randint(0, len(self.big_eras), (1,),
                                            generator=self.g))]
        rows = self.era_rows[e]
        perm = torch.randperm(len(rows), generator=self.g)
        return rows[perm[:self.B]].to(DEV)


def make_grad_fn(model):
    """Per-sample grad fn with randomness='different' (dropout-compatible)."""
    def compute_loss(params, buffers, x, y):
        out = functional_call(model, (params, buffers), (x.unsqueeze(0),))
        return F.mse_loss(out.squeeze(-1), y.unsqueeze(0))
    return vmap(grad_and_value(compute_loss, has_aux=False),
                in_dims=(None, None, 0, 0), randomness="different")


def cos(a, b):
    na, nb = a.norm(), b.norm()
    if na < 1e-12 or nb < 1e-12:
        return float("nan")
    return float((a @ b / (na * nb)).item())


def kept_energy_fraction(G, mp_factor):
    """Bounded kept-energy diagnostic from the full normalized spectrum."""
    norms = G.norm(dim=1, keepdim=True).clamp(min=1e-12)
    Gn = G / norms
    ev = torch.linalg.eigvalsh(Gn @ Gn.T).flip(0).clamp(min=0)
    tot = ev.sum()
    if tot < 1e-12:
        return 0.0
    thr = mp_factor * tot / G.shape[0]
    return float((ev[ev > thr].sum() / tot).item())


def train_filter_off(Xtr, ytr, era_codes, seed, steps, B):
    model = make_mlp(Xtr.shape[1], DROPOUT, seed)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sampler = Sampler(era_codes, B, seed)
    model.train()
    t0 = time.time()
    for step in range(steps):
        idx = sampler.batch()
        opt.zero_grad(set_to_none=True)
        loss = F.mse_loss(model(Xtr[idx]).squeeze(-1), ytr[idx])
        loss.backward()
        opt.step()
        if step % 400 == 0:
            print(f"    step {step:4d} loss {loss.item():.5f}", flush=True)
    if DEV == "cuda":
        torch.cuda.synchronize()
    return model, time.time() - t0, {"log": []}, None


def train_filtered(Xtr, ytr, era_codes, seed, steps, B, arm,
                   match_traj=None):
    """arm in {filter_on, c4_random, gaf}. match_traj: dict with per-step
    'k' and 'ratio' arrays from the filter_on run of the same seed
    (required for c4_random)."""
    model = make_mlp(Xtr.shape[1], DROPOUT, seed)
    base = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    filt = SpectralConsensusFilter(model, base, loss_fn=None,
                                   mp_factor=MP_FACTOR)  # hard mode (F2)
    grad_fn = make_grad_fn(model)
    sampler = Sampler(era_codes, B, seed)
    g_sub = torch.Generator(device="cpu").manual_seed(seed + 777)  # C4 RNG
    uniform = torch.ones(B, device=DEV) / B
    gaf_thresh = GAF_SIGMA * np.sqrt(B)

    k_traj = np.zeros(steps, dtype=np.int32)
    ratio_traj = np.zeros(steps, dtype=np.float32)
    log = []
    model.train()
    t0 = time.time()
    for step in range(steps):
        idx = sampler.batch()
        p = {k: v for k, v in model.named_parameters()}
        bufs = {k: v for k, v in model.named_buffers()}
        psg, psl = grad_fn(p, bufs, Xtr[idx], ytr[idx])
        G = filt._flatten_grads(psg, B).detach()
        mean_g = G.mean(dim=0)

        if arm == "filter_on":
            consensus, diag = filt._spectral_filter(G, B)  # F11 verbatim
            k_t, ratio_t = diag["k"], diag["consensus_ratio"]
        elif arm == "c4_random":
            k_t = int(match_traj["k"][step])
            target_ratio = float(match_traj["ratio"][step])
            R = torch.randn(B, k_t, generator=g_sub).to(DEV)
            U_r, _ = torch.linalg.qr(R)
            w = U_r @ (U_r.T @ uniform)
            consensus = w @ G
            cn = consensus.norm()
            mg = mean_g.norm()
            if cn > 1e-12 and mg > 1e-12:
                consensus = consensus * (target_ratio * mg / cn)
                ratio_t = target_ratio
            else:
                ratio_t = 0.0
        elif arm == "gaf":
            sign_sum = torch.sign(G).sum(dim=0)
            mask = sign_sum.abs() >= gaf_thresh
            consensus = mean_g * mask
            k_t = int(mask.sum().item())  # here: # coords kept, not eigs
            mg = mean_g.norm().item()
            ratio_t = consensus.norm().item() / mg if mg > 1e-12 else 0.0
        k_traj[step] = k_t
        ratio_traj[step] = ratio_t

        if step % LOG_EVERY == 0:
            entry = {"step": step, "loss": float(psl.mean().item()),
                     "k": int(k_t), "ratio": float(ratio_t),
                     "cos_vs_mean": cos(consensus, mean_g)}
            if arm == "filter_on":
                entry["kept_energy_frac"] = kept_energy_fraction(G, MP_FACTOR)
            if arm == "gaf":
                entry["frac_coords_kept"] = float(k_t) / G.shape[1]
            log.append(entry)
            if step % 400 == 0:
                print(f"    step {step:4d} loss {entry['loss']:.5f} "
                      f"k {k_t:6d} ratio {ratio_t:.3f} "
                      f"cos {entry['cos_vs_mean']:.3f}", flush=True)

        filt._set_grads(consensus)
        base.step()
        base.zero_grad()
    if DEV == "cuda":
        torch.cuda.synchronize()
    traj = {"k": k_traj, "ratio": ratio_traj}
    return model, time.time() - t0, {"log": log}, traj


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-shard", required=True)
    ap.add_argument("--verdict-shard", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--smoke", action="store_true",
                    help="tiny local dry-run: few steps, small B, seeds=[0], "
                         "eval on the TRAIN shard (verdict stays blinded)")
    args = ap.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    steps, B, seeds = STEPS, BATCH, SEEDS
    if args.smoke:
        steps, B, seeds = 30, 256, [0]

    print(f"device: {DEV}, torch {torch.__version__}", flush=True)
    if DEV == "cuda":
        print(f"gpu: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"FROZEN: t07 lr={LR} wd={WD} do={DROPOUT}, steps={steps}, "
          f"B={B}, seeds={seeds}, mode=hard mp_factor={MP_FACTOR} (F2), "
          f"within-era batches (F6), GAF rule |sum sign|>= "
          f"{GAF_SIGMA}*sqrt(B), C4 rule = k+ratio traj matched to "
          f"filter_on same seed, random orthonormal sample-subspace",
          flush=True)

    proto = json.load(open(Path(args.ref) / "protocol.json"))
    train_allowed = set(proto["shard_train_eras"])
    verdict_allowed = set(proto["verdict_block"])
    assert not verdict_allowed & set(proto["tuning_block"])
    assert not verdict_allowed & train_allowed

    t_start = time.time()
    Xtr, ytr, era_tr, _ = load_shard(args.train_shard, train_allowed,
                                     "train shard")
    eval_name = "TRAIN-shard (smoke; verdict stays blinded)" if args.smoke \
        else "verdict shard"
    if args.smoke:
        Xev, yev, era_ev, eras_ev = load_shard(args.train_shard,
                                               train_allowed, eval_name)
    else:
        Xev, yev, era_ev, eras_ev = load_shard(args.verdict_shard,
                                               verdict_allowed, eval_name)
    yev_np = yev.cpu().numpy()

    # zero-predictor sanity (seeded, no information)
    rng = np.random.default_rng(12345)
    zp = rng.standard_normal(Xev.shape[0])
    pe_zero = per_era_metrics(zp, yev_np, era_ev, eras_ev)
    print(f"\nzero-predictor: mean spearman "
          f"{pe_zero['spearman'].mean():+.5f}, mean numerai_corr "
          f"{pe_zero['numerai_corr'].mean():+.5f} over {len(pe_zero)} eras",
          flush=True)

    per_era_sp = pd.DataFrame({"era": pe_zero["era"]})
    per_era_nc = per_era_sp.copy()
    per_era_sp["zero_pred"] = pe_zero["spearman"].to_numpy()
    per_era_nc["zero_pred"] = pe_zero["numerai_corr"].to_numpy()

    diagnostics = {}
    timings = {}
    for seed in seeds:
        traj_on = None
        for arm in ARM_ORDER:
            tag = f"{arm}_s{seed}"
            print(f"\n### {tag}", flush=True)
            if arm == "filter_off":
                model, dt, diag, traj = train_filter_off(
                    Xtr, ytr, era_tr, seed, steps, B)
            else:
                mt = traj_on if arm == "c4_random" else None
                assert arm != "c4_random" or mt is not None
                model, dt, diag, traj = train_filtered(
                    Xtr, ytr, era_tr, seed, steps, B, arm, match_traj=mt)
                if arm == "filter_on":
                    traj_on = traj
                    np.savez_compressed(outdir / f"traj_filter_on_s{seed}.npz",
                                        **traj)
                elif arm == "gaf":
                    np.savez_compressed(outdir / f"traj_gaf_s{seed}.npz",
                                        **traj)
            preds = predict(model, Xev)
            pe = per_era_metrics(preds, yev_np, era_ev, eras_ev)
            per_era_sp[tag] = pe["spearman"].to_numpy()
            per_era_nc[tag] = pe["numerai_corr"].to_numpy()
            diagnostics[tag] = diag
            timings[tag] = dt
            print(f"  {tag}: mean spearman {pe['spearman'].mean():+.5f}, "
                  f"mean numerai_corr {pe['numerai_corr'].mean():+.5f} "
                  f"(train {dt:.1f}s)", flush=True)
            del model
            if DEV == "cuda":
                torch.cuda.empty_cache()

    per_era_sp.to_csv(outdir / "per_era_spearman.csv", index=False)
    per_era_nc.to_csv(outdir / "per_era_numerai_corr.csv", index=False)
    with open(outdir / "diagnostics.json", "w") as f:
        json.dump({"frozen": {"lr": LR, "wd": WD, "dropout": DROPOUT,
                              "steps": steps, "batch": B, "seeds": seeds,
                              "mp_factor": MP_FACTOR, "mode": "hard",
                              "composition": "within-era",
                              "gaf_rule": f"|sum sign| >= {GAF_SIGMA}"
                                          f"*sqrt(B)",
                              "c4_rule": "random orthonormal sample-subspace"
                                         ", k(t) and consensus-ratio(t) "
                                         "matched to filter_on same seed"},
                   "timings_s": timings, "runs": diagnostics},
                  f, indent=2, default=float)
    print(f"\nTotal job time: {(time.time()-t_start)/60:.1f} min", flush=True)
    print("wrote per_era_numerai_corr.csv, per_era_spearman.csv, "
          "diagnostics.json, traj_*.npz", flush=True)


if __name__ == "__main__":
    main()
