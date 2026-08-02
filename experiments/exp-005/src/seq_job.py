#!/usr/bin/env python3
"""exp-005: sequence-arm architecture-consistency test (GRU on within-era
Numerai, same shards/machinery as exp-004's MLP verdict job).

ONE Slurm GPU job. ALL ARM CONFIGS ARE FROZEN IN THIS FILE, IN CODE, BEFORE
ANY EVALUATION TOUCHES THE VERDICT SHARD.

Frozen inputs (binding, reused verbatim from exp-004 — see plan.md):
  F1 : train = shard_train (eras 0425-0574); eval = shard_verdict
       (eras 0971-1225, 255 eras). Hard asserts against ref/protocol.json.
  F2 : threshold mode = hard, mp_factor = 2.0 (SpectralConsensusFilter,
       exact B x B MP-threshold filter from spectral_optimizer.py — F11).
  F6 : within-era batch composition, B = 1024.
  F13: within-era Numerai framing (same shards as exp-004) -> the result
       supports the "architecture consistency" claim; OHLCV NOT used.
  GRU config (t07 FALLBACK, pre-authorized): the tuning shard
       (/ephemeral/.../shard_tuning.parquet) was wiped after exp-003; the
       plan's binding fallback is to SKIP the GRU tuning sweep and reuse
       t07 (lr 1e-3, wd 1e-3, dropout 0.2, 2000 steps @ B=1024) — recorded
       as a limitation. The spectral arm gets NO tuning sweep either
       (same F12 affordability asymmetry as exp-004).

Architecture (the ONLY change vs exp-004): each row's 705 medium features
are reshaped to a T=15 x D=47 sequence (15*47=705, no padding) consumed by
the exp-002-verified hand-rolled 1-layer functional GRU (hidden 64, pure
tensor ops -> composes with torch.func vmap(grad)), dropout 0.2 on the
final hidden state before a sequence-to-one scalar regression head.
Rationale (plan.md): the claim under test is optimizer-effect consistency
across ARCHITECTURES on the same data/task — the sequence construction only
needs a genuinely recurrent compute graph over real Numerai features; it is
not claimed to be the natural architecture for the task.

Correctness guard (mandatory, exp-002 caveat): at job start, on the actual
cluster torch build, the B=64 vmap-vs-python-loop per-sample-gradient
assert is re-run for this GRU on real train-shard rows (eval mode so
dropout is inactive and both paths are deterministic). max abs diff > 1e-5
=> abort with exit code 2 — do not train on incorrect gradients.

Arms (same train shard, same base config, same seeds {0,1,2}, identical
data order per seed via the per-seed within-era Sampler — as exp-004):
  1. filter_off : plain AdamW t07.
  2. filter_on  : SpectralConsensusFilter (hard, mp_factor 2.0) on AdamW t07.
  3. c4_random  : C4 norm-matched random-subspace control, exp-004 matching
     rule verbatim: at step t, project the uniform weight vector onto a
     RANDOM k(t)-dim sample-subspace (random B x k(t) orthonormal basis,
     dedicated RNG seed+777), k(t) = same-seed filter_on run's realized k
     at step t; rescale so ||update||/||mean_grad|| equals that run's
     realized consensus_ratio at step t.
  GAF arm: CUT (plan: optional, cut first).

Per-sample gradients: torch.func vmap with randomness="different" (dropout).
The filtering math is the verbatim SpectralConsensusFilter._spectral_filter.

Diagnostics every LOG_EVERY=50 steps in filtered arms: k, consensus ratio,
kept-energy fraction (filter_on), C3 cosine (filtered vs mean gradient).
Full per-step k/ratio trajectories saved for the C4 matching + audit.

Evaluation: after training, each (arm, seed) model predicts ONCE on the
verdict shard; per-era numerai_corr and spearman saved to CSVs (same format
as exp-004). Plus the seeded zero-predictor sanity column.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import vmap, grad, grad_and_value, functional_call
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
from spectral_optimizer import SpectralConsensusFilter  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------- FROZEN CONFIG (do not modify after submission) ----------
STEPS = 2000
BATCH = 1024
LR = 1e-3          # t07 (fallback: tuning shard wiped, no GRU sweep)
WD = 1e-3          # t07
DROPOUT = 0.2      # t07
SEEDS = [0, 1, 2]
MP_FACTOR = 2.0    # F2
T_SEQ = 15         # 705 = 15 x 47, no padding
D_SEQ = 47
HIDDEN = 64
LOG_EVERY = 50
EVAL_CHUNK = 65536
MIN_ERA_ROWS = 100
GUARD_B = 64
GUARD_TOL = 1e-5
ARM_ORDER = ["filter_off", "filter_on", "c4_random"]
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


class GRURegressor(nn.Module):
    """1-layer GRU, hand-rolled cell (pure tensor ops -> vmap-compatible,
    verified in exp-002), sequence-to-one scalar regression head with
    dropout on the final hidden state. Input: flat (B, 705) Numerai rows,
    reshaped internally to (B, T=15, D=47)."""

    def __init__(self, t_seq, d_in, h, dropout):
        super().__init__()
        self.w_ih = nn.Parameter(torch.randn(3 * h, d_in) * (1.0 / d_in ** 0.5))
        self.w_hh = nn.Parameter(torch.randn(3 * h, h) * (1.0 / h ** 0.5))
        self.b_ih = nn.Parameter(torch.zeros(3 * h))
        self.b_hh = nn.Parameter(torch.zeros(3 * h))
        self.head_w = nn.Parameter(torch.randn(h) * (1.0 / h ** 0.5))
        self.head_b = nn.Parameter(torch.zeros(1))
        self.t_seq, self.d_in, self.h_dim = t_seq, d_in, h
        self.dropout = dropout

    def forward(self, x):
        # x: (B, 705) -> (B, T, D) -> (B,) scalar prediction
        bsz = x.shape[0]
        x = x.reshape(bsz, self.t_seq, self.d_in)
        h = x.new_zeros(bsz, self.h_dim)
        for t in range(self.t_seq):
            gi = x[:, t, :] @ self.w_ih.T + self.b_ih
            gh = h @ self.w_hh.T + self.b_hh
            i_r, i_z, i_n = gi.chunk(3, dim=1)
            h_r, h_z, h_n = gh.chunk(3, dim=1)
            r = torch.sigmoid(i_r + h_r)
            z = torch.sigmoid(i_z + h_z)
            n = torch.tanh(i_n + r * h_n)
            h = (1.0 - z) * n + z * h
        h = F.dropout(h, p=self.dropout, training=self.training)
        return h @ self.head_w + self.head_b


def make_gru(seed):
    torch.manual_seed(seed)
    return GRURegressor(T_SEQ, D_SEQ, HIDDEN, DROPOUT).to(DEV)


class Sampler:
    """Within-era batch sampler (F6), seeded per training seed so every arm
    at the same seed sees the IDENTICAL batch sequence (verbatim exp-004)."""

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
        return F.mse_loss(out, y.unsqueeze(0))
    return vmap(grad_and_value(compute_loss, has_aux=False),
                in_dims=(None, None, 0, 0), randomness="different")


def correctness_guard(Xtr, ytr):
    """Mandatory exp-002 caveat: re-run the B=64 vmap-vs-loop per-sample
    gradient assert for the GRU on THIS torch build, on real train rows.
    Eval mode (dropout inactive) so both paths are deterministic. Abort on
    failure — do not train on incorrect gradients."""
    print(f"\n-- correctness guard: vmap(grad) vs python-loop reference "
          f"(GRU, B={GUARD_B}, eval mode, real train rows) --", flush=True)
    model = make_gru(0)
    model.eval()
    x, y = Xtr[:GUARD_B], ytr[:GUARD_B]
    params = {k: v for k, v in model.named_parameters()}
    buffers = {k: v for k, v in model.named_buffers()}

    def loss_fn(p, b, xi, yi):
        out = functional_call(model, (p, b), (xi.unsqueeze(0),))
        return F.mse_loss(out, yi.unsqueeze(0))

    t0 = time.time()
    g_vmap = vmap(grad(loss_fn), in_dims=(None, None, 0, 0))(
        params, buffers, x, y)
    names = list(params.keys())
    g_loop = {k: [] for k in names}
    for i in range(GUARD_B):
        loss = F.mse_loss(model(x[i:i + 1]), y[i:i + 1])
        grads = torch.autograd.grad(loss, list(model.parameters()))
        for k, gg in zip(names, grads):
            g_loop[k].append(gg.detach().clone())
    g_loop = {k: torch.stack(v) for k, v in g_loop.items()}
    max_diff = 0.0
    for k in names:
        d = (g_vmap[k] - g_loop[k]).abs().max().item()
        ref = g_loop[k].abs().max().item()
        print(f"  {k:8s} shape={tuple(g_loop[k].shape)}  "
              f"max_abs_diff={d:.3e}  ref_max_abs={ref:.3e}", flush=True)
        max_diff = max(max_diff, d)
    ok = max_diff <= GUARD_TOL
    print(f"GUARD max abs diff (all params): {max_diff:.3e} -> "
          f"{'PASS' if ok else 'FAIL'} (tol {GUARD_TOL:.0e}) "
          f"[{time.time()-t0:.1f}s]", flush=True)
    if not ok:
        print("ABORT: correctness guard failed on this torch build; "
              "not training on unverified gradients.", flush=True)
        sys.exit(2)


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
    model = make_gru(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sampler = Sampler(era_codes, B, seed)
    model.train()
    t0 = time.time()
    for step in range(steps):
        idx = sampler.batch()
        opt.zero_grad(set_to_none=True)
        loss = F.mse_loss(model(Xtr[idx]), ytr[idx])
        loss.backward()
        opt.step()
        if step % 400 == 0:
            print(f"    step {step:4d} loss {loss.item():.5f}", flush=True)
    if DEV == "cuda":
        torch.cuda.synchronize()
    return model, time.time() - t0, {"log": []}, None


def train_filtered(Xtr, ytr, era_codes, seed, steps, B, arm,
                   match_traj=None):
    """arm in {filter_on, c4_random}. match_traj: dict with per-step 'k' and
    'ratio' arrays from the filter_on run of the same seed (c4_random)."""
    model = make_gru(seed)
    base = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    filt = SpectralConsensusFilter(model, base, loss_fn=None,
                                   mp_factor=MP_FACTOR)  # hard mode (F2)
    grad_fn = make_grad_fn(model)
    sampler = Sampler(era_codes, B, seed)
    g_sub = torch.Generator(device="cpu").manual_seed(seed + 777)  # C4 RNG
    uniform = torch.ones(B, device=DEV) / B

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
        k_traj[step] = k_t
        ratio_traj[step] = ratio_t

        if step % LOG_EVERY == 0:
            entry = {"step": step, "loss": float(psl.mean().item()),
                     "k": int(k_t), "ratio": float(ratio_t),
                     "cos_vs_mean": cos(consensus, mean_g)}
            if arm == "filter_on":
                entry["kept_energy_frac"] = kept_energy_fraction(G, MP_FACTOR)
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
        out.append(model(X[i:i + EVAL_CHUNK]).cpu())
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
                         "eval on the TRAIN shard")
    args = ap.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    steps, B, seeds = STEPS, BATCH, SEEDS
    if args.smoke:
        steps, B, seeds = 30, 256, [0]

    print(f"device: {DEV}, torch {torch.__version__}", flush=True)
    if DEV == "cuda":
        print(f"gpu: {torch.cuda.get_device_name(0)}", flush=True)
    n_params = sum(p.numel() for p in make_gru(0).parameters())
    print(f"FROZEN: GRU T={T_SEQ} D={D_SEQ} (15x47=705, no padding) "
          f"H={HIDDEN} params={n_params}; t07 lr={LR} wd={WD} do={DROPOUT} "
          f"(FALLBACK: tuning shard wiped -> plan-authorized reuse of t07, "
          f"NO GRU sweep; spectral arm has no sweep either, F12 asymmetry "
          f"recorded), steps={steps}, B={B}, seeds={seeds}, mode=hard "
          f"mp_factor={MP_FACTOR} (F2), within-era batches (F6), C4 rule = "
          f"k+ratio traj matched to filter_on same seed, random orthonormal "
          f"sample-subspace; GAF arm CUT (plan: cut first). F13: within-era "
          f"Numerai framing -> architecture-consistency claim.", flush=True)

    proto = json.load(open(Path(args.ref) / "protocol.json"))
    train_allowed = set(proto["shard_train_eras"])
    verdict_allowed = set(proto["verdict_block"])
    assert not verdict_allowed & set(proto["tuning_block"])
    assert not verdict_allowed & train_allowed

    t_start = time.time()
    Xtr, ytr, era_tr, _ = load_shard(args.train_shard, train_allowed,
                                     "train shard")
    eval_name = "TRAIN-shard (smoke)" if args.smoke else "verdict shard"
    if args.smoke:
        Xev, yev, era_ev, eras_ev = load_shard(args.train_shard,
                                               train_allowed, eval_name)
    else:
        Xev, yev, era_ev, eras_ev = load_shard(args.verdict_shard,
                                               verdict_allowed, eval_name)
    yev_np = yev.cpu().numpy()

    # mandatory correctness guard on this torch build BEFORE any training
    correctness_guard(Xtr, ytr)

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
        json.dump({"frozen": {"arch": f"GRU T={T_SEQ} D={D_SEQ} H={HIDDEN} "
                                      f"params={n_params}",
                              "lr": LR, "wd": WD, "dropout": DROPOUT,
                              "steps": steps, "batch": B, "seeds": seeds,
                              "mp_factor": MP_FACTOR, "mode": "hard",
                              "composition": "within-era",
                              "config_provenance": "t07 fallback (tuning "
                                  "shard wiped; plan-authorized; no GRU "
                                  "sweep, no spectral sweep)",
                              "c4_rule": "random orthonormal sample-subspace"
                                         ", k(t) and consensus-ratio(t) "
                                         "matched to filter_on same seed",
                              "gaf_arm": "cut (plan: cut first)"},
                   "timings_s": timings, "runs": diagnostics},
                  f, indent=2, default=float)
    print(f"\nTotal job time: {(time.time()-t_start)/60:.1f} min", flush=True)
    print("wrote per_era_numerai_corr.csv, per_era_spearman.csv, "
          "diagnostics.json, traj_*.npz", flush=True)


if __name__ == "__main__":
    main()
