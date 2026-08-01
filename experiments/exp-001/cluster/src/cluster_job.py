#!/usr/bin/env python3
"""Sub-components #7 (vmap throughput) + #1 (spectral engagement, headline).
Runs as ONE Slurm --qos=debug GPU job on the MATS cluster.

#1 engagement grid (400 steps each, log every 10):
    threshold mode {hard, soft, variance} x B {256, 1024}
    x batch composition {within-era, mixed-era}            (F6)
  plus C2 permuted-target null runs (hard mode, both B, both compositions).
  Diagnostics per logged step: k (eigendirections kept), consensus_ratio
  (fraction of mean-grad norm passed), top-10 eigenvalues, C3 cosines
  (grad-level filtered-vs-mean and AdamW-update-level filtered-vs-unfiltered),
  full eigenspectrum snapshot every 100 steps, C5 era-identity probes every
  50 steps (within-era configs).

#7 throughput: 200 timed steps at B {256, 1024}, filter-on (hard) and
  filter-off (plain batch backward + AdamW), mixed-era composition.

Data: Numerai v5 shard (150 recent train eras, 705 medium features,
<=2560 rows/era) read from /ephemeral (copied from NFS if needed).
Model: MLP 705-256-64-1, per-sample MSE on centered target, AdamW lr 1e-3.
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

import sys
sys.path.insert(0, str(Path(__file__).parent))
from spectral_optimizer import SpectralConsensusFilter  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
LOG_EVERY = 10
SPEC_EVERY = 100
PROBE_EVERY = 50
STEPS = 400
TIMING_STEPS = 200
TIMING_WARMUP = 20
FULL_RUN_STEPS = 2000  # definition of one "full training run" for projection
LR = 1e-3


def load_shard(path):
    df = pd.read_parquet(path)
    feat_cols = [c for c in df.columns if c.startswith("feature")]
    X = torch.tensor(df[feat_cols].to_numpy(dtype=np.float32),
                     device=DEV)
    X = (X - 2.0) / 2.0  # int8 levels 0..4 -> [-1, 1]
    y = torch.tensor(df["target"].to_numpy(dtype=np.float32),
                     device=DEV) - 0.5
    era_codes, era_uniques = pd.factorize(df["era"], sort=True)
    era = torch.tensor(era_codes, device=DEV)
    print(f"shard: {X.shape[0]} rows, {X.shape[1]} features, "
          f"{len(era_uniques)} eras ({era_uniques[0]}..{era_uniques[-1]})",
          flush=True)
    return X, y, era, list(map(str, era_uniques))


def make_mlp(d, seed=0):
    torch.manual_seed(seed)
    m = nn.Sequential(nn.Linear(d, 256), nn.ReLU(),
                      nn.Linear(256, 64), nn.ReLU(),
                      nn.Linear(64, 1)).to(DEV)
    return m


def per_sample_mse(out, y):
    return F.mse_loss(out.squeeze(-1), y)


def adamw_update_dir(opt, params, flat_grad):
    """Hypothetical AdamW update direction for a candidate flat gradient,
    given the optimizer's CURRENT state (no state mutation). wd=0."""
    group = opt.param_groups[0]
    b1, b2 = group["betas"]
    eps = group["eps"]
    out = []
    offset = 0
    for p in params:
        n = p.numel()
        g = flat_grad[offset:offset + n].reshape(p.shape)
        offset += n
        st = opt.state.get(p, {})
        if "exp_avg" in st:
            m, v, t = st["exp_avg"], st["exp_avg_sq"], st["step"]
            t = int(t.item()) if torch.is_tensor(t) else int(t)
        else:
            m = torch.zeros_like(p)
            v = torch.zeros_like(p)
            t = 0
        m2 = b1 * m + (1 - b1) * g
        v2 = b2 * v + (1 - b2) * g * g
        mhat = m2 / (1 - b1 ** (t + 1))
        vhat = v2 / (1 - b2 ** (t + 1))
        out.append((mhat / (vhat.sqrt() + eps)).reshape(-1))
    return torch.cat(out)


class Sampler:
    """Batch sampler for the two F6 compositions."""

    def __init__(self, era, B, composition, seed):
        self.B = B
        self.composition = composition
        self.g = torch.Generator(device="cpu").manual_seed(seed)
        era_cpu = era.cpu()
        self.n = era_cpu.shape[0]
        self.era_ids = era_cpu.unique()
        self.era_rows = {int(e): (era_cpu == e).nonzero().squeeze(-1)
                         for e in self.era_ids}
        # eras with enough rows for batch + heldout probe
        self.big_eras = [e for e, r in self.era_rows.items()
                         if len(r) >= B + 256]

    def batch(self):
        """Returns (idx, era_id or None, heldout_same_idx or None)."""
        if self.composition == "mixed":
            idx = torch.randint(0, self.n, (self.B,), generator=self.g)
            return idx.to(DEV), None, None
        e = self.big_eras[int(torch.randint(0, len(self.big_eras), (1,),
                                            generator=self.g))]
        rows = self.era_rows[e]
        perm = torch.randperm(len(rows), generator=self.g)
        idx = rows[perm[:self.B]]
        held = rows[perm[self.B:self.B + 256]]
        return idx.to(DEV), e, held.to(DEV)

    def other_era_batch(self, exclude_era, n=256):
        choices = [e for e in self.big_eras if e != exclude_era]
        e = choices[int(torch.randint(0, len(choices), (1,),
                                      generator=self.g))]
        rows = self.era_rows[e]
        perm = torch.randperm(len(rows), generator=self.g)
        return rows[perm[:n]].to(DEV), e


def mean_grad_of(model, X, y, idx):
    """Batch-mean gradient (flat) via standard backward; leaves params clean."""
    model.zero_grad(set_to_none=True)
    loss = F.mse_loss(model(X[idx]).squeeze(-1), y[idx])
    loss.backward()
    g = torch.cat([p.grad.reshape(-1) for p in model.parameters()])
    model.zero_grad(set_to_none=True)
    return g


def cos(a, b):
    na, nb = a.norm(), b.norm()
    if na < 1e-12 or nb < 1e-12:
        return float("nan")
    return float((a @ b / (na * nb)).item())


def run_engagement(X, y, era, mode, B, composition, permuted, seed=0,
                   steps=None):
    steps = STEPS if steps is None else steps
    tag = (f"mode={mode}_B={B}_comp={composition}"
           + ("_permuted" if permuted else ""))
    print(f"\n### engagement run {tag}", flush=True)
    yy = y
    if permuted:
        g = torch.Generator(device="cpu").manual_seed(seed + 999)
        yy = y[torch.randperm(len(y), generator=g).to(DEV)]
    model = make_mlp(X.shape[1], seed=seed)
    base = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.0)
    kw = {}
    if mode == "soft":
        kw["soft"] = True
    elif mode == "variance":
        kw["variance_threshold"] = 0.9
    filt = SpectralConsensusFilter(model, base, loss_fn=per_sample_mse, **kw)
    sampler = Sampler(era, B, composition, seed)
    params = list(model.parameters())

    log = []
    spectra = {}
    t0 = time.time()
    for step in range(steps):
        idx, cur_era, held = sampler.batch()
        p = {k: v for k, v in model.named_parameters()}
        bufs = {k: v for k, v in model.named_buffers()}
        psg, psl = filt._grad_and_value_fn(p, bufs, X[idx], yy[idx])
        G = filt._flatten_grads(psg, B).detach()
        consensus, diag = filt._spectral_filter(G, B)
        mean_g = G.mean(dim=0)

        entry = None
        if step % LOG_EVERY == 0:
            entry = {
                "step": step,
                "loss": float(psl.mean().item()),
                "k": diag["k"],
                "consensus_ratio": diag["consensus_ratio"],
                "top_eigs": diag["eigenvalues"],
                "cos_grad_filtered_vs_mean": cos(consensus, mean_g),
                "cos_update_filtered_vs_unfiltered": cos(
                    adamw_update_dir(base, params, consensus),
                    adamw_update_dir(base, params, mean_g)),
            }
        if step % SPEC_EVERY == 0:
            norms = G.norm(dim=1, keepdim=True).clamp(min=1e-12)
            Gn = G / norms
            ev = torch.linalg.eigvalsh(Gn @ Gn.T).flip(0).clamp(min=0)
            spectra[str(step)] = ev.detach().cpu().numpy()
        if (entry is not None and composition == "within"
                and step % PROBE_EVERY == 0):
            oidx, oera = sampler.other_era_batch(cur_era)
            gh = mean_grad_of(model, X, yy, held)
            go = mean_grad_of(model, X, yy, oidx)
            entry.update({
                "c5_cos_filtered_sameera": cos(consensus, gh),
                "c5_cos_filtered_othererra": cos(consensus, go),
                "c5_cos_mean_sameera": cos(mean_g, gh),
                "c5_cos_mean_otherera": cos(mean_g, go),
            })
        if entry is not None:
            log.append(entry)
            if step % 100 == 0:
                print(f"  step {step:3d} loss {entry['loss']:.5f} "
                      f"k {entry['k']:4d} ratio "
                      f"{entry['consensus_ratio']:.3f} "
                      f"cosF {entry['cos_grad_filtered_vs_mean']:.3f}",
                      flush=True)

        filt._set_grads(consensus)
        base.step()
        base.zero_grad()
    dt = time.time() - t0
    print(f"  {tag}: {steps} steps in {dt:.1f}s "
          f"({dt/steps*1000:.1f} ms/step)", flush=True)
    return {"tag": tag, "mode": mode, "B": B, "composition": composition,
            "permuted": permuted, "steps": steps, "seconds": dt,
            "ms_per_step": dt / steps * 1000, "log": log}, spectra


def run_timing(X, y, era, B, filter_on, seed=0):
    tag = f"timing_B={B}_{'filteron' if filter_on else 'filteroff'}"
    model = make_mlp(X.shape[1], seed=seed)
    base = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.0)
    sampler = Sampler(era, B, "mixed", seed)
    filt = (SpectralConsensusFilter(model, base, loss_fn=per_sample_mse)
            if filter_on else None)
    times = []
    for step in range(TIMING_STEPS):
        idx, _, _ = sampler.batch()
        torch.cuda.synchronize() if DEV == "cuda" else None
        t0 = time.time()
        if filter_on:
            filt.step(X[idx], y[idx])
        else:
            base.zero_grad()
            loss = F.mse_loss(model(X[idx]).squeeze(-1), y[idx])
            loss.backward()
            base.step()
        torch.cuda.synchronize() if DEV == "cuda" else None
        times.append(time.time() - t0)
    ms = float(np.mean(times[TIMING_WARMUP:]) * 1000)
    proj_min = ms * FULL_RUN_STEPS / 1000 / 60
    print(f"  {tag}: {ms:.1f} ms/step; projected {FULL_RUN_STEPS}-step "
          f"full run: {proj_min:.1f} min", flush=True)
    return {"tag": tag, "B": B, "filter_on": filter_on, "ms_per_step": ms,
            "projected_full_run_min": proj_min,
            "full_run_steps": FULL_RUN_STEPS}


def main():
    global STEPS, TIMING_STEPS, LOG_EVERY, SPEC_EVERY, PROBE_EVERY
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--quick", action="store_true",
                    help="tiny CPU dry-run for local pipeline validation")
    args = ap.parse_args()
    b_list = (256, 1024)
    if args.quick:
        STEPS, TIMING_STEPS = 25, 25
        LOG_EVERY, SPEC_EVERY, PROBE_EVERY = 5, 10, 10
        b_list = (64, 128)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"device: {DEV}, torch {torch.__version__}", flush=True)
    if DEV == "cuda":
        print(f"gpu: {torch.cuda.get_device_name(0)}", flush=True)
    t_start = time.time()
    X, y, era, era_names = load_shard(args.shard)
    n_params = sum(p.numel() for p in make_mlp(X.shape[1]).parameters())
    print(f"MLP params: {n_params}", flush=True)

    # ---------- #7 throughput ----------
    print("\n===== #7 vmap throughput =====", flush=True)
    timing = [run_timing(X, y, era, B, fo)
              for B in b_list for fo in (True, False)]
    with open(outdir / "timing.json", "w") as f:
        json.dump(timing, f, indent=2)

    # ---------- #1 engagement grid ----------
    print("\n===== #1 spectral engagement =====", flush=True)
    runs = []
    all_spectra = {}
    for mode in ("hard", "soft", "variance"):
        for B in b_list:
            for comp in ("within", "mixed"):
                r, spec = run_engagement(X, y, era, mode, B, comp,
                                         permuted=False)
                runs.append(r)
                for s, ev in spec.items():
                    all_spectra[f"{r['tag']}__step{s}"] = ev
    # C2 permuted-target nulls (hard mode)
    for B in b_list:
        for comp in ("within", "mixed"):
            r, spec = run_engagement(X, y, era, "hard", B, comp,
                                     permuted=True)
            runs.append(r)
            for s, ev in spec.items():
                all_spectra[f"{r['tag']}__step{s}"] = ev

    with open(outdir / "engagement.json", "w") as f:
        json.dump(runs, f, indent=2)
    np.savez_compressed(outdir / "spectra.npz", **all_spectra)
    print(f"\nTotal job time: {(time.time()-t_start)/60:.1f} min", flush=True)
    print("wrote timing.json, engagement.json, spectra.npz", flush=True)


if __name__ == "__main__":
    main()
