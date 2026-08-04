#!/usr/bin/env python3
"""B2-resumable sweep + refit harness (exp-f03 deliverable pattern).

Adapted from the parent run's exp-003 src/sweep.py. That harness kept its
trial results in an in-memory `trials` list and wrote sweep_results.json once
at the end of main() -- so a Slurm --time SIGKILL loses every completed trial.
This harness keeps the parent's structure (config grid -> train_one -> record,
one seed per trial) and changes exactly the persistence discipline:

  SWEEP phase (`--phase sweep`):
    * each trial's record is appended to out/trials.jsonl via TrialStore
      (write + flush + fsync) the moment the trial completes;
    * ON START the harness scans trials.jsonl, SKIPS completed trial_ids, and
      runs only the remainder -- the resume-on-restart discipline;
    * a marker file out/state/trial_<id>.started is touched when a trial
      begins (used by the kill-test supervisor for deterministic kill timing;
      in the fold jobs it doubles as an in-flight indicator).

  REFIT phase (`--phase refit`):
    * trains a model with SpectralGradientFilter active, checkpointing
      model + AdamW + filter + all RNG streams every --ckpt-every steps via
      checkpoint.save_checkpoint (atomic tmp+rename);
    * ON START, if out/refit_ckpt.pt exists it restores and continues from
      the checkpointed step; on completion writes out/refit_done.json.

  `--phase all` runs sweep then refit -- the exact pattern the fold jobs
  (EU-2/3/4) will use on every (re)start: scan persisted trials, skip
  completed, resume any checkpointed refit.

Synthetic task: tiny fp64 MLP regression on fixed-seed synthetic data.
CPU only (this component is authorized zero-GPU; device is hardcoded).
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from trial_store import TrialStore
from checkpoint import save_checkpoint, load_checkpoint
from spectral_filter import SpectralGradientFilter

DEV = "cpu"                    # zero-GPU component: never cuda
torch.set_num_threads(1)       # determinism across processes
torch.set_default_dtype(torch.float64)   # fp64 (bit-continuation criterion)

# Synthetic stand-ins for the parent's grid (6 fast trials)
LRS = [3e-4, 1e-3, 3e-3]
WDS = [0.0, 1e-3]
SWEEP_SEED = 0
TRIAL_STEPS = 4000             # ~2.5 s/trial on one CPU thread (long enough
                               # for the supervisor's kill to land mid-trial)
BATCH = 64
N_ROWS, N_FEAT = 4096, 20

REFIT_STEPS = 60
REFIT_WARMUP = 5               # filter active well before the end


def make_data(seed=123):
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(N_ROWS, N_FEAT, generator=g)
    w = torch.randn(N_FEAT, generator=g)
    y = X @ w / N_FEAT ** 0.5 + 0.5 * torch.randn(N_ROWS, generator=g)
    return X.to(DEV), y.to(DEV)


def make_mlp(d, seed):
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(d, 16), nn.ReLU(),
                         nn.Linear(16, 1)).to(DEV)


def train_one(X, y, lr, wd, seed, steps=TRIAL_STEPS):
    """Parent sweep.py's train_one, reduced to the synthetic task."""
    model = make_mlp(X.shape[1], seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    g = torch.Generator(device="cpu").manual_seed(seed)
    n = X.shape[0]
    model.train()
    t0 = time.time()
    loss = None
    for step in range(steps):
        idx = torch.randint(0, n, (BATCH,), generator=g)
        opt.zero_grad(set_to_none=True)
        loss = F.mse_loss(model(X[idx]).squeeze(-1), y[idx])
        loss.backward()
        opt.step()
    return model, float(loss.item()), time.time() - t0


def run_sweep(outdir):
    store = TrialStore(outdir / "trials.jsonl")
    state_dir = outdir / "state"
    state_dir.mkdir(parents=True, exist_ok=True)

    done = store.completed_ids()
    print(f"[sweep] resume scan: {len(done)} completed trial(s) on disk: "
          f"{sorted(done)}", flush=True)

    X, y = make_data()
    grid = [(lr, wd) for lr in LRS for wd in WDS]
    ran = []
    for tid0, (lr, wd) in enumerate(grid):
        tid = tid0 + 1                       # trial ids 1..6
        tag = f"t{tid:02d}_lr{lr:g}_wd{wd:g}"
        if tid in done:
            print(f"[sweep] SKIP trial {tid} ({tag}): already completed",
                  flush=True)
            continue
        print(f"[sweep] RUN  trial {tid} ({tag})", flush=True)
        (state_dir / f"trial_{tid}.started").touch()
        model, final_loss, secs = train_one(X, y, lr, wd, SWEEP_SEED)
        store.append({"trial_id": tid, "tag": tag, "lr": lr, "wd": wd,
                      "seed": SWEEP_SEED, "final_loss": final_loss,
                      "train_seconds": secs,
                      "completed_at": time.time()})   # durable BEFORE next trial
        ran.append(tid)
        del model
    n = store.assert_no_duplicates()
    print(f"[sweep] done: ran {ran}, {n} unique records on disk", flush=True)
    return ran


def run_refit(outdir, ckpt_every):
    """Refit with SpectralGradientFilter, resumable from atomic checkpoints."""
    ckpt_path = outdir / "refit_ckpt.pt"
    done_path = outdir / "refit_done.json"
    if done_path.exists():
        print("[refit] already complete, nothing to do", flush=True)
        return

    torch.manual_seed(777)
    np.random.seed(777)
    X, y = make_data(seed=321)
    model = make_mlp(N_FEAT, seed=777)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    filt = SpectralGradientFilter(model, opt, rank=8, warmup=REFIT_WARMUP)
    data_g = torch.Generator(device="cpu").manual_seed(777)

    start = 0
    if ckpt_path.exists():
        start, extra = load_checkpoint(ckpt_path, model=model, optimizer=opt,
                                       filt=filt, data_generator=data_g)
        print(f"[refit] RESUMED from checkpoint at step {start} "
              f"(filter step_count={filt.step_count})", flush=True)
    else:
        print("[refit] fresh start", flush=True)

    n = X.shape[0]
    for step in range(start, REFIT_STEPS):
        idx = torch.randint(0, n, (BATCH,), generator=data_g)
        opt.zero_grad(set_to_none=True)
        loss = F.mse_loss(model(X[idx]).squeeze(-1), y[idx])
        loss.backward()
        filt.filter_grad()
        opt.step()
        done_step = step + 1
        if done_step % ckpt_every == 0 and done_step < REFIT_STEPS:
            save_checkpoint(ckpt_path, model=model, optimizer=opt, filt=filt,
                            step=done_step, data_generator=data_g,
                            extra={"loss": float(loss.item())})
            print(f"[refit] checkpoint @ step {done_step} "
                  f"loss {loss.item():.6f}", flush=True)
        time.sleep(0.02)   # slow the loop so a kill can land mid-refit

    with open(done_path, "w") as f:
        json.dump({"steps": REFIT_STEPS, "resumed_from": start,
                   "final_loss": float(loss.item())}, f)
        f.flush()
        os.fsync(f.fileno())
    print(f"[refit] COMPLETE at step {REFIT_STEPS} (resumed_from={start}) "
          f"final loss {loss.item():.6f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--phase", choices=["sweep", "refit", "all"],
                    default="all")
    ap.add_argument("--ckpt-every", type=int, default=10)
    args = ap.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"[harness] pid={os.getpid()} phase={args.phase} torch "
          f"{torch.__version__} dev={DEV}", flush=True)
    if args.phase in ("sweep", "all"):
        run_sweep(outdir)
    if args.phase in ("refit", "all"):
        run_refit(outdir, args.ckpt_every)
    print("[harness] EXIT ok", flush=True)


if __name__ == "__main__":
    main()
