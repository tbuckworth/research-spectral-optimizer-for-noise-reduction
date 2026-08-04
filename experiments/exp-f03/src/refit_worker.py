#!/usr/bin/env python3
"""Worker for the refit checkpoint roundtrip (exp-f03 sub-check 2).

Three modes, each run as its OWN process by roundtrip_compare.py:

  reference  train N+M steps uninterrupted; record the flat parameter vector
             and loss after every one of steps N+1..N+M; save filter V/S at
             the end.
  phase1     identical setup, train N steps, save checkpoint (model + AdamW +
             SpectralGradientFilter state + all RNG streams).
  phase2     FRESH process: build model/opt/filter with a DIFFERENT init seed
             (proving restore overwrites everything), load the checkpoint,
             train M further steps, record the same trajectory.

Bit-continuation criterion: phase2's 10-step trajectory must be bitwise
identical (fp64) to the reference run's steps N+1..N+M.

Training loop mirrors harness.run_refit: fp64, CPU, single thread, AdamW +
SpectralGradientFilter(rank=8, warmup=5), data order from a dedicated
torch.Generator (seed 777) so arm pairing survives -- the filter itself
consumes no RNG, but all four streams are checkpointed regardless.
"""
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from checkpoint import save_checkpoint, load_checkpoint
from spectral_filter import SpectralGradientFilter

DEV = "cpu"
torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)

N_STEPS = 25          # checkpoint point
M_STEPS = 10          # continuation window under comparison
BATCH = 64
N_ROWS, N_FEAT = 4096, 20
INIT_SEED = 777
DATA_SEED = 321


def make_data():
    g = torch.Generator().manual_seed(DATA_SEED)
    X = torch.randn(N_ROWS, N_FEAT, generator=g)
    w = torch.randn(N_FEAT, generator=g)
    y = X @ w / N_FEAT ** 0.5 + 0.5 * torch.randn(N_ROWS, generator=g)
    return X.to(DEV), y.to(DEV)


def build(seed):
    torch.manual_seed(seed)
    model = nn.Sequential(nn.Linear(N_FEAT, 16), nn.ReLU(),
                          nn.Linear(16, 1)).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    filt = SpectralGradientFilter(model, opt, rank=8, warmup=5)
    data_g = torch.Generator(device="cpu").manual_seed(INIT_SEED)
    return model, opt, filt, data_g


def flat_params(model):
    return torch.cat([p.detach().reshape(-1) for p in model.parameters()]).clone()


def train_steps(model, opt, filt, data_g, X, y, n_steps, record_from=None):
    traj_params, traj_losses = [], []
    n = X.shape[0]
    for step in range(n_steps):
        idx = torch.randint(0, n, (BATCH,), generator=data_g)
        opt.zero_grad(set_to_none=True)
        loss = F.mse_loss(model(X[idx]).squeeze(-1), y[idx])
        loss.backward()
        filt.filter_grad()
        opt.step()
        if record_from is not None and step >= record_from:
            traj_params.append(flat_params(model))
            traj_losses.append(loss.item())
    return traj_params, traj_losses


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["reference", "phase1", "phase2"],
                    required=True)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    X, y = make_data()

    if args.mode == "reference":
        model, opt, filt, data_g = build(INIT_SEED)
        traj, losses = train_steps(model, opt, filt, data_g, X, y,
                                   N_STEPS + M_STEPS, record_from=N_STEPS)
        torch.save({"params": traj, "losses": losses,
                    "V": filt.V.clone(), "S": filt.S.clone(),
                    "step_count": filt.step_count,
                    "grad_mean": filt.grad_mean.clone()},
                   out / "ref_traj.pt")
        print(f"[reference] {N_STEPS + M_STEPS} steps done; recorded last "
              f"{M_STEPS}; final loss {losses[-1]:.17g}", flush=True)

    elif args.mode == "phase1":
        model, opt, filt, data_g = build(INIT_SEED)
        train_steps(model, opt, filt, data_g, X, y, N_STEPS)
        save_checkpoint(out / "roundtrip_ckpt.pt", model=model, optimizer=opt,
                        filt=filt, step=N_STEPS, data_generator=data_g)
        print(f"[phase1] trained {N_STEPS} steps, checkpoint saved "
              f"(filter step_count={filt.step_count}, "
              f"basis k={filt.V.shape[1]})", flush=True)

    else:  # phase2 -- fresh process, different init seed, restore, continue
        model, opt, filt, data_g = build(999)      # deliberately wrong seed
        step, _ = load_checkpoint(out / "roundtrip_ckpt.pt", model=model,
                                  optimizer=opt, filt=filt,
                                  data_generator=data_g)
        assert step == N_STEPS
        traj, losses = train_steps(model, opt, filt, data_g, X, y, M_STEPS,
                                   record_from=0)
        torch.save({"params": traj, "losses": losses,
                    "V": filt.V.clone(), "S": filt.S.clone(),
                    "step_count": filt.step_count,
                    "grad_mean": filt.grad_mean.clone()},
                   out / "resume_traj.pt")
        print(f"[phase2] restored at step {step} (filter "
              f"step_count={filt.step_count}), ran {M_STEPS} more; final "
              f"loss {losses[-1]:.17g}", flush=True)


if __name__ == "__main__":
    main()
