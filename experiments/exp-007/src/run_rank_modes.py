#!/usr/bin/env python3
"""Exact parameter-space high-rank and top-eigenspace ablation trajectories."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
RUN_ROOT = ROOT.parent.parent
sys.path[:0] = [str(ROOT / "src"), str(RUN_ROOT / "experiments/exp-006/src"),
                "/home/titus/pyg/optimizers"]
from run_overfit import atomic_json, evaluate, monitor_indices
from spectral_filter_fixed import StableSpectralGradientFilter

ARCH = [2376, 256, 64, 1]
SHARD = Path("/media/titus/big/tmp/numerai-v5-full-shard")


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2376, 256), nn.ReLU(),
                                 nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


class ComplementFilter(StableSpectralGradientFilter):
    def __init__(self, *args, renormalize=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.renormalize_complement = bool(renormalize)

    def _project_gradient(self, g):
        if self.V is None:
            return g
        V = self.V if self.proj_k is None else self.V[:, :self.proj_k]
        complement = g - V @ (V.T @ g)
        if self.renormalize_complement:
            complement = complement * (g.norm() / complement.norm().clamp_min(1e-12))
        return complement


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["adamw", "top", "remove", "remove-renorm"], required=True)
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20260805)
    ap.add_argument("--eval-every", type=int, default=250)
    args = ap.parse_args()
    if args.mode != "adamw" and args.rank < 1:
        ap.error("filtered modes require --rank >= 1")

    X = np.load(SHARD / "X_u8.npy", mmap_mode="r")
    Y = np.load(SHARD / "y_f32.npy", mmap_mode="r")
    E = np.load(SHARD / "era_i16.npy", mmap_mode="r")
    eras = np.asarray(E)
    train = np.where(eras <= 791)[0]
    valid = np.where((eras >= 796) & (eras <= 891))[0]
    train_monitor = monitor_indices(train, eras, 128, 7711)
    valid_monitor = monitor_indices(valid, eras, 512, 7712)

    np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda")
    model = MLP().to(device)
    parameter_count = sum(p.numel() for p in model.parameters())
    assert parameter_count == 625_025
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.0)
    filt = None
    if args.mode != "adamw":
        cls = StableSpectralGradientFilter if args.mode == "top" else ComplementFilter
        extra = {} if args.mode == "top" else {"renormalize": args.mode == "remove-renorm"}
        filt = cls(model, optimizer, rank=args.rank, decay=0.99, warmup=100,
                   normalize="none", weighting="hard", alpha=1.0,
                   soft_residual=True, adaptive="none", relative_eig_tol=1e-8,
                   stabilize_every=100, **extra)
    rng = np.random.default_rng(args.seed + 1)
    curve, recent = [], []
    label = f"{args.mode}-r{args.rank}"
    output = ROOT / "out" / f"rankmode-{label}-seed{args.seed}.json"
    started = time.time()
    checkpoints = {1, 25, 50, 100, args.steps}
    checkpoints.update(range(args.eval_every, args.steps + 1, args.eval_every))
    for step in range(1, args.steps + 1):
        j = rng.choice(train, 1024, replace=True)
        x = torch.from_numpy(np.asarray(X[j], np.float32) / 4).to(device)
        y = torch.from_numpy(np.asarray(Y[j], np.float32) - 0.5).to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = F.mse_loss(model(x), y); loss.backward()
        diag = filt.filter_grad() if filt else None
        optimizer.step()
        recent.append(float(loss.detach()))
        if len(recent) > 250: recent.pop(0)
        if step in checkpoints:
            row = {"step": step, "sample_equivalent_epochs": step * 1024 / len(train),
                   "batch_mse_250": float(np.mean(recent)),
                   "train": evaluate(model, X, Y, E, train_monitor, device),
                   "valid": evaluate(model, X, Y, E, valid_monitor, device),
                   "elapsed_seconds": time.time() - started}
            if diag is not None: row["filter"] = diag
            curve.append(row)
            atomic_json(output, {"mode": args.mode, "rank": args.rank,
                "seed": args.seed, "architecture": ARCH,
                "parameter_count": parameter_count, "train_rows": int(len(train)),
                "valid_rows": int(len(valid)), "test_touched": False,
                "exact_full_parameter_space_basis": True,
                "paired_initialization_and_batch_stream": True, "curve": curve})
            print(json.dumps(row), flush=True)
    print(f"DONE {output}", flush=True)


if __name__ == "__main__":
    main()
