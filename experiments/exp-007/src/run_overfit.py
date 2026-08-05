#!/usr/bin/env python3
"""Paired overparameterized Numerai trajectories without touching TEST."""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
RUN_ROOT = ROOT.parent.parent
sys.path[:0] = [
    str(RUN_ROOT / "experiments/exp-006/src"),
    "/home/titus/pyg/optimizers",
]
from spectral_filter_fixed import StableSpectralGradientFilter
from block_spectral import BlockSpectralGradientFilter

ARCH = [2376, 2048, 1024, 512, 1]
SHARD = Path(os.environ.get("NUMERAI_SHARD", "/media/titus/big/tmp/numerai-v5-full-shard"))


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        layers = []
        for a, b in zip(ARCH[:-2], ARCH[1:-1]):
            layers.extend([nn.Linear(a, b), nn.ReLU()])
        layers.append(nn.Linear(ARCH[-2], ARCH[-1]))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def numerai_corr(pred, target):
    pred = np.asarray(pred, np.float64)
    target = np.asarray(target, np.float64)
    order = np.argsort(pred, kind="mergesort")
    ranks = np.empty(len(pred), np.float64)
    ranks[order] = np.arange(len(pred))
    ranks = (ranks + 0.5) / len(pred) - 0.5
    pred_t = np.sign(ranks) * np.abs(ranks) ** 1.5
    target_t = np.sign(target - 0.5) * np.abs(target - 0.5) ** 1.5
    if pred_t.std() == 0 or target_t.std() == 0:
        return 0.0
    return float(np.corrcoef(pred_t, target_t)[0, 1])


def evaluate(model, X, Y, E, idx, device):
    model.eval()
    predictions = []
    with torch.no_grad():
        for start in range(0, len(idx), 4096):
            j = idx[start:start + 4096]
            x = torch.from_numpy(np.asarray(X[j], np.float32) / 4).to(device)
            predictions.append(model(x).cpu().numpy())
    pred = np.concatenate(predictions)
    y = np.asarray(Y[idx], np.float32)
    era = np.asarray(E[idx])
    mse = float(np.mean((pred - (y - 0.5)) ** 2))
    by_era = [numerai_corr(pred[era == e], y[era == e]) for e in np.unique(era)]
    model.train()
    return {"mse": mse, "mean_per_era_corr": float(np.mean(by_era)),
            "corr_std": float(np.std(by_era)), "n_rows": int(len(idx)),
            "n_eras": int(len(by_era))}


def monitor_indices(indices, eras, per_era, seed):
    rng = np.random.default_rng(seed)
    out = []
    for era in np.unique(eras[indices]):
        candidates = indices[eras[indices] == era]
        out.append(rng.choice(candidates, min(per_era, len(candidates)), replace=False))
    return np.sort(np.concatenate(out))


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as handle:
        json.dump(value, handle, indent=2)
        handle.flush(); os.fsync(handle.fileno())
    os.replace(tmp, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=["adamw", "spectral"], required=True)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--relative-eig-tol", type=float, default=1e-8)
    parser.add_argument("--blockwise", action="store_true")
    parser.add_argument("--block-size", type=int, default=250000)
    parser.add_argument("--projection-mode", choices=["top", "remove", "remove-renorm"], default="top")
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--eval-every", type=int, default=250)
    args = parser.parse_args()

    X = np.load(SHARD / "X_u8.npy", mmap_mode="r")
    Y = np.load(SHARD / "y_f32.npy", mmap_mode="r")
    E = np.load(SHARD / "era_i16.npy", mmap_mode="r")
    eras = np.asarray(E)
    train = np.where(eras <= 791)[0]
    valid = np.where((eras >= 796) & (eras <= 891))[0]
    train_monitor = monitor_indices(train, eras, 128, 7711)
    valid_monitor = monitor_indices(valid, eras, 512, 7712)

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda")
    model = MLP().to(device)
    parameter_count = sum(p.numel() for p in model.parameters())
    assert parameter_count == 7_491_585 and parameter_count > len(train)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.0)
    filt = None
    if args.arm == "spectral":
        common = dict(decay=0.99, warmup=100, normalize="none", weighting="hard",
                      alpha=1.0, soft_residual=True, adaptive="none",
                      relative_eig_tol=args.relative_eig_tol, stabilize_every=100)
        if args.blockwise:
            filt = BlockSpectralGradientFilter(model, total_rank=args.rank,
                                               block_size=args.block_size,
                                               projection_mode=args.projection_mode, **common)
        else:
            filt = StableSpectralGradientFilter(model, optimizer, rank=args.rank, **common)
    rng = np.random.default_rng(args.seed + 1)
    curve, recent = [], []
    mode_suffix = f"-{args.projection_mode}" if args.blockwise else ""
    output = ROOT / "out" / f"{args.arm}{mode_suffix}-r{args.rank}-seed{args.seed}.json"
    started = time.time()
    checkpoints = {1, 25, 50, 100, args.steps}
    checkpoints.update(range(args.eval_every, args.steps + 1, args.eval_every))
    for step in range(1, args.steps + 1):
        j = rng.choice(train, 1024, replace=True)
        x = torch.from_numpy(np.asarray(X[j], np.float32) / 4).to(device)
        y = torch.from_numpy(np.asarray(Y[j], np.float32) - 0.5).to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = F.mse_loss(model(x), y)
        loss.backward()
        diagnostic = None
        if filt is not None:
            diagnostic = filt.filter_grad()
        optimizer.step()
        recent.append(float(loss.detach()))
        if len(recent) > 250:
            recent.pop(0)
        if step in checkpoints:
            row = {"step": step, "sample_equivalent_epochs": step * 1024 / len(train),
                   "batch_mse_250": float(np.mean(recent)),
                   "train": evaluate(model, X, Y, E, train_monitor, device),
                   "valid": evaluate(model, X, Y, E, valid_monitor, device),
                   "elapsed_seconds": time.time() - started}
            if diagnostic is not None:
                row["filter"] = diagnostic
            curve.append(row)
            payload = {"arm": args.arm, "rank": args.rank, "seed": args.seed,
                       "architecture": ARCH, "parameter_count": parameter_count,
                       "learning_rate": args.learning_rate,
                       "relative_eig_tol": args.relative_eig_tol,
                       "blockwise": args.blockwise, "block_size": args.block_size,
                       "projection_mode": args.projection_mode,
                       "train_rows": int(len(train)), "valid_rows": int(len(valid)),
                       "test_touched": False, "steps": args.steps,
                       "paired_batch_stream": True, "curve": curve}
            atomic_json(output, payload)
            print(json.dumps(row), flush=True)
    print(f"DONE {output}", flush=True)


if __name__ == "__main__":
    main()
