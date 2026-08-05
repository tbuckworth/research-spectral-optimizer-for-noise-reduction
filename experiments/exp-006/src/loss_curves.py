#!/usr/bin/env python3
"""Paired AdamW/spectral loss trajectories on identical Numerai batches."""
from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import driver
from spectral_filter_fixed import StableSpectralGradientFilter

ROOT = Path(__file__).resolve().parent.parent


def fixed_mse(model, X, Y, idx, device):
    model.eval(); total = 0.0; count = 0
    with torch.no_grad():
        for start in range(0, len(idx), 4096):
            j = idx[start:start+4096]
            x = torch.from_numpy(np.asarray(X[j], dtype=np.float32)/4).to(device)
            y = torch.from_numpy(np.asarray(Y[j], dtype=np.float32)-.5).to(device)
            pred = model(x); total += float(F.mse_loss(pred, y, reduction="sum")); count += len(j)
    model.train(); return total/count


def train_arm(arm, seed, steps, protocol, X, Y, train_idx, monitor_idx):
    model_seed = driver.derive_seed(protocol["protocol_id"], 1, seed, "model-data")
    random.seed(model_seed); np.random.seed(model_seed % 2**32)
    torch.manual_seed(model_seed); torch.cuda.manual_seed_all(model_seed)
    rng = np.random.default_rng(model_seed); device = torch.device("cuda")
    model = driver.MLP(protocol["data_and_model"]["architecture"], 0.0).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.003, weight_decay=0.0)
    filt = None if arm == "AdamW" else StableSpectralGradientFilter(
        model, optimizer, rank=16, decay=.99, warmup=100,
        normalize="none", weighting="hard", relative_eig_tol=1e-8,
        stabilize_every=100)
    losses, ema, fixed = [], [], []
    ema_value = None
    checkpoints = set([1, 25, 50, 100] + list(range(200, steps+1, 200)) + [steps])
    for step in range(1, steps+1):
        j = rng.choice(train_idx, protocol["data_and_model"]["batch_size"], replace=True)
        x = torch.from_numpy(np.asarray(X[j], dtype=np.float32)/4).to(device)
        y = torch.from_numpy(np.asarray(Y[j], dtype=np.float32)-.5).to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = F.mse_loss(model(x), y); loss.backward()
        if filt is not None: filt.filter_grad()
        optimizer.step()
        value = float(loss); losses.append(value)
        ema_value = value if ema_value is None else .98*ema_value + .02*value
        ema.append(ema_value)
        if step in checkpoints:
            fixed.append({"step": step, "mse": fixed_mse(model, X, Y, monitor_idx, device)})
            print(f"{arm} seed={seed} step={step}/{steps} batch={value:.8f} fixed={fixed[-1]['mse']:.8f}", flush=True)
    return {"arm": arm, "seed": seed, "steps": steps, "batch_loss": losses,
            "ema_0.98": ema, "fixed_train_mse": fixed,
            "last_100_mean": float(np.mean(losses[-100:])),
            "last_500_mean": float(np.mean(losses[-500:])),
            "fixed_final": fixed[-1]["mse"]}


def main():
    protocol_path = Path(os.environ.get("NUMERAI_PROTOCOL", ROOT/"protocol.json"))
    protocol = json.load(open(protocol_path))
    shard_env = os.environ.get("NUMERAI_SHARD")
    shard = Path(shard_env) if shard_env else driver.find_shard(protocol)
    X, Y, E = driver.open_shard(shard)
    train_idx = np.where(np.asarray(E) <= 891)[0]
    # Fixed monitor chosen once, independently of all model seeds and arms.
    monitor_rng = np.random.default_rng(620260805)
    monitor_idx = np.sort(monitor_rng.choice(train_idx, 65536, replace=False))
    endpoints = [2301, 2301, 2301, 2301, 3452]
    rows = []
    for seed, steps in enumerate(endpoints):
        for arm in ("AdamW", "StableSpectral"):
            rows.append(train_arm(arm, seed, steps, protocol, X, Y, train_idx, monitor_idx))
    driver.atomic_json(ROOT/"out"/"paired-loss-curves.json", {
        "monitor_rows": len(monitor_idx), "monitor_selection_seed": 620260805,
        "same_initialization_and_batch_stream_within_seed": True, "runs": rows})


if __name__ == "__main__": main()
