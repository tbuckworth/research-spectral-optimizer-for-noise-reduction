#!/usr/bin/env python3
"""Replay actual Numerai MLP gradients through the corrected local filter."""
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


def projection_metrics(filt, g):
    projected = filt._project_gradient(g)
    gn = g.double().norm().clamp_min(1e-30)
    pn = projected.double().norm().clamp_min(1e-30)
    return {
        "k": int(filt.V.shape[1]),
        "cosine": float(torch.dot(g.double(), projected.double())/(gn*pn)),
        "norm_ratio": float(pn/gn),
        "orthogonality_error": filt.orthogonality_error(),
        "relative_eig_min": float(filt.S.double().square().min()/filt.S.double().square().max()),
    }


def summarize(rows):
    return {key: {"median": float(np.median([r[key] for r in rows])),
                  "p95": float(np.quantile([r[key] for r in rows], .95)),
                  "max": float(np.max([r[key] for r in rows])),
                  "last": float(rows[-1][key])}
            for key in rows[0]}


def main():
    p = json.loads(Path(os.environ["NUMERAI_PROTOCOL"]).read_text())
    shard = driver.find_shard(p)
    X, Y, E = driver.open_shard(shard)
    train_idx = np.where(np.asarray(E) <= 791)[0]
    seed = 20260805
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    rng = np.random.default_rng(seed); dev = torch.device("cuda")
    model = driver.MLP(p["data_and_model"]["architecture"], 0.0).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=.003, weight_decay=0.0)

    configs = [(rank, tol) for rank in (8, 32, 128, 512, 2048)
               for tol in (1e-6, 1e-8, 1e-10)]
    trackers = {(rank, tol): StableSpectralGradientFilter(
        model, opt, rank=rank, decay=.99, warmup=100, weighting="hard",
        normalize="none", relative_eig_tol=tol, stabilize_every=100)
        for rank, tol in configs}
    rows = {key: [] for key in trackers}

    for step in range(1, 1001):
        j = rng.choice(train_idx, p["data_and_model"]["batch_size"], replace=True)
        x = torch.from_numpy(np.asarray(X[j], dtype=np.float32)/4).to(dev)
        y = torch.from_numpy(np.asarray(Y[j], dtype=np.float32)-.5).to(dev)
        opt.zero_grad(set_to_none=True); loss = F.mse_loss(model(x), y); loss.backward()
        g = torch.cat([q.grad.reshape(-1) for q in model.parameters()]).detach()
        for key, filt in trackers.items():
            filt.step_count += 1; filt._update_svd(g)
            if step > 100: rows[key].append(projection_metrics(filt, g))

        # Keep the underlying gradient stream independent of every tracker.
        opt.step()
        if step % 100 == 0: print(f"step={step} loss={loss.item():.8f}", flush=True)

    result = {f"rank={rank},tol={tol:g}": summarize(rows[(rank, tol)])
              for rank, tol in configs}
    driver.atomic_json(ROOT/"out"/"numerai-mechanism-diagnostic.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__": main()
