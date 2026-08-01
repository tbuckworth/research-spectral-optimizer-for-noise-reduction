#!/usr/bin/env python3
"""Verify analytically-derived claim: for a scalar-output model trained with
per-sample MSE, the row-normalized per-sample-gradient similarity matrix
S = Gn Gn^T satisfies S(y') = D S(y) D with D = diag(sign of residual),
for ANY targets y' at the SAME parameter point. Hence its eigenvalues (and
therefore k, the MP-threshold count, and every spectrum-derived engagement
diagnostic) are target-independent at fixed parameters.

Proof sketch: g_i = 2 (f(x_i) - y_i) * d f(x_i)/d theta, so the row-normalized
gradient is sign(f_i - y_i) * J_i/||J_i||; the target enters only through the
sign. D is orthogonal, so eigenvalues of D S D equal eigenvalues of S.

This script checks it numerically in float64 on a fresh MLP + random data.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import vmap, grad_and_value, functional_call

torch.manual_seed(0)
B, d = 64, 50
model = nn.Sequential(nn.Linear(d, 32), nn.ReLU(), nn.Linear(32, 1)).double()
X = torch.randn(B, d, dtype=torch.float64)
y1 = torch.randn(B, dtype=torch.float64)
y2 = y1[torch.randperm(B)]          # permuted targets
y3 = torch.randn(B, dtype=torch.float64)  # entirely different targets

params = {k: v for k, v in model.named_parameters()}
bufs = {k: v for k, v in model.named_buffers()}


def loss_fn(p, b, x, y):
    out = functional_call(model, (p, b), (x.unsqueeze(0),))
    return F.mse_loss(out.squeeze(), y)


gfn = vmap(grad_and_value(loss_fn), in_dims=(None, None, 0, 0))


def spectrum(y):
    g, _ = gfn(params, bufs, X, y)
    G = torch.cat([g[k].reshape(B, -1) for k in g], dim=1)
    Gn = G / G.norm(dim=1, keepdim=True).clamp(min=1e-12)
    return torch.linalg.eigvalsh(Gn @ Gn.T).flip(0)


e1, e2, e3 = spectrum(y1), spectrum(y2), spectrum(y3)
print("max |eig(real) - eig(permuted)|   =", float((e1 - e2).abs().max()))
print("max |eig(real) - eig(unrelated)|  =", float((e1 - e3).abs().max()))
print("top-3 eigenvalues (real):     ", e1[:3].tolist())
print("top-3 eigenvalues (permuted): ", e2[:3].tolist())
assert (e1 - e2).abs().max() < 1e-9 and (e1 - e3).abs().max() < 1e-9
print("CONFIRMED: spectrum is target-independent at fixed parameters (fp64).")
