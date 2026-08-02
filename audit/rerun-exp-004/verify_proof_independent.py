#!/usr/bin/env python3
"""AUDIT: independent check of the exp-001 target-independence claim.
Different seed/arch/batch from the producer's script, plus two controls:
(1) the UNNORMALIZED gradient similarity spectrum SHOULD be target-dependent
    (if it weren't, the check would be vacuous);
(2) a multi-output (2-dim) MSE head SHOULD break target-independence
    (the proof is claimed specific to scalar output)."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import vmap, grad, functional_call

torch.manual_seed(20260802)
B, d = 128, 37
model = nn.Sequential(nn.Linear(d, 21), nn.Tanh(), nn.Linear(21, 11),
                      nn.ReLU(), nn.Linear(11, 1)).double()
X = torch.randn(B, d, dtype=torch.float64)
y1 = torch.randn(B, dtype=torch.float64)
y2 = torch.randn(B, dtype=torch.float64)

params = dict(model.named_parameters())
bufs = dict(model.named_buffers())


def loss_fn(p, b, x, y):
    out = functional_call(model, (p, b), (x.unsqueeze(0),))
    return F.mse_loss(out.squeeze(), y)


gfn = vmap(grad(loss_fn), in_dims=(None, None, 0, 0))


def G_of(y):
    g = gfn(params, bufs, X, y)
    return torch.cat([g[k].reshape(B, -1) for k in g], dim=1)


def spec(G, normalize):
    if normalize:
        G = G / G.norm(dim=1, keepdim=True).clamp(min=1e-12)
    return torch.linalg.eigvalsh(G @ G.T).flip(0)


G1, G2 = G_of(y1), G_of(y2)
dn = (spec(G1, True) - spec(G2, True)).abs().max().item()
du = (spec(G1, False) - spec(G2, False)).abs().max().item()
print(f"row-NORMALIZED spectrum max|diff| across targets: {dn:.3e} "
      f"(claim: ~0)")
print(f"UNNORMALIZED  spectrum max|diff| across targets: {du:.3e} "
      f"(control: should be >> 0)")
assert dn < 1e-9 and du > 1e-6

# multi-output control: proof should NOT hold
model2 = nn.Sequential(nn.Linear(d, 16), nn.ReLU(),
                       nn.Linear(16, 2)).double()
p2, b2 = dict(model2.named_parameters()), dict(model2.named_buffers())
Y1 = torch.randn(B, 2, dtype=torch.float64)
Y2 = torch.randn(B, 2, dtype=torch.float64)


def loss2(p, b, x, y):
    out = functional_call(model2, (p, b), (x.unsqueeze(0),))
    return F.mse_loss(out.squeeze(0), y)


gfn2 = vmap(grad(loss2), in_dims=(None, None, 0, 0))


def spec2(Y):
    g = gfn2(p2, b2, X, Y)
    G = torch.cat([g[k].reshape(B, -1) for k in g], dim=1)
    Gn = G / G.norm(dim=1, keepdim=True).clamp(min=1e-12)
    return torch.linalg.eigvalsh(Gn @ Gn.T).flip(0)


dm = (spec2(Y1) - spec2(Y2)).abs().max().item()
print(f"2-output MSE normalized spectrum max|diff|: {dm:.3e} "
      f"(control: should be >> 0)")
assert dm > 1e-6
print("AUDIT CONFIRMED: target-independence holds for scalar-output "
      "row-normalized case and ONLY there (both controls target-dependent).")
