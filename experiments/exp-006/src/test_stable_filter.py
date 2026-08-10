#!/usr/bin/env python3
"""Atomic numerical tests for the local stable spectral filter."""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from spectral_filter_fixed import StableSpectralGradientFilter


class Flat(nn.Module):
    def __init__(self, p, dtype=torch.float64):
        super().__init__(); self.w = nn.Parameter(torch.zeros(p, dtype=dtype))


def make_filter(p, rank, dtype=torch.float64, **kwargs):
    model = Flat(p, dtype=dtype)
    opt = torch.optim.SGD(model.parameters(), lr=0)
    relative_eig_tol = kwargs.pop("relative_eig_tol", 1e-12)
    stabilize_every = kwargs.pop("stabilize_every", 7)
    filt = StableSpectralGradientFilter(
        model, opt, rank=rank, decay=.9, warmup=0,
        weighting="hard", relative_eig_tol=relative_eig_tol,
        stabilize_every=stabilize_every, **kwargs)
    return model, filt


def feed(model, filt, g):
    model.w.grad = g.clone()
    filt.step_count += 1
    filt._update_svd(g)


def represented_covariance(filt):
    if filt.V is None:
        return None
    return (filt.V * filt.S.to(filt.V).square().unsqueeze(0)) @ filt.V.T


def test_exact_covariance_agreement():
    torch.manual_seed(1)
    p, rank = 9, 9
    model, filt = make_filter(p, rank)
    mean = None; exact = None
    for step, g in enumerate(torch.randn(40, p, dtype=torch.float64), 1):
        if mean is None:
            mean = g.clone()
        else:
            mean = .9 * mean + .1 * g
        centered = g - mean
        feed(model, filt, g)
        if exact is None and centered.norm() > 0:
            exact = torch.outer(centered, centered)
        elif exact is not None:
            exact = .9 * exact + .1 * torch.outer(centered, centered)
        if exact is not None:
            got = represented_covariance(filt)
            torch.testing.assert_close(got, exact, rtol=2e-9, atol=2e-10)
            assert filt.orthogonality_error() < 1e-9


def test_projection_is_nonexpansive_and_idempotent():
    torch.manual_seed(2)
    model, filt = make_filter(30, 12)
    for g in torch.randn(80, 30, dtype=torch.float64):
        feed(model, filt, g)
    for g in torch.randn(20, 30, dtype=torch.float64):
        pg = filt._project_gradient(g)
        ppg = filt._project_gradient(pg)
        assert pg.norm() <= g.norm() * (1 + 1e-10)
        torch.testing.assert_close(ppg, pg, rtol=1e-9, atol=1e-10)
    assert filt.orthogonality_error() < 1e-9


def test_global_gradient_scale_invariance():
    torch.manual_seed(3)
    stream = torch.randn(70, 24, dtype=torch.float64)
    filters = []
    for scale in (1e-5, 1.0, 1e5):
        model, filt = make_filter(24, 10)
        for g in stream:
            feed(model, filt, scale*g)
        filters.append(filt)
    projectors = [f.V @ f.V.T for f in filters]
    for p in projectors[1:]:
        torch.testing.assert_close(p, projectors[0], rtol=2e-8, atol=2e-9)
    assert len({f.V.shape[1] for f in filters}) == 1


def test_state_resume_equivalence():
    torch.manual_seed(4)
    stream = torch.randn(60, 20, dtype=torch.float64)
    m1, f1 = make_filter(20, 8)
    for g in stream[:31]: feed(m1, f1, g)
    state = {"V": f1.V.clone(), "S": f1.S.clone(),
             "mean": f1.grad_mean.clone(), "step": f1.step_count,
             "stabilizations": f1.stabilization_count}
    for g in stream[31:]: feed(m1, f1, g)

    m2, f2 = make_filter(20, 8)
    f2.V = state["V"].clone(); f2.S = state["S"].clone()
    f2.grad_mean = state["mean"].clone(); f2.step_count = state["step"]
    f2.stabilization_count = state["stabilizations"]
    for g in stream[31:]: feed(m2, f2, g)
    torch.testing.assert_close(f2.S, f1.S, rtol=0, atol=0)
    torch.testing.assert_close(f2.V @ f2.V.T, f1.V @ f1.V.T, rtol=0, atol=0)


def test_rank_one_planted_subspace():
    torch.manual_seed(5)
    p = 40
    planted = torch.randn(p, dtype=torch.float64); planted /= planted.norm()
    model, filt = make_filter(p, 6)
    for _ in range(100):
        g = torch.randn((), dtype=torch.float64) * planted
        feed(model, filt, g)
    alignment = torch.abs(torch.dot(filt.V[:, 0], planted))
    assert alignment > .999999


def test_fp32_long_stream_stays_a_projection():
    torch.manual_seed(6)
    model, filt = make_filter(512, 128, dtype=torch.float32,
                              relative_eig_tol=1e-8, stabilize_every=100)
    for g in torch.randn(1200, 512, dtype=torch.float32):
        feed(model, filt, g)
    assert filt.V.shape[1] == 128
    assert filt.orthogonality_error() < 5e-4
    for g in torch.randn(20, 512, dtype=torch.float32):
        pg = filt._project_gradient(g)
        assert pg.norm() <= g.norm() * (1 + 2e-5)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test(); print(f"PASS {test.__name__}")
