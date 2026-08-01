#!/usr/bin/env python3
"""Sub-component #6: regression integration smoke test (local CPU) + C1
spiked-covariance unit tests.

Tests, in order:
  1. SpectralConsensusFilter wrapping AdamW on synthetic Numerai-like
     regression (300 features, 5-bin-style continuous target), 2-layer MLP,
     per-sample MSE, 200 steps. Loss must decrease; diagnostics emitted.
  2. Filter-off path (uniform weights == mean gradient) must reproduce plain
     AdamW to numerical tolerance.
  3. Streaming variant (WeightCovarianceFilterV2, regression-adapted COPY)
     runs 200 steps without exceptions, loss decreases.
  4. C1 spiked-covariance unit tests on _spectral_filter directly:
     (a) pure i.i.d. noise gradients -> filter keeps ~0 directions
     (b) planted coherent spike -> filter keeps it
     (c) i.i.d. noise + correlated zero-signal confound -> MEASURE whether
         the confound is kept (report, don't judge)

All output to stdout (captured into run.log by the caller).
"""
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from spectral_optimizer import SpectralConsensusFilter  # noqa: E402
from weight_cov_optimizer_v2_regression import WeightCovarianceFilterV2Reg  # noqa: E402

torch.manual_seed(0)
OUT = Path(__file__).parent.parent / "out"
OUT.mkdir(exist_ok=True)


def make_data(n=4096, d=300, seed=0, snr=0.5):
    """Numerai-like synthetic regression: y = Xw + noise, then binned to 5
    levels {0,.25,.5,.75,1} like the Numerai target, centered to [-0.5,0.5]."""
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n, d, generator=g)
    w = torch.randn(d, generator=g) / d**0.5
    signal = X @ w
    noise = torch.randn(n, generator=g)
    raw = snr * signal + noise
    # bin into 5 quantile levels like the Numerai target
    q = torch.quantile(raw, torch.tensor([0.05, 0.25, 0.75, 0.95]))
    y = torch.bucketize(raw, q).float() / 4.0 - 0.5
    return X, y


def make_mlp(d=300, seed=0):
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(d, 64), nn.ReLU(),
                         nn.Linear(64, 32), nn.ReLU(),
                         nn.Linear(32, 1))


def per_sample_mse(out, y):
    # out: (1, 1) from functional_call on unsqueezed sample; y: (1,)
    return F.mse_loss(out.squeeze(-1), y)


def test_1_integration():
    print("\n=== TEST 1: SpectralConsensusFilter + AdamW regression, 200 steps ===")
    X, y = make_data()
    model = make_mlp()
    base = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)
    opt = SpectralConsensusFilter(model, base, loss_fn=per_sample_mse)
    B = 256
    losses, ks, ratios = [], [], []
    t0 = time.time()
    for step in range(200):
        idx = torch.randint(0, X.shape[0], (B,))
        loss, diag = opt.step(X[idx], y[idx])
        losses.append(loss)
        ks.append(diag["k"])
        ratios.append(diag["consensus_ratio"])
        if step % 50 == 0 or step == 199:
            print(f"  step {step:3d} loss {loss:.5f} k {diag['k']:3d} "
                  f"consensus_ratio {diag['consensus_ratio']:.3f} "
                  f"top_eigs {[round(e, 2) for e in diag['eigenvalues'][:4]]}")
    dt = time.time() - t0
    first, last = sum(losses[:20]) / 20, sum(losses[-20:]) / 20
    decreased = last < first
    print(f"  mean loss first-20 {first:.5f} last-20 {last:.5f} "
          f"decreased={decreased}  ({dt:.1f}s, {dt/200*1000:.0f} ms/step CPU)")
    print(f"  diagnostics emitted: k range [{min(ks)},{max(ks)}], "
          f"ratio range [{min(ratios):.3f},{max(ratios):.3f}]")
    return {"loss_first20": first, "loss_last20": last, "decreased": decreased,
            "k_min": min(ks), "k_max": max(ks),
            "ratio_min": min(ratios), "ratio_max": max(ratios)}


def _run_equivalence(dtype, steps=50, B=256):
    """Returns (single_step_grad_diff, param_diff_after_steps) for
    filter-off (uniform mean of per-sample vmap grads) vs plain AdamW."""
    X, y = make_data()
    X, y = X.to(dtype), y.to(dtype)
    mA = make_mlp(seed=7).to(dtype)
    oA = torch.optim.AdamW(mA.parameters(), lr=1e-3, weight_decay=0.0)
    mB = make_mlp(seed=7).to(dtype)
    oB = torch.optim.AdamW(mB.parameters(), lr=1e-3, weight_decay=0.0)
    filt = SpectralConsensusFilter(mB, oB, loss_fn=per_sample_mse)
    torch.manual_seed(123)
    batches = [torch.randint(0, X.shape[0], (B,)) for _ in range(steps)]
    grad_diff_step0 = None
    for i, idx in enumerate(batches):
        # arm A
        oA.zero_grad()
        loss = F.mse_loss(mA(X[idx]).squeeze(-1), y[idx])
        loss.backward()
        # arm B: filter OFF == uniform weights == mean of per-sample grads
        params = {k: v for k, v in mB.named_parameters()}
        buffers = {k: v for k, v in mB.named_buffers()}
        psg, _ = filt._grad_and_value_fn(params, buffers, X[idx], y[idx])
        flat = filt._flatten_grads(psg, B)
        filt._set_grads(flat.mean(dim=0))
        if i == 0:
            grad_diff_step0 = max((pa.grad - pb.grad).abs().max().item()
                                  for pa, pb in zip(mA.parameters(),
                                                    mB.parameters()))
        oA.step()
        oB.step()
        oB.zero_grad()
    param_diff = max((pa - pb).abs().max().item()
                     for pa, pb in zip(mA.parameters(), mB.parameters()))
    return grad_diff_step0, param_diff


def test_2_filter_off_equivalence():
    print("\n=== TEST 2: filter-off path == plain AdamW (numerical tolerance) ===")
    g32, p32 = _run_equivalence(torch.float32)
    g64, p64 = _run_equivalence(torch.float64)
    print(f"  fp32: single-step grad max diff {g32:.2e}, "
          f"50-step param max diff {p32:.2e}")
    print(f"  fp64: single-step grad max diff {g64:.2e}, "
          f"50-step param max diff {p64:.2e}")
    # Algorithmic equivalence: fp64 trajectory must agree to near machine
    # precision; fp32 single-step gradient to fp32 epsilon scale. fp32
    # 50-step drift is reported (AdamW amplifies rounding) but not gated.
    ok = (g64 < 1e-12) and (p64 < 1e-9) and (g32 < 1e-6)
    print(f"  algorithmic equivalence (fp64 traj < 1e-9, fp32 grad < 1e-6): "
          f"{'PASS' if ok else 'FAIL'}")
    return {"fp32_grad_diff": g32, "fp32_param_diff_50step": p32,
            "fp64_grad_diff": g64, "fp64_param_diff_50step": p64,
            "pass": ok}


def test_3_streaming_variant():
    print("\n=== TEST 3: streaming variant (regression-adapted copy), 200 steps ===")
    X, y = make_data()
    model = make_mlp(seed=3)
    base = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)
    opt = WeightCovarianceFilterV2Reg(model, base, rank=50, warmup=20)
    B = 256
    losses = []
    for step in range(200):
        idx = torch.randint(0, X.shape[0], (B,))
        loss, diag = opt.step(X[idx], y[idx].unsqueeze(-1))
        losses.append(loss)
        if step % 50 == 0 or step == 199:
            print(f"  step {step:3d} loss {loss:.5f} "
                  f"basis_rank {diag.get('basis_rank', 0)} "
                  f"filtering={diag['filtering_active']}")
    first, last = sum(losses[:20]) / 20, sum(losses[-20:]) / 20
    decreased = last < first
    print(f"  mean loss first-20 {first:.5f} last-20 {last:.5f} decreased={decreased}")
    return {"loss_first20": first, "loss_last20": last, "decreased": decreased}


def _filter_G(G, mode="hard"):
    """Run _spectral_filter on a constructed per-sample gradient matrix."""
    d = G.shape[1]
    model = nn.Linear(2, 1)  # dummy; _spectral_filter doesn't touch the model
    base = torch.optim.AdamW(model.parameters())
    kw = {}
    if mode == "soft":
        kw["soft"] = True
    elif mode == "variance":
        kw["variance_threshold"] = 0.9
    filt = SpectralConsensusFilter(model, base, loss_fn=per_sample_mse, **kw)
    g, diag = filt._spectral_filter(G, G.shape[0])
    # count above MP threshold (before the max(1,.) floor)
    ev = torch.tensor(diag["eigenvalues"])
    return g, diag


def test_4_c1_units():
    print("\n=== TEST 4 (C1): spiked-covariance unit tests on _spectral_filter ===")
    B, p = 256, 5000
    g = torch.Generator().manual_seed(42)
    results = {}

    def analyze(name, G, ref_dir=None):
        row = {}
        for mode in ["hard", "soft", "variance"]:
            cg, diag = _filter_G(G, mode)
            mean_grad = G.mean(0)
            cos_mean = F.cosine_similarity(cg, mean_grad, dim=0).item() \
                if cg.norm() > 1e-12 else float("nan")
            entry = {"k": diag["k"], "ratio": diag["consensus_ratio"],
                     "top_eigs": [round(e, 3) for e in diag["eigenvalues"][:5]],
                     "cos_vs_mean_grad": round(cos_mean, 4)}
            if ref_dir is not None:
                cos_ref = F.cosine_similarity(cg, ref_dir, dim=0).abs().item() \
                    if cg.norm() > 1e-12 else float("nan")
                entry["cos_vs_planted_dir"] = round(cos_ref, 4)
            # raw count above MP threshold, no floor
            ev = torch.linalg.eigvalsh(
                (G / G.norm(dim=1, keepdim=True).clamp(min=1e-12))
                @ (G / G.norm(dim=1, keepdim=True).clamp(min=1e-12)).T
            ).flip(0).clamp(min=0)
            thr = 2.0 * ev.sum().item() / B
            entry["n_above_MP_thr"] = int((ev > thr).sum().item())
            row[mode] = entry
            print(f"  [{name}][{mode:8s}] k={entry['k']:3d} "
                  f"raw_above_thr={entry['n_above_MP_thr']:3d} "
                  f"ratio={entry['ratio']:.3f} cos_mean={cos_mean:.3f}"
                  + (f" cos_planted={entry.get('cos_vs_planted_dir', float('nan')):.3f}"
                     if ref_dir is not None else ""))
        return row

    # (a) pure i.i.d. noise
    G_noise = torch.randn(B, p, generator=g)
    results["a_iid_noise"] = analyze("a: iid noise      ", G_noise)

    # (b) planted spike: every sample shares coherent direction u (+ noise).
    # Amplitude 15 -> cross-sample correlation rho = a^2/(a^2+p) ~ 0.043 ->
    # spike eigenvalue ~ 1 + B*rho ~ 12, clearly above the MP-factor
    # threshold of 2.0. (Amplitude 3 gives eig ~1.45 < 2.0: sub-threshold
    # by construction, not a detection test.)
    u = torch.randn(p, generator=g)
    u = u / u.norm()
    z = 1.0 + 0.1 * torch.randn(B, 1, generator=g)  # same-sign loadings
    G_spike = 15.0 * z * u.unsqueeze(0) + torch.randn(B, p, generator=g)
    results["b_planted_spike"] = analyze("b: planted spike  ", G_spike, ref_dir=u)

    # (c) i.i.d. noise + correlated zero-signal confound: samples share a
    # coherent direction c with random-sign loadings summing to ~0 (a factor
    # that correlates samples but contributes ~nothing to the mean gradient
    # i.e. no persistent descent signal). Same amplitude as (b) so the
    # confound genuinely spikes the spectrum.
    c = torch.randn(p, generator=g)
    c = c / c.norm()
    signs = (torch.randint(0, 2, (B, 1), generator=g).float() * 2 - 1)
    signs = signs - signs.mean()  # exact zero-mean loadings
    G_conf = 15.0 * signs * c.unsqueeze(0) + torch.randn(B, p, generator=g)
    results["c_zero_signal_confound"] = analyze("c: 0-signal conf. ", G_conf,
                                                ref_dir=c)
    return results


if __name__ == "__main__":
    print(f"torch {torch.__version__}, CPU smoke test, seed 0")
    t0 = time.time()
    out = {}
    out["test1_integration"] = test_1_integration()
    out["test2_filter_off"] = test_2_filter_off_equivalence()
    out["test3_streaming"] = test_3_streaming_variant()
    out["test4_c1"] = test_4_c1_units()
    out["total_seconds"] = time.time() - t0
    with open(OUT / "smoke_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nTotal: {out['total_seconds']:.1f}s. Results -> out/smoke_results.json")
