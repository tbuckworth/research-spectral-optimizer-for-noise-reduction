"""exp-002: Sequence-arm vmap feasibility (hand-rolled GRU cell).

Tests whether per-sample gradients can be computed through a recurrent model
with torch.func vmap(grad(...)), correctly and fast enough for a sequence-model
arm of the spectral-optimizer study.

CPU-only correctness smoke test. Steps:
  1. Hand-rolled 1-layer GRU cell as an nn.Module with explicit matmul ops
     (no cuDNN nn.GRU), sequence-to-one regression head.
  2. Synthetic Numerai-shaped within-era sequence data (T=16, D=100).
  3. vmap(grad(per_sample_loss)) at B=64 vs plain per-sample python-loop
     reference; require max abs diff <= 1e-5 over all parameter tensors.
  4. CPU timing at B=1024 (per-sample grads + BxB eigh + full filtered step);
     project a full training run on one L40 with a stated conservative
     CPU->GPU scaling factor.
  5. One end-to-end SpectralConsensusFilter step on the GRU model.
"""

import copy
import os
import sys
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # enforce CPU

import torch
import torch.nn as nn
from torch.func import functional_call, grad, grad_and_value, vmap

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spectral_optimizer import SpectralConsensusFilter  # local copy (read-only source repo)

torch.manual_seed(0)

DEVICE = torch.device("cpu")
D_IN = 100     # feature dim (Numerai-shaped: ~50-300 dims)
H = 64         # hidden dim
T = 16         # sequence length (within-era framing, ~10-20)
B_CHECK = 64   # batch for correctness check
B_TIME = 1024  # batch for timing (max planned per-sample-grad batch)


class GRURegressor(nn.Module):
    """1-layer GRU, hand-rolled cell (pure tensor ops -> vmap-compatible),
    sequence-to-one scalar regression head."""

    def __init__(self, d_in, h):
        super().__init__()
        self.w_ih = nn.Parameter(torch.randn(3 * h, d_in) * (1.0 / d_in ** 0.5))
        self.w_hh = nn.Parameter(torch.randn(3 * h, h) * (1.0 / h ** 0.5))
        self.b_ih = nn.Parameter(torch.zeros(3 * h))
        self.b_hh = nn.Parameter(torch.zeros(3 * h))
        self.head_w = nn.Parameter(torch.randn(h) * (1.0 / h ** 0.5))
        self.head_b = nn.Parameter(torch.zeros(1))
        self.h = h

    def forward(self, x):
        # x: (B, T, D) -> (B,) scalar prediction
        bsz = x.shape[0]
        h = x.new_zeros(bsz, self.h)
        for t in range(x.shape[1]):
            gi = x[:, t, :] @ self.w_ih.T + self.b_ih
            gh = h @ self.w_hh.T + self.b_hh
            i_r, i_z, i_n = gi.chunk(3, dim=1)
            h_r, h_z, h_n = gh.chunk(3, dim=1)
            r = torch.sigmoid(i_r + h_r)
            z = torch.sigmoid(i_z + h_z)
            n = torch.tanh(i_n + r * h_n)
            h = (1.0 - z) * n + z * h
        return h @ self.head_w + self.head_b


def mse_loss(pred, target):
    return ((pred - target) ** 2).mean()


def make_data(n, t, d, seed):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, t, d, generator=g)
    # weak-signal continuous target: tiny linear signal + heavy noise
    w = torch.randn(d, generator=g) / d ** 0.5
    y = 0.05 * (x.mean(dim=1) @ w) + torch.randn(n, generator=g)
    return x, y


def per_sample_grads_vmap(model, x, y):
    params = {k: v for k, v in model.named_parameters()}
    buffers = {k: v for k, v in model.named_buffers()}

    def loss_fn(p, b, xi, yi):
        out = functional_call(model, (p, b), (xi.unsqueeze(0),))
        return mse_loss(out, yi.unsqueeze(0))

    fn = vmap(grad(loss_fn), in_dims=(None, None, 0, 0))
    return fn(params, buffers, x, y)


def per_sample_grads_loop(model, x, y):
    names = [k for k, _ in model.named_parameters()]
    out = {k: [] for k in names}
    for i in range(x.shape[0]):
        model.zero_grad(set_to_none=True)
        loss = mse_loss(model(x[i:i + 1]), y[i:i + 1])
        grads = torch.autograd.grad(loss, list(model.parameters()))
        for k, g in zip(names, grads):
            out[k].append(g.detach().clone())
    return {k: torch.stack(v) for k, v in out.items()}


def main():
    print(f"torch {torch.__version__}, device={DEVICE}, threads={torch.get_num_threads()}")
    model = GRURegressor(D_IN, H).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"GRU regressor: D={D_IN}, H={H}, T={T}, params={n_params}")

    # ---------- 1. Correctness: vmap vs per-sample loop ----------
    x, y = make_data(B_CHECK, T, D_IN, seed=1)
    t0 = time.perf_counter()
    g_vmap = per_sample_grads_vmap(model, x, y)
    t_vmap_small = time.perf_counter() - t0
    t0 = time.perf_counter()
    g_loop = per_sample_grads_loop(model, x, y)
    t_loop = time.perf_counter() - t0

    max_diff = 0.0
    print("\n-- correctness: vmap(grad) vs python-loop reference (B=%d) --" % B_CHECK)
    for k in g_loop:
        d = (g_vmap[k] - g_loop[k]).abs().max().item()
        gnorm = g_loop[k].abs().max().item()
        print(f"  {k:10s} shape={tuple(g_loop[k].shape)}  max_abs_diff={d:.3e}  ref_max_abs={gnorm:.3e}")
        max_diff = max(max_diff, d)
    correctness_pass = max_diff <= 1e-5
    print(f"MAX ABS DIFF (all params): {max_diff:.3e}  -> {'PASS' if correctness_pass else 'FAIL'} (tol 1e-5)")
    print(f"timing: vmap B={B_CHECK}: {t_vmap_small:.3f}s, loop B={B_CHECK}: {t_loop:.3f}s")

    # ---------- 2. CPU timing at B=1024 ----------
    xb, yb = make_data(B_TIME, T, D_IN, seed=2)
    # warmup
    _ = per_sample_grads_vmap(model, xb[:64], yb[:64])

    n_rep = 3
    times_grad, times_eigh = [], []
    for _ in range(n_rep):
        t0 = time.perf_counter()
        g = per_sample_grads_vmap(model, xb, yb)
        times_grad.append(time.perf_counter() - t0)
        G = torch.cat([g[k].reshape(B_TIME, -1) for k in g], dim=1)
        Gn = G / G.norm(dim=1, keepdim=True).clamp(min=1e-12)
        S = Gn @ Gn.T
        t0 = time.perf_counter()
        torch.linalg.eigh(S)
        times_eigh.append(time.perf_counter() - t0)
    t_grad = sum(times_grad) / n_rep
    t_eigh = sum(times_eigh) / n_rep
    print(f"\n-- CPU timing (B={B_TIME}, mean of {n_rep}) --")
    print(f"  per-sample grads (vmap): {t_grad:.3f}s   runs: {[f'{t:.3f}' for t in times_grad]}")
    print(f"  eigh {B_TIME}x{B_TIME}:          {t_eigh:.3f}s   runs: {[f'{t:.3f}' for t in times_eigh]}")

    # full filtered step at B=1024 via SpectralConsensusFilter
    model_t = copy.deepcopy(model)
    opt = torch.optim.AdamW(model_t.parameters(), lr=1e-3)
    filt = SpectralConsensusFilter(model_t, opt, loss_fn=mse_loss)
    filt.step(xb[:64], yb[:64])  # warmup/compile
    times_step = []
    for _ in range(n_rep):
        t0 = time.perf_counter()
        filt.step(xb, yb)
        times_step.append(time.perf_counter() - t0)
    t_step = sum(times_step) / n_rep
    print(f"  full filtered step:      {t_step:.3f}s   runs: {[f'{t:.3f}' for t in times_step]}")

    # ---------- 3. Projection to one L40 ----------
    # Conservative scaling assumption (exp-001 timing not available at run
    # time): an L40 (90+ TFLOPS fp32-TC, 864 GB/s) vs this 8-thread desktop
    # CPU on batched-matmul-dominated work is conservatively taken as 10x
    # (typical observed factors for vmap per-sample-grad workloads are
    # 30-100x). eigh at 1024x1024 is also faster on GPU; we conservatively
    # apply the same 10x to the whole step.
    SPEEDUP = 10.0
    N_STEPS = 2000  # assumed full training run: e.g. ~2000 filtered steps
    proj_step = t_step / SPEEDUP
    proj_run_min = proj_step * N_STEPS / 60.0
    budget_min = 25.0
    runtime_pass = proj_run_min < budget_min
    print(f"\n-- L40 projection (conservative {SPEEDUP:.0f}x CPU->L40 speedup) --")
    print(f"  projected step time (B={B_TIME}): {proj_step:.3f}s")
    print(f"  projected full run ({N_STEPS} steps): {proj_run_min:.1f} min "
          f"-> {'PASS' if runtime_pass else 'FAIL'} (budget {budget_min:.0f} min)")
    max_steps = int(budget_min * 60.0 / proj_step)
    print(f"  max affordable steps within {budget_min:.0f} min: {max_steps}")

    # ---------- 4. End-to-end filtered optimizer step on the GRU ----------
    print("\n-- end-to-end SpectralConsensusFilter step on GRU (B=64) --")
    model_e = copy.deepcopy(model)
    opt_e = torch.optim.AdamW(model_e.parameters(), lr=1e-3)
    filt_e = SpectralConsensusFilter(model_e, opt_e, loss_fn=mse_loss)
    before = {k: v.detach().clone() for k, v in model_e.named_parameters()}
    loss, diag = filt_e.step(x, y)
    changed = {k: (v.detach() - before[k]).abs().max().item()
               for k, v in model_e.named_parameters()}
    finite = all(torch.isfinite(v).all() for v in model_e.parameters())
    all_changed = all(c > 0 for c in changed.values())
    e2e_pass = finite and all_changed and loss == loss  # loss==loss: not NaN
    print(f"  loss={loss:.6f}, k={diag['k']}, consensus_ratio={diag['consensus_ratio']:.4f}")
    print(f"  top eigenvalues: {[f'{e:.3f}' for e in diag['eigenvalues'][:5]]}")
    print(f"  params all finite: {finite}; all params updated: {all_changed}")
    print(f"  max param change per tensor: " +
          ", ".join(f"{k}={c:.2e}" for k, c in changed.items()))
    print(f"  end-to-end filtered step: {'PASS' if e2e_pass else 'FAIL'}")

    # ---------- Verdict ----------
    verdict = correctness_pass and runtime_pass and e2e_pass
    print("\n==== SUMMARY ====")
    print(f"correctness (max_abs_diff<=1e-5):        {'PASS' if correctness_pass else 'FAIL'} ({max_diff:.3e})")
    print(f"projected L40 run < {budget_min:.0f} min:            {'PASS' if runtime_pass else 'FAIL'} ({proj_run_min:.1f} min @ {N_STEPS} steps)")
    print(f"end-to-end filtered step on GRU:         {'PASS' if e2e_pass else 'FAIL'}")
    print(f"VERDICT: {'PASS' if verdict else 'FAIL'}")


if __name__ == "__main__":
    main()
