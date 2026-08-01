# Experiment 002 Results

## Component Tested
Component #2 — Sequence-arm vmap feasibility: per-sample gradients through a
recurrent (hand-rolled GRU) model via `torch.func` `vmap(grad(...))`, consumed
by the unmodified `SpectralConsensusFilter`.

## Verdict: PASS

## Setup
- Environment: local desktop CPU (4 torch threads), Python 3, torch 2.11.0+cu128 (run forced to CPU via `CUDA_VISIBLE_DEVICES=""`). No GPU used; no cluster job needed (projection not borderline).
- Duration: 9.4 s wall-clock for the full test script; well under budget. Peak RSS 1.65 GB.
- Resources used: local CPU only, as designated by the plan. The optional `--qos=debug` cluster GPU job was NOT used — the runtime projection is 6x under budget even with a deliberately conservative scaling factor, so it did not qualify as "borderline".
- Code: `gru_vmap_test.py` (test harness), `spectral_optimizer.py` (verified byte-identical to the read-only source `/home/titus/pyg/optimizers/experiments/spectral_optimizer.py`). Full raw output in `run.log`.

## What Was Tested

A 1-layer GRU cell was hand-rolled as pure tensor ops (explicit matmuls +
sigmoid/tanh gates, no cuDNN `nn.GRU`), with a sequence-to-one scalar
regression head (31,937 params; D=100 features, T=16 steps — Numerai-shaped
within-era synthetic sequences with a weak linear signal under heavy noise).
Per-sample gradients were computed with `vmap(grad(per_sample_loss))` via
`functional_call` over the params dict, at B=64, and compared element-wise
against a plain per-sample python-loop `autograd.grad` reference across all
six parameter tensors.

CPU cost was then measured at B=1024 (per-sample grads, 1024x1024 `eigh`, and
the full `SpectralConsensusFilter.step`), and a full sequence-model training
run on one L40 was projected. Finally, one end-to-end filtered optimizer step
(vmap per-sample grads -> spectral filter -> AdamW step) was run on the GRU
model and checked for finiteness and parameter movement.

## Results

### Raw Output
```
torch 2.11.0+cu128, device=cpu, threads=4
GRU regressor: D=100, H=64, T=16, params=31937

-- correctness: vmap(grad) vs python-loop reference (B=64) --
  w_ih       shape=(64, 192, 100)  max_abs_diff=9.537e-07  ref_max_abs=2.552e+00
  w_hh       shape=(64, 192, 64)  max_abs_diff=2.086e-07  ref_max_abs=5.298e-01
  b_ih       shape=(64, 192)  max_abs_diff=5.960e-07  ref_max_abs=1.221e+00
  b_hh       shape=(64, 192)  max_abs_diff=2.384e-07  ref_max_abs=6.297e-01
  head_w     shape=(64, 64)  max_abs_diff=1.669e-06  ref_max_abs=5.317e+00
  head_b     shape=(64, 1)  max_abs_diff=4.768e-07  ref_max_abs=5.723e+00
MAX ABS DIFF (all params): 1.669e-06  -> PASS (tol 1e-5)

-- CPU timing (B=1024, mean of 3) --
  per-sample grads (vmap): 0.764s
  eigh 1024x1024:          0.035s
  full filtered step:      1.174s

exp-001 results.md available for measured CPU->L40 scaling: False

-- L40 projection (conservative 10x CPU->L40 speedup) --
  projected step time (B=1024): 0.117s
  projected full run (2000 steps): 3.9 min -> PASS (budget 25 min)
  max affordable steps within 25 min: 12773

-- end-to-end SpectralConsensusFilter step on GRU (B=64) --
  loss=1.424806, k=4, consensus_ratio=0.4574
  top eigenvalues: ['3.965', '2.507', '2.147', '2.030', '1.991']
  params all finite: True; all params updated: True

==== SUMMARY ====
correctness (max_abs_diff<=1e-5):        PASS (1.669e-06)
projected L40 run < 25 min:            PASS (3.9 min @ 2000 steps)
end-to-end filtered step on GRU:         PASS
VERDICT: PASS
```

### Metrics
| Metric | Target | Actual | Pass? |
|--------|--------|--------|-------|
| vmap vs loop max abs diff (all param tensors, B=64) | <= 1e-5 | 1.669e-06 | Y |
| Projected sequence-model run on one L40 (2000 filtered steps, B=1024) | < 25 min | 3.9 min | Y |
| End-to-end SpectralConsensusFilter step on GRU | runs, finite, params move | ran; all finite; all 6 tensors updated; k=4, consensus_ratio=0.457 | Y |
| Wall-clock spent on vmap surgery | < 2 h cap | ~0 (no surgery needed) | Y |

### Analysis

The hand-rolled functional GRU cell composes with `torch.func`
`vmap(grad(...))` with no modification: per-sample gradients through the
16-step recurrence match the per-sample loop reference to 1.7e-6 max abs
difference (against reference gradient magnitudes of order 0.5-5.7), an
order of magnitude inside the 1e-5 tolerance and consistent with fp32
accumulation-order differences only. No vmap incompatibilities were
encountered — no in-place ops, no `.item()` calls, no cuDNN kernels in the
path. The one known constraint (cuDNN-backed `nn.GRU`/`nn.LSTM` do not
compose with vmap) was designed around from the start by hand-rolling the
cell; it never fired as a runtime error.

On cost: one full filtered step at B=1024 (per-sample grads 0.76 s + eigh
0.035 s + filter/AdamW overhead) takes 1.17 s on a 4-thread desktop CPU. The
`eigh` on the 1024x1024 similarity matrix is only ~3% of the step; the vmap
gradient computation dominates, which is batched-matmul work that GPUs
accelerate strongly. Scaling assumption (stated per plan): exp-001's measured
CPU-vs-L40 factor was preferred but its results.md did not exist at run time
or at final check (exp-001 runs concurrently), so a conservative documented
10x CPU->L40 factor was applied to the whole step — deliberately below the
30-100x typically observed for vmap per-sample-grad matmul workloads on a
~90 TFLOPS-class card vs a 4-thread desktop CPU. Even so, a 2000-step run
projects to 3.9 min, and ~12,770 steps fit inside the 25-min budget. The
conclusion is insensitive to the scaling assumption: the budget is met at
any speedup >= 1.6x, and a bare CPU run (1x) would be 39 min — only ~1.6x
over, so the GPU merely needs to beat the desktop CPU by 60% for feasibility.

The integration target also holds: the unmodified `SpectralConsensusFilter`
(byte-identical copy of the source repo file) consumed the GRU's per-sample
gradients directly — its internal `vmap(grad_and_value(functional_call(...)))`
path worked on the recurrent model without any changes — and produced a
finite filtered step that moved every parameter tensor, with sane diagnostics
(k=4 eigendirections kept above the MP-factor threshold, 45.7% of the mean-
gradient norm passed) on this synthetic weak-signal data.

## Unexpected Observations
- At B=64 the vmap path (0.81 s) was slower than the python loop (0.30 s) on
  CPU — vmap's batching overhead exceeds its benefit at small batch on CPU.
  At B=1024 vmap amortizes (0.76 s vs a projected ~4.8 s for the loop). This
  is CPU-specific and irrelevant to the GPU projection, but worth knowing for
  any CPU-side debugging.
- Version skew: this correctness result was obtained on torch 2.11.0 locally;
  the cluster venv pins torch 2.5.1+cu121. The `torch.func`
  vmap/grad/functional_call API is stable across that range, but the GPU
  experiments that build on this path should re-run the cheap B=64
  correctness assert on the cluster as a guard (seconds of runtime).
- The filter kept k=4 directions with consensus_ratio 0.457 on pure synthetic
  weak-signal data — the mechanism engages (is not degenerate) even on this
  toy regime, which is a mild positive signal for the engagement diagnostics
  planned in the real-data experiments.

## Implications

What this tells us: the criterion was met — the sequence-architecture arm is
feasible. Per-sample gradients through a functional hand-rolled GRU are
correct under vmap, cheap enough (projected minutes, not hours, per training
run on one L40), and plug into the existing SpectralConsensusFilter without
modification. The MLP-only fallback (the pre-authorized FAIL scope decision)
is NOT needed.

**Amendment F13 (binding claim wording).** This feasibility test used
synthetic *Numerai-shaped within-era sequences* (T=16 of 100-dim feature
vectors), i.e., the within-era Numerai framing. The result therefore supports
the **"architecture consistency"** claim route — sequence arm on the same
dataset (within-era Numerai) as the MLP arm. If the eventual sequence arm is
instead run on OHLCV, F13 fires: the claim must be renamed to **"second
setting"** unless a cheap MLP-on-OHLCV arm is added. Nothing in this
experiment forces the OHLCV route; the demonstrated code path is
dataset-agnostic.

Next steps: proceed to exp-003 as planned. When the sequence arm runs on the
cluster GPU, (a) re-run the B=64 correctness assert on torch 2.5.1 (seconds),
and (b) replace the conservative 10x projection with the measured CPU->L40
factor from exp-001 once its results exist.
