# exp-f02 — Component #5: SpectralGradientFilter integration + alpha=0 identity + planted-subspace + eigh fallback + A8 diagnostics (LOCAL CPU, zero GPU)

**Component**: #5 in `<run-dir>/decomposition.md` (lambda 0.22, P=0.80) — READ ITS FULL SECTION ("Component 5") plus amendment D1 in the "Round-2 Challenge Addendum". Also read the A8 verification-depth section of `<run-dir>/success-criteria.md`.

**Run dir**: `/media/titus/big/researcher-output/2026-08-03-followup-follow-up-spectral-optimizer-redo-pxp-filter-walk-`
**Filter (READ-ONLY source)**: `/home/titus/pyg/optimizers/spectral_filter.py` — class `SpectralGradientFilter` at line 52, `filter_grad()` at line 303. FIRST ACTION: copy this file to `<run-dir>/experiments/exp-f02/src/spectral_filter.py`. That copy is the run's canonical filter — all later experiments (including cluster jobs) will use it. If a fix is ever needed it happens in the copy, documented; NEVER edit the original in ~/pyg.
**Parent eigh-fallback patch (READ-ONLY reference)**: `/media/titus/big/researcher-output/2026-07-31-spectral-optimizer-for-noise-reduction-on-financial-timeseri/audit/rerun-exp-004/src/` — the parent audit's working CPU-fp64 eigh fallback (parent audit Finding 5: fp32 cuSOLVER eigh fails stochastically under rank collapse). Transplant its pattern to the (k+1)×(k+1) eigh in the run's filter copy, with a logged firing count, wired from the start.

## Task

Local CPU only. All work inside `<run-dir>/experiments/exp-f02/` (src/, out/, run.log).

Usage pattern: `filt = SpectralGradientFilter(model, base_opt, rank=k)`; per step: `loss.backward(); filt.filter_grad(); base_opt.step()`. Use `normalize="none"` (parent finding H7). Knobs that must be reachable and exercised at least once: rank, decay, warmup, weighting (hard/soft), alpha, soft_residual, energy_threshold, adaptive (none/effrank/gap).

Build a small synthetic Numerai-shaped regression task (2-layer MLP, ~feature-count scaled down, synthetic eras) and run the five sub-checks:

(a) **Filtered training runs**: 200 steps with the filter active (hard top-k, modest rank), loss falls, all diagnostics emitted (see below).
(b) **alpha=0 soft identity**: in fp64, 50-step parameter trajectories of [soft weighting, alpha=0] vs plain AdamW — bit-identical or max abs diff <= 1e-12. **A structural identity failure (alpha=0 is not a no-op) is STOP-and-report** — do not engineer around it, do not substitute files; report FAIL immediately with the evidence.
(c) **Planted-subspace correctness**: construct a task whose gradient stream has a planted dominant direction; verify hard top-k recovers it (cosine between the filter's top learned basis direction and the planted direction > ~0.9). This exercises the filtering path the identity leaves untouched. A failure here is a filtering-path bug: fix in the run's copy and document, or STOP if it resists (> 2h surgery).
(d) **Forced fallback**: monkeypatch/force a `torch.linalg.LinAlgError` in the per-step eigh → the CPU-fp64 fallback fires, is counted+logged, training continues.
(e) **Zero-predictor sanity**: a seeded zero/constant predictor evaluates to ≈ 0 numerai_corr on synthetic eras (validates the eval plumbing's null).

**A8/B3/D1 diagnostics module** (deliverable code, reused by every later arm-B run — put it in src/ as an importable module): per-step (i) filtered-vs-unfiltered gradient cosine, (ii) kept-norm fraction, (iii) realized k(t), (iv) basis-rotation rate as principal angles between the realized basis at t and t−Δ, (v) VALID-score-vs-step logging hook (D1). Verify it emits sane values on run (a) (principal angles in [0, π/2], kept-norm in [0,1]).

Also verify: arm B/C RNG consumption drawn from a separate torch.Generator so seed pairing across arms does not silently break (demonstrate: same data order for arm A and arm B runs at the same seed).

## Pass criterion
All five sub-checks pass; identity holds in fp64; diagnostics module emits sane values; fallback count logged; RNG-separation demonstrated.

## Fail criterion
Structural identity failure (STOP-and-report), planted direction not recovered after <= 2h surgery, or integration needs > 2h surgery.

## Constraints
- NEVER modify files outside `<run-dir>`. `~/pyg/` and the parent run dir are READ-ONLY (copy, never edit).
- Local torch is 2.11.0 (cluster pins 2.5.1 — do NOT try to match locally; the on-cluster identity re-assert in EU-1 covers the version risk; note any 2.11-only API you rely on in results.md so EU-1 can watch for it).
- Write `results.md` with PASS/FAIL per sub-check, the identity max-diff number, the planted-direction cosine, the fallback firing count, and a short "interface notes" section (anything surprising about the filter's API/behaviour that later cluster jobs must know). Full stdout to `run.log`.
