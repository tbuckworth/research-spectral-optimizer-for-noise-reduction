# Experiment f02 Results

## Component Tested
Component #5 (decomposition.md): SpectralGradientFilter integration + alpha=0
identity + planted-subspace + eigh fallback + A8/B3/D1 diagnostics (LOCAL CPU,
zero GPU).

## Verdict: PASS

All five sub-checks passed, plus the RNG-separation demonstration. The alpha=0
identity holds **bit-exactly** in fp64 (max abs param diff = 0.0 over 50
steps). No STOP condition was hit.

## Setup
- Environment: Python 3.12.3, torch 2.11.0+cu128 (run forced to CPU via
  `CUDA_VISIBLE_DEVICES=""` + `DEVICE="cpu"`; `torch.use_deterministic_algorithms(True)`),
  numpy 1.26.4, scipy 1.17.1. Local machine only — no cluster, no Slurm, no GPU.
- Duration: 1.0 s total compute (see `run.log` SUMMARY); wall including setup ~minutes.
- Resources used: local CPU only (authorized "light analysis" under the compute
  profile). No large artifacts; outputs are two small JSON files in `out/`.

## What Was Tested

The run's canonical filter copy was created first:
`~/pyg/optimizers/spectral_filter.py` → `experiments/exp-f02/src/spectral_filter.py`
(original untouched). One documented modification was made to the copy: the
per-step (k+1)×(k+1) `torch.linalg.eigh` in `_update_svd` is wrapped in a
CPU-fp64 fallback with a logged firing count (`self.eigh_fallback_count`),
transplanting the parent audit's Finding-5 patch pattern
(parent `audit/rerun-exp-004/src/spectral_optimizer.py`, "AUDIT-PATCH": fp32
cuSOLVER eigh fails stochastically under rank collapse; CPU-fp64 retry of the
same decomposition, cast back). The modification is documented in the file
header ("RUN-COPY MOD 1").

On a synthetic Numerai-shaped regression task (40 features, 40 eras × 128
rows, low-SNR linear signal + per-era shift + noise, target in [0,1]; 2-layer
MLP 40→32→1, p = 1,345; AdamW base optimizer; `normalize="none"` per parent
finding H7), the five sub-checks from plan.md were run: (a) a 200-step
filtered training run (hard top-k, rank 8, warmup 20) with the new A8/B3/D1
diagnostics module (`src/diagnostics.py`) plus a 7-config knob smoke covering
rank / decay / warmup / weighting hard+soft / alpha / soft_residual /
energy_threshold / adaptive none+effrank+gap; (b) fp64 50-step parameter
trajectories of [soft, alpha=0, soft_residual=True] vs plain AdamW on
identical pre-generated batches; (c) a planted-dominant-direction gradient
stream (g_t = ±3·u + 0.1·noise, sign-randomized so covariance centering keeps
u) fed to the filter at hard top-4 for 300 steps; (d) a monkeypatched
`torch.linalg.LinAlgError` forced on the fp32 eigh attempt at steps
{30, 31, 60} of a 100-step training run; (e) numerai_corr eval plumbing
(parent exp-001 implementation + zero-variance guard) on the synthetic eras
with a zero predictor, a seeded random predictor, and a positive control.

## Results

### Raw Output
```
(a) loss first20 0.36903 -> last20 0.06692 (falls: True); diagnostics sane: True
    19 rotation measurements, mean angle 0.0998 rad, max 0.3426 rad (all in [0, pi/2])
    kept-norm fraction in [0.4280, 0.9431]; realized k(t) = 8 throughout
    D1 valid-score hook: 8 entries, all finite (valid corr ~ +0.08)
    knob smoke: all 7 configs ok=True
(b) max abs param diff after 50 fp64 steps: 0.000e+00; bit-identical: True
(c) |cos(V[:,0], planted u)| = 0.976981 (need > 0.9)
    probe u+v_orth: cos(filtered, u) = 0.9945, kept-norm fraction = 0.7077
(d) eigh_fallback_count = 3 (expected 3); training completed 100 steps,
    final loss 0.0820, params finite; count surfaced in diagnostics records: 3
(e) zero predictor mean per-era corr +0.000000; random predictor +0.004623
    (40 eras); target-as-preds positive control +0.9882
(+) arm A vs arm B data order identical over 30 steps: True;
    negative control (arm draws from DATA generator) diverges at step 1
OVERALL: PASS (1.0s total)
```

### Metrics
| Metric | Target | Actual | Pass? |
|--------|--------|--------|-------|
| (a) 200-step filtered loss falls | last20 < first20 | 0.36903 → 0.06692 | Y |
| (a) diagnostics sane (cosine, kept-norm, angles) | angles ∈ [0, π/2], kept-norm ∈ [0,1] | max angle 0.3426 rad; kept-norm ∈ [0.428, 0.943] | Y |
| (a) all sweep knobs exercised | 8 knobs, no failure | 7 configs, all ok | Y |
| (b) alpha=0 identity max abs diff (fp64, 50 steps) | ≤ 1e-12 | **0.0 (bit-identical)** | Y |
| (c) planted-direction cosine | > 0.9 | **0.976981** | Y |
| (d) fallback firing count logged, training continues | 3 firings, no crash | 3 firings, finite loss/params | Y |
| (e) zero-predictor mean per-era corr | ≈ 0 | +0.000000 (exact, via zero-variance guard) | Y |
| (e) random-predictor null | \|mean\| < 0.06 | +0.004623 | Y |
| RNG separation: paired data order at same seed | identical | identical (and negative control diverges) | Y |

### Analysis

The integration surface is as advertised: the four-line usage pattern
(`backward(); filt.filter_grad(); base_opt.step()`) works unchanged, every
sweep knob the staged A4 allocation needs is reachable, and the filter's own
diagnostics dict carries `kept_rank`/`basis_rank` cleanly into the new
diagnostics module. The alpha=0 identity is not merely within 1e-12 but
bit-exact, which is the structurally expected outcome: with `alpha=0,
soft_residual=True` the reweighting term is `V @ ((w−1)·coeffs)` with `w`
exactly 1.0, and the norm-preserving rescale divides two bitwise-equal norms —
the filtering path adds exact zeros to the gradient. This means the identity
holds regardless of the precision of the internal eigh, which de-risks the
on-cluster fp32 re-assert in EU-1.

The planted-subspace check exercises the path the identity cannot see: the
streaming covariance recovered the planted direction at cosine 0.977 after 300
steps at decay 0.99, and the hard top-4 projection of a u+v_orth probe points
at u with cosine 0.995 while shedding ~29% of the probe norm — the filtering
path filters. The Finding-5 fallback transplant works as intended: three
forced fp32 eigh failures each fired the CPU-fp64 retry, were counted, printed,
surfaced in the per-step diagnostics records, and training continued to a
finite loss. The eval plumbing's null is validated (constant predictor → exact
0 via the zero-variance guard; independent random predictor → +0.0046 over 40
eras, within noise; target-as-predictions → +0.988), and seed pairing across
arms survives arm-B-style extra RNG consumption when drawn from a separate
`torch.Generator` — the negative control shows pairing breaks at the very
first step if arm randomness touches the data generator, confirming the
separation is load-bearing for the A6 paired-seeds design.

One sanity observation supporting the diagnostics module's correctness: for
the hard (pure orthogonal projection) path, per-step
`filtered_unfiltered_cosine` equals `kept_norm_fraction` to ~6 decimals
(mean 0.65797 both), which is the mathematical identity
cos(g, Pg) = ‖Pg‖/‖g‖ for orthogonal projectors — the two independent
computations agree.

## Unexpected Observations
- `energy_threshold=0.90` collapsed the retained basis to k=1 on this task,
  and both adaptive rules (`effrank`, `gap`) settled at realized k=2 — on a
  low-SNR task the gradient spectrum is top-heavy, so energy/adaptive configs
  can be far more aggressive than their `rank` cap suggests. The A2 realized-
  k(t) logging (already emitted per step by the diagnostics module) is exactly
  the visibility instrument for this on the real data.
- The soft alpha=1.0 config trains noticeably slower (final loss 0.219 vs
  ~0.06–0.07 for hard configs at 40 steps) — expected from eigenvalue-
  proportional downweighting, but a reminder that stage-2 alpha refinement can
  move VALID scores a lot.
- First harness attempt crashed on a test-harness bug (duplicate `warmup`
  kwarg in the knob-smoke loop, my code, not the filter); fixed and fully
  rerun. `run.log` is the complete clean rerun.

## Interface notes (for later cluster jobs / EU-1)
- **Gram/eigh dtype follows torch's *default* dtype**, not the model's: `gram
  = torch.zeros(k+1, k+1)` in `_update_svd`. In nominal fp32 runs the per-step
  eigh is fp32 on CPU (the gram is built on CPU even in GPU runs — S lives on
  CPU, V on the model device). The fp64 identity run therefore used
  `torch.set_default_dtype(torch.float64)`. The Finding-5 fallback guards the
  fp32 LAPACK path; its count is cumulative per filter instance and is NOT
  reset by `reset()`.
- **`filter_grad()` reassigns every `p.grad`** to a reshaped slice of a new
  flat tensor each filtered step — any code caching `p.grad` references
  (e.g. hand-rolled grad clipping held across steps) will silently read stale
  tensors. Use the diagnostics wrapper's before/after capture pattern instead.
- Covariance updates from step 1; filtering only activates at
  `step_count > warmup`. During warmup the diagnostics wrapper reports
  cosine = kept-norm = 1 (no-op), `filtering_active=False`.
- `energy_threshold` is honored only when `adaptive=="none"` (basis-truncating
  semantics); adaptive rules instead narrow `proj_k` while keeping the basis
  broad. `kept_rank` in the filter's diagnostics is the realized k(t).
- Soft weighting renormalizes to preserve ‖g‖, so `kept_norm_fraction` ≈ 1.0
  by construction on the soft path; distance-from-identity there is carried by
  the cosine, not the norm.
- **No torch-2.11-only APIs relied on**: the copy and the diagnostics module
  use `torch.linalg.eigh/qr/svdvals`, `torch.Generator`, `torch.linalg.LinAlgError`
  — all present in the cluster's pinned torch 2.5.1. EU-1's on-cluster alpha=0
  re-assert covers the residual version risk.
- Deliverables for later arm-B runs: `src/spectral_filter.py` (the run's
  canonical filter copy), `src/diagnostics.py` (A8/B3/D1: per-step cosine,
  kept-norm fraction, realized k(t), principal-angle rotation rate every Δ
  steps, `log_valid_score()` D1 hook, `summary()` with
  distance-from-identity), `src/numerai_eval.py` (per-era numerai_corr with
  zero-variance guard).

## Implications

What this tells us: the criterion was met — the p×p `SpectralGradientFilter`
integrates cleanly, its no-op control is exact, its filtering path is
demonstrably correct on a planted subspace, the mandated Finding-5 eigh
fallback is wired and observable, the eval plumbing's null is validated, and
seed pairing is robust to arm-specific RNG. Next steps: the run's canonical
filter copy and diagnostics module in `src/` are ready to be shipped to the
cluster experiment units (EU-1 pre-flight: timing, realized rank grid, and the
on-cluster fp32/2.5.1 alpha=0 re-assert; then the fold jobs), and the arm-C
B3 CPU simulation (Component #2) can consume `principal_angles()` from the
same diagnostics module.
