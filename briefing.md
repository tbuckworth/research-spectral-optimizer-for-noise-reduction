# Research Briefing: Spectral Optimizer (for noise reduction) on Financial Timeseries Data

**Run ID**: 2026-07-31-spectral-optimizer-for-noise-reduction-on-financial-timeseri
**Status**: complete — honest negative result (audit exit: true-null)
**Date**: 2026-07-31

## Topic & Motivation

Can the existing Spectral Optimizer — a gradient-filtering wrapper around
Adam/AdamW from the prior project at `~/pyg/optimizers` that eigendecomposes
the B×B inter-sample gradient similarity matrix and keeps only directions
above a Marchenko–Pastur-style threshold — reduce the effect of noise when
training on large, noisy financial time-series data? Concretely: does it
improve out-of-sample predictive performance vs tuned AdamW, and is any
effect consistent across a plain MLP and a recurrent architecture?

The prior project's strongest result was label-noise robustness (MNIST @ 90%
label noise: ~80% vs ~37% for Adam), but it also found the filter hurts when
signal is weak and not gradient-dominant — so transfer to the low-SNR
financial regime was genuinely uncertain in both directions. Dataset: Numerai
v5 public tournament data (large, noisy, obfuscated financial features,
era-based out-of-sample evaluation).

## Literature Findings

Top sources (28 curated references across academic / lab-blog / community
channels):

1. **Gradient Agreement Filtering (Chaubard et al. 2024)** — closest
   mechanism; differentiator is MP-spectral thresholding vs simple agreement.
2. **Coherent Gradients (Chatterjee & Zielinski 2020)** — theoretical
   foundation for why consensus filtering should work.
3. **Bouchaud et al. RMT-in-finance review** — canonical Marchenko–Pastur
   covariance cleaning in quant finance; applied to RETURNS covariance, never
   training gradients. The transposition appeared unpublished.
4. **Gorishniy et al. 2026 tabular-optimizer benchmark** — methodological
   template (matched tuning budgets); flags Muon beating AdamW.
5. **Feldman 2019 long-tail memorization** — the sharpest counter-hypothesis:
   consensus filtering could suppress rare genuine signal in heavy-tailed data.

Key gap identified: no published work combines per-sample gradient
RMT/eigenspectrum filtering with financial prediction, across three
independent search channels.

## Novelty

**Verdict: NOVEL** (with caveats). Closest existing work is Gradient
Agreement Filtering — same intervention level but pairwise orthogonality
checks on image classification, not MP-spectral thresholding on financial
regression. Caveat: novelty rests on absence of evidence in an active niche;
the contribution is the transfer verdict, not the optimizer itself.

## Experiment Design

Steinhardt lambda table (9 components, packed into 5 experiments along
dependency waves):

| Component | P_success | T (hrs) | lambda | Outcome |
|---|---|---|---|---|
| Spectral engagement on financial gradients | 0.40 | 0.75 | 1.22 | FAIL on C2 conjunct only (banked mechanistic finding) |
| Sequence-arm vmap feasibility (GRU) | 0.35 | 1.0 | 1.05 | PASS |
| Statistical power of paired per-era design | 0.60 | 0.5 | 1.02 | PASS |
| Design fits 5-experiment / 30-min budget | 0.80 | 0.25 | 0.89 | PASS |
| OHLCV fallback dataset | 0.80 | 0.5 | 0.45 | NOT NEEDED |
| Spectral Optimizer regression integration | 0.75 | 0.75 | 0.38 | PASS |
| vmap throughput on L40 | 0.85 | 0.5 | 0.33 | PASS |
| Baseline sanity: tuned AdamW in sane corr band | 0.70 | 1.5 | 0.24 | PASS |
| Numerai data + era-purged protocol | 0.85 | 1.0 | 0.16 | PASS |

5 experiments planned, 5 executed (~15 min total GPU on free-partition L40s,
plus ~9 min audit re-execution).

## Challenge Highlights

- **Critical assumptions**: (1) the MP threshold's null assumes i.i.d.
  samples, but financial per-sample gradients have cross-sectional factor
  structure — "engaged" can be spurious; (2) the filter-on/off ablation
  doesn't isolate the mechanism, since filtering also changes update norm
  (fixed with the C4 norm/k-matched random-subspace control — which proved
  load-bearing); (3) tuning/verdict selection contamination without era
  separation (fixed with F1 split).
- **Mentor-review verdict**: MAJOR_REVISIONS — all protocol-level and cheap;
  folded in as 13 binding amendments + 5 additions; construct-validity gate
  passed (outcome genuinely uncertain in both directions).
- **Top failure scenario**: selection/inference contamination — no
  tuning/verdict era separation plus era-autocorrelated targets breaking
  naive bootstrap CIs (mitigated by F1 split + F5 moving-block bootstrap).

## Results

| # | Component | Result | Key Finding |
|---|-----------|--------|-------------|
| exp-001 | Gate bundle (integration, data, throughput, engagement) | infra PASS; #1 FAIL on C2 only | PROOF: for scalar-output MSE the engagement eigenspectrum is exactly target-independent — "consensus" measures Jacobian/factor structure, not signal |
| exp-002 | Sequence-arm vmap feasibility | PASS | vmap(grad) through hand-rolled GRU matches loop reference to 1.7e-06 |
| exp-003 | Baseline sanity + power + F4 gate | PASS | Tuned AdamW +0.0196 in sane band; P(verdict reachable)=0.91 → GO; F3 frozen at 0.00398 |
| exp-004 | Seeded main comparison + mechanism controls | PASS (verdict returned) | HURTS → F12 downgrade: filter −0.00527 mean per-era corr, CI [−0.00886, −0.00181]; spectral vs random-subspace control: −0.00116 CI [−0.00366, +0.00134] — indistinguishable |
| exp-005 | GRU architecture-consistency arm | PASS | Verdict replicates: −0.00484 CI [−0.00758, −0.00211]; spectral-vs-random null replicates |

**Why the approach doesn't work**: the run's two co-primary mechanistic
findings jointly explain the negative. (1) For scalar-output MSE, per-sample
gradients are ±(residual sign) × a target-independent Jacobian direction, so
the filter's eigenselection is provably blind to the targets — it cannot be
selecting "signal directions". (2) Empirically, MP-eigenselection is
statistically indistinguishable from a norm/k-matched *random* subspace
projection on both architectures, and all filtered arms (spectral, random,
GAF-style) hurt similarly. The label-noise "coherence amplifier" story does
not transfer to low-SNR financial regression. Reportable claim (mandatory
F12 downgrade, since the spectral arm afforded ~1 tuning trial vs AdamW's
12): **no evidence of benefit under the affordable tuning budget**, with the
raw degradation numbers reported alongside. An independent audit re-executed
the main experiment with fresh seeds and an independently built evaluation
shard and reproduced everything (6-seed pool −0.00537 CI [−0.00853, −0.00222]).

## Paper Abstract

Per-sample gradient "consensus" filtering — eigendecomposing the B×B
inter-sample gradient similarity matrix and keeping only directions above a
Marchenko–Pastur-style threshold — has shown strong label-noise robustness in
classification. We ask whether this mechanism reduces the effect of noise
when training on large, low signal-to-noise financial data. Under a fully
pre-registered protocol on the Numerai v5 dataset (era-purged tuning/verdict
split, frozen thresholds, moving-block-bootstrap inference, compute-matched
tuning), the answer is negative: we find no evidence of benefit under the
affordable tuning budget, and at the pre-registered operating point the
filter significantly degraded out-of-sample performance on both architectures
tested (MLP: −0.00527 mean per-era correlation, 95% CI [−0.00886, −0.00181];
GRU: −0.00484, CI [−0.00758, −0.00211], against a +0.0064 baseline edge). Two
mechanistic contributions explain why the label-noise story does not
transfer. First, we prove that for scalar-output MSE the filter's engagement
eigenspectrum is exactly target-independent: per-sample gradients reduce to
±(residual sign) × a target-independent Jacobian direction, so "consensus"
measures feature/factor structure, not signal (verified in fp64 to 8.9e-16;
independently re-verified). Second, matched controls show MP-eigenselection
is statistically indistinguishable from a norm- and k-matched random-subspace
projection on both architectures. An independent audit re-executed the main
experiment with fresh seeds and an independently constructed evaluation shard
and reproduced the result. Claims are scoped to a subsampled, low-edge regime.

## Surprises & Non-Obvious Findings

- **The target-independence proof** (exp-001): the C2 real-vs-permuted-target
  diagnostic failed *by mathematical proof*, not noise — for scalar-output
  MSE the entire engagement eigenspectrum is identical under any target
  permutation (verified fp64 to 8.9e-16). The prior project's "consensus on
  signal" interpretation cannot apply to this loss class at all.
- **The filter AMPLIFIES rather than shrinks**: update norm ~10x the mean
  gradient on the MLP (ratio > 1 on 25–70% of steps) — the opposite of the
  implicit-regularization-by-shrinkage hypothesis.
- **Different engagement regime, same harm**: on the GRU the filter engages
  completely differently (k~60, mild attenuation) vs the MLP (k~1, ~10x
  amplification) yet produces near-identical degradation and the same
  spectral-vs-random null — strengthening the "generic subspace
  projection/rescaling, not spectral selection" account.
- **No Feldman-tail pattern**: the unconditioned target-dispersion split
  shows uniform hurt — the long-tail memorization counter-hypothesis did not
  manifest as tail-specific behaviour.
- **Audit-only discovery**: the producer code crashes stochastically by seed
  (`linalg.eigh` fp32 cuSOLVER non-convergence under the documented
  rank-collapse regime) — the original runs were 6-of-6-seed lucky-complete.
  Disclosed in the paper; results unaffected (fp64 fallback reproduced all).
- **Regime drift**: the verdict-block baseline (+0.0064) fell below the F9
  sanity band calibrated on tuning eras — claims are scoped to this low-edge
  subsampled regime.

## Open Questions & Suggested Follow-Ups

- **Full-scale verdict run**: the pre-registered result is at a subsampled,
  low-edge operating point (B≈1024, subsampled eras). A B≈4096 full-era run
  (~10–20 L40-hours or A100-class, ~$50–100 cloud — exceeds the free
  partition's 30-min cap) tests whether the null/harm persists at scale.
- **Loss classes where the theorem doesn't bind**: the target-independence
  proof is specific to scalar-output MSE. Multi-output heads, ranking, or
  corr-aligned losses make the eigenspectrum target-dependent again — the
  mechanism could in principle work there; this is the cheapest route to
  rescuing the approach.
- **Small closure bundle** (~25 min on one L40): GRU re-tune (~10 GPU-min,
  removes the t07-fallback-config limitation), FNC metric run, C5
  era-identity probe, and additional C4 seed pairs to tighten the ±0.004
  equivalence resolution.
