# Experiment 004 Results — Seeded Main Comparison + Mechanism Controls (the verdict experiment)

## Component Tested
The main deliverable: the pre-registered three-way verdict on the motivating
question — does the Spectral Optimizer (SpectralConsensusFilter wrapping tuned
AdamW) improve out-of-sample performance vs tuned AdamW alone on noisy
financial data (Numerai v5)? Plus the C4 random-subspace and GAF-style
mechanism-attribution controls and the F8 tail-era breakdown.

## Verdict: PASS (experiment returned a verdict category)

**Pre-registered verdict category: HURTS.**
**Final claim after mandatory F12 downgrade (binding, signatures recorded in
exp-003 before unblinding): "no evidence of benefit under the affordable
tuning budget."**

Headline paired comparison (filter_on − filter_off), 255 verdict-block eras,
3 seeds, moving-block bootstrap L=4, 10,000 resamples, 95% CI:

- mean per-era numerai_corr diff: **−0.00527**, CI **[−0.00886, −0.00181]**
  — CI excludes 0 and |mean| exceeds the frozen F3 threshold 0.00398.
- spearman variant: −0.00607, CI [−0.00927, −0.00292] (threshold 0.00391 —
  same category).
- per-seed diffs (paired, same-seed): s0 −0.00473 [−0.00929, −0.00034],
  s1 −0.00589 [−0.01007, −0.00172], s2 −0.00521 [−0.01069, +0.00023].
  All three point estimates negative; direction is seed-consistent.
- F10 corr-Sharpe (same joint block-bootstrap machinery): on 0.109 vs off
  0.309, diff −0.200, CI [−0.416, −0.004] — excludes zero, same direction.

## Setup

- **Cluster job**: Slurm job **6439**, one NVIDIA L40, `--qos=debug`,
  total job time **5.4 min** (torch 2.5.1+cu121). Full stdout in `run.log`
  (= `logs/slurm-6439.out`). Verdict analysis run locally on CPU
  (`src/analyze_verdict.py`; log `out/analyze_verdict.log`; full machine-
  readable output `out/verdict_analysis.json`).
- **Frozen inputs honored verbatim**: F1 blocks (train shard eras 0425–0574,
  150 eras; verdict eras 0971–1225, 255 eras; tuning block untouched here);
  F2 mode = hard, mp_factor 2.0; F6 within-era batches, B=1024; baseline
  config t07 (lr 1e-3, wd 1e-3, dropout 0.2, 2000 steps), MLP 197k params,
  705 medium features; F3 = 0.00398 (nc) / 0.00391 (sp); F5 L=4; seeds
  {0,1,2}, identical data order per seed across arms.
- **First unblinding**: this experiment performed the first and only
  evaluation on the verdict block; all arms/configs were frozen beforehand
  (spectral arm settings transferred from the prior project per F12 record:
  one pre-registered spectral configuration, no spectral tuning sweep).
- **Arms** (all on the same train shard, same seeds): filter_off (plain
  tuned AdamW, ~2.8 s/run), filter_on (SpectralConsensusFilter hard/2.0,
  ~60 s/run), C4 random-subspace control (random orthonormal sample-subspace
  with k(t) and update-norm ratio(t) trajectories matched to the same-seed
  filter_on run, ~16 s/run), GAF-style sign-agreement ablation
  (|sum sign| ≥ 2.0·√B per coordinate, ~20 s/run).

## Results

### Arm levels (mean per-era numerai_corr over the 255 verdict eras, 3-seed mean)

| Arm | numerai_corr | spearman | corr-Sharpe | per-seed nc |
|---|---|---|---|---|
| filter_off (tuned AdamW) | **+0.00641** | +0.00698 | 0.309 | +0.00530 / +0.00660 / +0.00732 |
| filter_on (spectral) | +0.00113 | +0.00090 | 0.109 | +0.00057 / +0.00071 / +0.00211 |
| C4 random-subspace | +0.00229 | +0.00112 | 0.178 | +0.00038 / +0.00217 / +0.00431 |
| GAF sign-agreement | −0.00042 | +0.00004 | −0.032 | −0.00151 / +0.00117 / −0.00091 |
| zero-predictor | +0.00063 | +0.00057 | — | (sanity: ≈0 ✓) |

### Paired comparisons (moving-block bootstrap, L=4, 95% CI, nc)

| Comparison | mean diff | 95% CI | category vs F3 |
|---|---|---|---|
| filter_on − filter_off (**headline**) | −0.00527 | [−0.00886, −0.00181] | hurts → F12 downgrade |
| c4_random − filter_off | −0.00412 | [−0.00794, −0.00036] | hurts |
| gaf − filter_off | −0.00682 | [−0.01011, −0.00352] | hurts |
| **filter_on − c4_random** | −0.00116 | [−0.00366, +0.00134] | no detectable difference |
| filter_on − gaf | +0.00155 | [−0.00081, +0.00392] | no detectable difference |

### Mechanism attribution (the load-bearing C4 result)

The spectral filter is **statistically indistinguishable from its norm- and
k-matched random-subspace control** (−0.00116, CI spanning 0). Under the
amended interpretation frame (exp-001's proof that eigenselection is exactly
target-independent for scalar-output MSE), this closes the loop: not only is
the selection target-blind by proof, its *effect on generalization* is no
different from projecting onto a random sample-subspace with the same kept-k
and update-norm trajectory. MP-eigenselection adds nothing measurable beyond
generic subspace projection on this task. All three filtered arms (spectral,
random-subspace, sign-agreement) hurt by similar amounts — the harm is
attributable to per-sample-agreement-style update restriction/rescaling
itself, not to the spectral mechanism specifically.

### F12 downgrade — why it applies and its direction nuance

The downgrade was pre-registered as binding in exp-003 before any verdict
data existed. The operative signature is on the **spectral side**: the
matched-compute tuning budget afforded ~1 spectral configuration vs AdamW's
12 trials, so "hurts" cannot be distinguished from "under-tuned spectral
arm" at this budget. Honest note on the other recorded signature: the
*baseline's* boundary/improving-trend signature (wd, dropout at grid max)
implies the baseline may itself be under-tuned — fixing that would, if
anything, *strengthen* the hurts direction. The downgrade is therefore
conservative and one-sided by design: the reportable claim is **no evidence
of benefit under the affordable tuning budget**, with the raw "hurts"
numbers reported alongside.

### F9 sanity check (regime drift — reportable, not a failure)

Realized verdict-block baseline nc **+0.00641** sits BELOW the tuning-block
sanity band [0.0087, 0.0522]; the leakage gate (0.0696) did NOT fire. The
verdict eras (0971–1225, later period) are a genuinely harder regime than
the tuning eras — consistent with era-wise non-stationarity, and exactly why
F1 separated the blocks. Consequence worth stating: the frozen absolute F3
threshold (0.00398, set as 25% of the tuning-block 3-seed baseline) is ~62%
of the *realized* verdict-block baseline — the relative bar turned out
stricter than intended. This does not affect the verdict (the CI excludes
zero regardless; the mean also exceeds the threshold), but the F3
tuning-vs-verdict calibration gap is a limitation to carry forward.

### F8 tail-era breakdown (descriptive; iid bootstrap within buckets)

By baseline per-era corr quartile (filter_on − filter_off, nc):

| Quartile (baseline corr) | n | mean diff | ~95% CI |
|---|---|---|---|
| Q1 (worst eras) | 64 | **+0.01902** | [+0.01571, +0.02271] |
| Q2 | 63 | +0.00048 | [−0.00183, +0.00292] |
| Q3 | 64 | −0.01084 | [−0.01365, −0.00796] |
| Q4 (best eras) | 64 | **−0.02966** | [−0.03343, −0.02606] |

A strong monotone pattern: the filter "helps" precisely where the baseline
does worst and hurts most where the baseline does best. **Caveat (recorded
in advance in the analysis code): bucketing on the baseline's own realized
per-era corr mechanically induces regression-to-the-mean in the paired
diff, so much of this gradient is expected artifactually; descriptive
only.** The complementary split by per-era target dispersion (which is not
outcome-conditioned) shows a roughly uniform hurt across quartiles
(−0.0022 to −0.0070, three of four CIs excluding 0 or nearly so) — i.e.,
no evidence of the Feldman-style pattern that harm concentrates in
high-dispersion (tail) eras. The variance-reduction story ("filter trades
tail performance for stability") is not supported by the dispersion split;
the corr-Sharpe also went DOWN (−0.200), so the filter did not buy
stability either.

### Engagement diagnostics (during verdict runs — filter did real work)

- filter_on: selective on **100% of logged steps** (0 < k < B); k_median
  1–4 of 1024; kept_energy_frac median 0.968–0.980 (< the 0.995 no-op
  guard); update-norm ratio ||filtered||/||mean_grad|| median **9.5–13.9**
  (range up to ~35) — the filter predominantly **amplified** the update
  rather than shrinking it, confirming exp-001's observation that this
  mechanism is not norm-shrinkage regularization. C3 cos-vs-mean median
  −0.29 to +0.72 — far from mean-gradient degeneracy.
- The pre-authorized FAIL condition (silent disengagement: k≈0 or
  kept-norm≈1 throughout) did NOT occur. The verdict is attributable to an
  engaged, selective mechanism.
- Notable dynamics: k collapses from ~52 at step 0 to a median of ~1
  eigendirection for most of training, with the update norm ~10× the mean
  gradient — effectively the filter replaces the averaged gradient with a
  single amplified dominant eigendirection of the per-sample gradient
  similarity matrix. C4 matched exactly this (k, ratio) trajectory, which
  is what makes the null spectral-vs-random comparison meaningful.

### Timings

Per-run train times on one L40: filter_off ~2.8 s, filter_on ~60 s,
c4_random ~16 s, gaf ~20 s; total job 5.4 min (well under the 25-min
per-job cap; ~7 min of the 12.7-min budget projection unused).

## Unexpected Observations

1. **The C4 null is the sharpest finding**: spectral MP-eigenselection ≈
   random subspace at matched k/norm — the "consensus" machinery
   contributes nothing measurable beyond its generic projection/rescaling
   side effects on this task. Together with exp-001's target-independence
   proof, the two results give a consistent mechanistic account.
2. All three filtered arms hurt by similar magnitudes (−0.004 to −0.007)
   despite very different mechanisms (eigenspectral, random-subspace,
   coordinate sign-agreement) — per-sample-agreement-style update
   modification as a class degrades OOS performance in this regime.
3. The filter amplifies (ratio ~10×) rather than shrinks — the a-priori
   plausible "implicit regularization by norm shrinkage" story is inverted
   here.
4. Regime drift between tuning and verdict blocks (baseline +0.0196 →
   +0.0064) — large, and itself a caution for any absolute pre-registered
   threshold on this dataset.

## Implications

The pre-registered protocol returned a decisive answer to the motivating
question, with the F12-mandated conservative wording: **no evidence that the
Spectral Optimizer improves out-of-sample performance on noisy financial
data under the affordable tuning budget; at the pre-registered operating
point it significantly degraded it** (−0.00527 mean per-era corr vs a
+0.00641 baseline, i.e., ~82% of the baseline's edge erased; corr-Sharpe
−0.200). The mechanism-attribution controls rule out the interesting
alternative explanations: the effect is not spectral-specific (C4 null) and
not a stability trade (Sharpe down, no dispersion-tail pattern).

Combined with exp-001's banked proof (eigenselection is target-independent
for scalar MSE), the run's overall story is coherent: on low-SNR financial
regression, per-sample gradient "consensus" measures Jacobian/factor
structure, not signal; filtering updates through it is target-blind
subspace projection with amplification, and that intervention hurts.

Nothing here triggers a kill criterion or retry: this is the pre-authorized
write-up path (verdict returned). exp-005 (sequence-arm architecture
consistency) remains authorized (exp-002 PASS, budget room) and tests
whether the negative transfers across architectures.
