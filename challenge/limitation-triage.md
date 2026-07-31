# Design-Time Limitation Triage (Step 6)

**Run**: 2026-07-31-spectral-optimizer-for-noise-reduction-on-financial-timeseri
**Inputs**: challenge/assumption-analysis.md (A#), challenge/mentor-review.md (M#), challenge/pre-mortem.md (P#)
**Rubric**: each residual weakness classified against the compute profile (MATS free partition, ≤5 experiments, <30-min jobs) and remaining budget as **fix-now-free** (protocol/pre-registration, zero compute), **fix-now-cheap** (folds into existing gate jobs or shares an array job), or **future-work** (carries a resource ask to Step 11).

**Mentor verdict**: MAJOR_REVISIONS — construct sound, no rethink; revisions are structural but cheap. All MAJOR_REVISIONS items land in the fix-now tiers below and are folded into the plan as **binding amendments**. Steps 7 and 9 MUST read this file; the amendments modify experiment pass criteria and arm composition without changing the lambda ordering.

## Convergence note

The three independent passes converged on the same root risks: (1) selection/inference contamination at a 0.005-corr decision threshold (M1, A7, P1, P3); (2) engagement diagnostics validating execution but not interpretation (A1, A3, M3, P2); (3) the ablation not isolating spectral selection from update shrinkage (A2, M-simpler-alternatives, P2); (4) the cut cascade stripping a null of its meaning (M-cut-order, A-rec5, P5). Convergence from independent passes raises confidence these are the real load-bearing gaps.

## Fix-now-free (binding amendments — zero compute, pre-registered here)

| # | Amendment | Source | Binding effect on plan |
|---|-----------|--------|------------------------|
| F1 | **Tuning/verdict era separation.** Split validation-side eras into a tuning block and a temporally later, embargoed final-verdict block. No tuning, mode selection, or peeking touches the verdict block. Power analysis (#3) is recomputed on the verdict block's era count. | M1 (load-bearing) | Modifies components #3, #8, #9 |
| F2 | **Pre-registered threshold-mode rule.** Threshold mode (hard/soft/variance) is fixed at the engagement gate (#1) from engagement diagnostics ONLY — before any out-of-sample performance is seen. It is not tuned on outcome. | M2 | Modifies #1 pass procedure |
| F3 | **Relative practical-significance threshold.** Verdict threshold = min(0.005, 0.25 × realized tuned-baseline mean per-era corr), pre-registered before unblinding the comparison. Prevents a low-landing baseline from silently turning 0.005 into a ~70% relative-effect requirement. | P1, A7 | Modifies success-criteria verdict definitions |
| F4 | **Joint go/no-go gate after #8 and #3.** Proceed to the seeded comparison only if the (block-)bootstrap shows P(some verdict category reachable) ≥ 0.6 given the realized baseline level and verdict-block era count. Otherwise re-scope the endpoint BEFORE spending slots. | P1 | New gate between pre-experiment gate and Step 9 |
| F5 | **Era-autocorrelation-valid inference.** Embargo ≥ ceil(target_horizon_days / era_spacing_days) eras; ALL CIs (including the #3 power simulation) use a moving-block bootstrap with block length ≥ target-horizon overlap. Check lag-1 autocorrelation of per-era corr as soon as #8 output exists. | P3 | Modifies #3, #9 |
| F6 | **Batch composition is an explicit design decision.** Within-era vs mixed-era minibatches chosen and justified in the experiment plan; #1 engagement diagnostics logged under BOTH compositions (same debug job, two configs). | M4, A1 | Modifies #1 |
| F7 | **Cut order inverted and made conditional.** Mechanism-discriminating controls (random-subspace norm-matched control, then GAF-style ablation) outrank the sequence arm. New cut order under pressure: Muon → GBT → sequence arm → GAF → random-subspace control (never seeds, never matched tuning, never the verdict-block separation). If early diagnostics point to a null, the discriminating ablation is protected ahead of everything optional. Check array-job co-scheduling before treating any control as a full slot. | M3, A-rec5, P5 | Modifies #4 cut arithmetic; resolves success-criteria/decomposition inconsistency |
| F8 | **Tail-era breakdown promoted to minimum-viable deliverable.** A null mean with significant era-quantile heterogeneity reads "heterogeneous effect", not "no effect". Costs re-slicing of existing predictions only. | A9 | Modifies minimum-viable deliverable definition |
| F9 | **Sanity/leakage bands recalibrated to current data.** During #9, recompute the corr sanity band and the leakage gate from current Numerai v5 example-model validation stats, not 2021 forum numbers. | A8 | Modifies #8/#9 gate values |
| F10 | **Corr-Sharpe escape hatch closed.** The alternative "clear corr-Sharpe improvement" route to a 'helps' verdict is tied to the same paired block-bootstrap CI machinery (CI excluding zero) or dropped. No untested secondary route to a verdict. | M-minor | Modifies success-criteria verdict definitions |
| F11 | **Optimizer identity fixed in writing.** "The Spectral Optimizer" = `spectral_optimizer.py` (exact B×B MP-threshold SpectralConsensusFilter). A #7-forced swap to the streaming rank-k variant is a SCOPE CHANGE requiring a novelty re-check and redefined engagement diagnostics — not a silent fallback. | A4 | Modifies #7 fallback semantics |
| F12 | **Tuning match defined in compute, not trial count.** Spectral-arm hyperparameter priors transferred from the prior repo (~/pyg/optimizers) as the search center, documented as part of the matched-budget argument. Watch for best-config-on-grid-boundary / non-monotone-sweep under-exploration signatures; if present with a "hurts" verdict, downgrade to "no evidence of benefit under the affordable tuning budget". | P4 | Modifies #8 and comparison protocol |
| F13 | **Sequence-arm claim renamed unless dataset is held fixed.** If the sequence arm runs on OHLCV (not within-era Numerai), either add a cheap MLP-on-OHLCV arm or claim "second setting", not "architecture consistency". | A6 | Modifies stretch-goal claim wording |

## Fix-now-cheap (folds into existing gate jobs; minutes of CPU/GPU)

| # | Amendment | Source | Cost |
|---|-----------|--------|------|
| C1 | **Spiked-covariance unit test** in the #6 local smoke: (a) i.i.d. noise → filter keeps ≈0; (b) planted spike → filter keeps it; (c) i.i.d. noise + correlated zero-signal confound → measure whether the filter keeps the confound. Verifies the diagnostics are RIGHT, not merely emitted. | A3 | ~30 min local CPU |
| C2 | **Permuted-target null-spectrum control** in the #1 debug job: same diagnostic with permuted targets (pure noise by construction); compare spectra to calibrate the MP bulk edge on this data's correlation structure. | M3 | Same debug job |
| C3 | **Two cosine diagnostics** added to #1 logging: filtered-update vs mean-gradient cosine, and filtered vs unfiltered update cosine. Pivot rule: if mean-gradient cosine > ~0.95 for all modes at both batch sizes, redesign around characterizing the degeneracy (filter ≈ mean-gradient smoothing) instead of filter-vs-baseline. | P2 | One-line logging |
| C4 | **Mechanism controls as co-scheduled arms**: random-subspace control matched to the measured kept-norm fraction (distinguishes "MP-selected directions matter" from "any gradient energy reduction regularizes"), and/or a mean-gradient-smoothing / larger-effective-batch control. Share array jobs with the main comparison where possible (GAF-style filtering is cheaper than the spectral arm). | A2, M-simpler, P2 | Co-scheduled arms, no new slots expected |
| C5 | **Era-identity probe of the kept subspace** in the #1 debug job: does the kept component predict within the current era much better than across eras? Direct check on "coherent = era factor, not signal". | A1 | Same debug job |

## Future-work (carry to Step 11 with resource asks)

| # | Item | Source | Resource ask |
|---|------|--------|--------------|
| W1 | Full-scale (non-subsampled) Numerai verdict at large batch (B≈4096): the B-conditionality means the claim here is conditional on B and subsample scale. | A5, P1 theme | Multi-hour multi-GPU jobs (exceeds 30-min cap); est. 10-20 L40-hours or A100-class |
| W2 | True architecture-consistency arm: sequence model AND MLP on the same dataset at full scale (within-era Numerai framing or OHLCV with both architectures). | A6 | 2-4 additional experiment slots' worth of GPU |
| W3 | Muon and GBT baseline context (designated cuts if budget pressure fires). | Success-criteria, P5 | ~2 experiment slots |
| W4 | MP-null theory under cross-sectional correlation: analytical or simulation characterization of the shifted bulk edge for per-sample financial gradients (beyond the C2 permutation control). | M3, A3 | CPU-heavy simulation study; no GPU |
| W5 | Era-heterogeneous effect modeling beyond the tail-era breakdown (regime-conditional verdicts). | A9 | Re-analysis + possibly more seeds |

## Accepted risks (stated, not fixed)

- **Residual irreducible risk** (P-residual): even with all mitigations, ~1-in-4 to 1-in-3 chance the deliverable is a caveated "could not determine" below the primary venue. Accepted knowingly — boundary/degeneracy findings are explicitly priced as deliverables in the success criteria.
- **A5 (B-conditionality)**: accepted; the claim is stated as conditional on B and subsample scale (see F13/W1).
- **A10-A12** (data availability, venue fit, lambda-table independence): accepted as-is.
