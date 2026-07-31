# Assumption Analysis

**Run**: 2026-07-31-spectral-optimizer-for-noise-reduction-on-financial-timeseri
**Scope note**: The decomposition already prices nine component risks via P_success. This analysis targets what the framing takes for granted *around* those components — assumptions baked into the pass criteria, the ablation design, and the identity of the thing being tested. Overlap with decomposition risks is flagged where it exists.

## Summary

Twelve load-bearing assumptions found: 4 critical (low confidence, high impact), 5 moderate, 3 background. The most consequential cluster is a single unexamined premise inherited from the prior project: that **gradient-coherent = generalizable signal** in a data regime where the samples are cross-sectionally correlated. That premise is what the MP threshold's null model, the spectral-engagement pass criterion (component #1), and the "null means mechanism engaged but didn't help" interpretive claim all silently rest on — and financial data is close to a worst case for it. A second independent critical assumption is that the filter-on/filter-off ablation isolates *noise filtering* rather than *update shrinkage*.

## Critical Assumptions (Low confidence, High impact)

### 1. "Gradient-coherent component = generalizable signal" transfers to cross-sectionally correlated data

- **Category**: Theoretical
- **Confidence**: Low
- **Currently assumed because**: The mechanism was validated on MNIST label noise, where samples are i.i.d. and the noise is *independent per sample* — there, the only thing many samples can agree on is real signal. The Coherent Gradients framing and the entire "coherence amplifier" characterization carry this over unexamined.
- **What changes if wrong**: In financial data the noise is *not* sample-independent. Samples within an era share common-factor exposure and era-wise target normalization; a transient regime factor produces gradient directions that many samples in a batch agree on, yet which do not generalize across eras. In that case the filter does not suppress noise — it **amplifies era-specific noise**, and the top eigendirections are exactly what FNC exists to neutralize. A "hurts" verdict could then be true but for a different mechanism than the Feldman long-tail story the write-up is primed to tell; a "helps" verdict on in-band validation eras could reflect regime-persistence luck. Note the design has not even specified **batch composition** (within-era vs. mixed-era minibatches), and this assumption's failure mode is drastically different in the two cases — this is an unmade design decision hiding inside an unstated assumption.
- **How to test**: (a) Specify batch composition explicitly and run the #1 engagement diagnostics under both within-era and mixed-era batching — if kept-eigendirection count or norm fraction changes materially between the two, the "coherent = signal" reading is confounded by era structure. (b) Cheap add-on to the diagnostics run: correlate the kept-subspace update with era identity / dominant features (does the kept component predict *within* the current era much better than across eras?). (c) The tail-era breakdown already planned partially covers the consequence but not the mechanism.
- **Relevant evidence**: `numerai_forum2021_fnc` (high raw corr with low FNC predicts drawdowns — the domain's own statement that cross-sectionally coherent structure is often not signal); Bouchaud/Potters RMT literature — the *reason* returns-covariance cleaning works is that common factors dominate the top eigenvalues, i.e., the domain's known top eigendirections are factors, not idiosyncratic truth.

### 2. Filter-on vs. filter-off at identical base config isolates the noise-filtering mechanism

- **Category**: Methodological
- **Confidence**: Low
- **Currently assumed because**: Success criteria call this "the cleanest isolation of the mechanism" and "the single most load-bearing comparison." It controls for hyperparameter luck, which is real — but it does not control for what the filter *also* does besides selecting directions.
- **What changes if wrong**: Projecting out eigendirections shrinks the gradient norm and reshapes AdamW's second-moment statistics; the filtered arm effectively trains with a different implicit learning rate, trust region, and weight-decay-to-update ratio. On a low-SNR task, *any* update shrinkage acts as regularization. A "helps" verdict would then be attributable to "smaller, smoother updates help on noisy data" (already known from the Adam-robustness literature) rather than to spectral selection specifically — the paper's central claim. The GAF ablation does not fix this; GAF also shrinks updates.
- **How to test**: Add a **random-subspace control**: keep a random k-dimensional gradient subspace (or randomly selected directions) matched to the filter's measured kept-norm fraction, same base config. Cost is one arm at the same per-run price as the GAF ablation — and it is arguably *higher* value than the GAF ablation, because it distinguishes "spectral selection matters" from "throwing away gradient energy matters." If budget forces a choice, this control supports the core claim; GAF supports only the novelty positioning.
- **Relevant evidence**: `yu2026signheavytailed`, `xie2022overlookedstructure` — Adam-family robustness comes precisely from update normalization/shaping, so norm-shaping confounds are not hypothetical in this regime.

### 3. Passing the spectral-engagement criterion means the mechanism is engaging *correctly*

- **Category**: Methodological
- **Confidence**: Low
- **Currently assumed because**: Component #1's pass criterion (kept directions strictly between 0 and B, norm fraction in [0.1, 0.9]) is treated as establishing that a downstream null reads "mechanism engaged but didn't help, not implementation no-op." That inference needs the MP threshold to be a *calibrated* null.
- **What changes if wrong**: The Marchenko-Pastur law is derived for independent samples. With cross-sectionally correlated per-sample gradients (assumption #1), eigenvalues exceed the MP bulk *spuriously* — the diagnostic will report healthy engagement while the filter keeps factor/era directions. The interpretive backbone of the whole verdict structure ("every verdict requires spectral engagement diagnostics") is then weaker than stated: engagement diagnostics can rule out a no-op, but they cannot certify the filter is separating signal from noise. The pivot deliverable for a #1 FAIL ("no gradient-coherent signal separable from the MP bulk") has the mirror-image problem — an apparent-engagement result does not license "coherent signal exists."
- **How to test**: A ~30-minute local CPU unit test that should be added to the #6 smoke: synthetic spiked-covariance gradients — (a) pure i.i.d. noise (filter should keep ≈0), (b) planted spike (filter should keep the planted direction), (c) i.i.d. noise plus a *common correlated component with zero mean effect on the target* (does the filter keep the confound?). Case (c) directly measures the miscalibration this assumption hides. This also independently verifies the MP-threshold implementation, which nothing in the current gate does — the #6 pass criterion checks that loss falls and diagnostics are *emitted*, not that they are *right*.
- **Relevant evidence**: Standard RMT: MP edge shifts under sample correlation. The prior repo's MNIST validation cannot have exposed this because MNIST batches are effectively i.i.d.

### 4. The verdict is about *one* well-defined optimizer, despite a pre-authorized variant swap

- **Category**: Independence / Scope
- **Confidence**: Medium (that the swap is needed is unlikely — P(#7)=0.85 — but the assumption that the swap is *harmless* is low-confidence)
- **Currently assumed because**: Component #7's fallback quietly substitutes `weight_cov_optimizer_v2.py` (streaming rank-k covariance filter) for `spectral_optimizer.py` (exact B×B MP-threshold filter) as "a designed substitute."
- **What changes if wrong**: These are different algorithms. The novelty verdict, the positioning against GAF, and the differentiation analysis all rest specifically on **full similarity-matrix eigendecomposition with an MP threshold**. If throughput forces the streaming variant, the paper's tested mechanism is no longer the mechanism its novelty claim describes — and the engagement diagnostics (MP bulk, eigendirections kept) may not even be well-defined for the rank-k streaming filter. The swap is a scope change, not a fallback.
- **How to test**: Decide now, in writing, which variant *is* "the Spectral Optimizer" for this study. If the streaming variant becomes primary, re-check the novelty positioning (rank-k gradient-covariance filtering is closer to existing preconditioner/low-pass work, e.g., K-FAC lineage and DP-PMLF) and redefine the engagement diagnostics before any verdict run.
- **Relevant evidence**: `novelty-assessment.md` differentiators are MP-spectral-specific; `martens2018kfacrnn`, `xu2025dppmlf` are nearer neighbors to the streaming variant than to the exact one.

## Moderate Assumptions (Medium confidence or Medium impact)

### 5. A verdict at B ∈ {256, 1024} on subsampled data answers the question about "large, noisy financial time-series data"

- **Category**: Scaling
- **Confidence**: Medium
- **Currently assumed because**: Subsampling eras/features is pre-authorized for budget fit; batch size is treated as an engineering parameter.
- **What changes if wrong**: For a consensus filter, B is not an engineering parameter — it *is* the sample size of the covariance estimate, and the MP bulk edge scales with B/p. The filter could genuinely help at B=4096 and be degenerate at B=256 (or vice versa). The honest claim is conditional on B and the subsample; the motivating question ("large" data) is at a different point in that space.
- **How to test**: Already partially planned (component #1 tests two batch sizes). Extend: report engagement *and* the verdict's point estimate at both B values rather than picking one; state the B-conditionality in the claim.
- **Relevant evidence**: `mccandlish2018gradientnoisescale` — gradient SNR is explicitly batch-size-dependent; the prior repo's results are at unspecified-here B and may not pin down the transfer.

### 6. The sequence arm tests *architecture* consistency

- **Category**: Methodological / Scope
- **Confidence**: Medium
- **Currently assumed because**: The success criteria phrase the second arm as an "architecture-consistency statement," but the likely realization is GRU-on-OHLCV vs. MLP-on-Numerai — architecture and dataset change simultaneously.
- **What changes if wrong**: If the two arms disagree, nothing attributes the disagreement to architecture vs. dataset (different SNR, different noise structure, different protocol built from scratch). The "consistent across architectures" claim would be unsupported even with both arms succeeding.
- **How to test**: Either (a) prefer the within-era sequence framing on Numerai (same dataset, different architecture), accepting weaker sequence structure, or (b) if OHLCV is used, add the MLP on OHLCV too (cheap — MLP runs are the fast ones) so architecture is varied within a dataset. Otherwise rename the claim "second setting," not "second architecture."
- **Relevant evidence**: Design constraint already discovered in Step 2 (era-reset asset IDs); the confound is the unexamined residue of that discovery.

### 7. The 0.005 practical-significance threshold will mean what it is advertised to mean

- **Category**: Methodological
- **Confidence**: Medium
- **Currently assumed because**: 0.005 was set as "~20–25% relative on a 0.02-corr task." The baseline pass band, however, is 0.005–0.05 — a baseline landing at 0.006 (in-band, gate passes) makes 0.005 an ~80% relative effect, and the equivalence-style "doesn't help" verdict becomes near-unfalsifiable while "helps" becomes near-unreachable.
- **What changes if wrong**: Verdict categories silently change stringency depending on where the baseline lands; the pre-registered threshold no longer encodes the intended judgment.
- **How to test**: Free. Pre-register the threshold as the max(0.005 absolute, ~25% of realized baseline corr) or commit to re-deriving it from the #8 result *before* unblinding the comparison, and record that rule now.
- **Relevant evidence**: Component #8 evidence text itself says the subsample and small tuning budget "may land low."

### 8. 2021-era practitioner sanity bands calibrate a 2026 v5-dataset subsampled pipeline

- **Category**: Data/Resource / Baseline
- **Confidence**: Medium
- **Currently assumed because**: The corr band (0.01–0.04), the leakage gate (>0.06), and corr-Sharpe ranges all come from 2021 forum posts about earlier dataset versions at full scale.
- **What changes if wrong**: The leakage gate is the study's single most important sanity check ("nothing downstream is interpretable until fixed"). If v5 targets/era structure or the chosen subsample shift the honest-performance band, the gate either misses leakage or falsely triggers, and the pipeline is tuned to a stale reference.
- **How to test**: One free check during #9: compute the sanity band from current v5 metadata/community-reported example-model scores (Numerai publishes example-model validation stats with the dataset), rather than the 2021 posts. Adjust the 0.06 gate accordingly.
- **Relevant evidence**: `numerai_forum2021_eras` / `numerai_forum2021_fnc` are the only protocol sources and are dated; the synthesis itself notes practitioner knowledge was "only partially surveyed."

### 9. A paired mean per-era difference is the right functional for the verdict

- **Category**: Methodological
- **Confidence**: Medium
- **Currently assumed because**: The verdict machinery is built on the mean paired per-era corr difference with a CI.
- **What changes if wrong**: Both the Feldman counter-hypothesis and assumption #1 predict *era-heterogeneous* effects (helps in calm regimes, hurts in tail regimes, or vice versa). A true heterogeneous effect can produce a mean ≈ 0 with a tight CI — the design would then confidently report "doesn't help" when the finding is "helps and hurts, conditionally," which is the more publishable result. The tail-era breakdown is listed only at the TMLR tier, so the minimum-viable path can emit the misleading verdict.
- **How to test**: Free: promote the tail-era/quantile breakdown (it re-slices existing predictions) into the minimum-viable deliverable, and pre-register that a null mean with significant era-quantile interaction reads "heterogeneous," not "no effect."
- **Relevant evidence**: `feldman2019longtailmemorization`; success-criteria already flags tail eras but only as a stretch tier.

## Background Assumptions (High confidence, Low impact)

### 10. Numerai data remains freely downloadable with usable era metadata

- **Category**: Data/Resource. Covered by component #9 at P=0.85; massively community-replicated. Acceptable risk.

### 11. TMLR/ICAIF will accept a negative-result transfer verdict on the author's own unpublished optimizer

- **Category**: Scope. The paper must introduce *and* evaluate the optimizer in one shot; TMLR's bar ("would some individuals be interested") is probably still met, and the workshop fallback exists. Low impact on the research itself; affects only venue choice.

### 12. The component probabilities in the lambda table are independent

- **Category**: Independence. Explicitly acknowledged in the decomposition (correlations noted, direction of bias discussed). Already surfaced; no action needed.

## Assumption Dependency Map

- **#1 is the root**: if "coherent = signal" fails in this domain, then #3 (engagement calibration) fails with it, the interpretive claim behind every verdict weakens, and #9's heterogeneity concern becomes the *expected* outcome rather than a corner case. #5 (batch-size conditionality) also sharpens, since era-factor coherence is strongly B-dependent.
- **#2 is independent of #1**: the shrinkage confound exists whether or not the coherence premise holds. Both can be true simultaneously (filter keeps factor directions AND shrinks updates), in which case a naive "helps" has two non-spectral explanations available.
- **#7 and #8 both hang on where the #8 baseline lands**: a low or stale-calibrated baseline breaks the practical-significance threshold and the leakage gate together.
- **#4 is a switch, not a chain**: if the variant swap fires, the novelty framing (Step 3) and diagnostics definitions (#1, #3) must all be revisited — it re-parents the whole verdict.

## Recommendations

**Test before proceeding (cheap, mostly folds into the existing gate):**
1. Add the spiked-covariance unit test (assumption #3) to the #6 local smoke — including the correlated-confound case (c). ~30 min CPU; it is currently the only thing standing between "diagnostics emitted" and "diagnostics correct."
2. Make batch composition (within-era vs. mixed-era) an explicit, logged design decision, and run the #1 diagnostics under both (assumption #1). Same debug job, two configs.
3. Recompute the sanity/leakage bands from current v5 example-model stats during #9 (assumption #8). Free.
4. Pre-register the practical-significance rule as baseline-relative before unblinding (assumption #7). Free.

**Plan revisions suggested:**
5. Add the random-subspace / norm-matched control (assumption #2), and rank it *above* the GAF ablation in the cut order — it defends the central claim; GAF defends only the positioning.
6. Promote the tail-era breakdown into the minimum-viable deliverable (assumption #9). It costs re-slicing, not GPU time.
7. Commit in writing to which code variant defines "the Spectral Optimizer"; treat a #7-forced swap to the streaming variant as a scope change requiring a novelty re-check, not a silent fallback (assumption #4).
8. Either run the sequence arm on the same dataset as the MLP, or add an MLP arm on the sequence dataset; otherwise claim "second setting," not "architecture consistency" (assumption #6).

**Acceptable risks:**
- #5 (B-conditionality) — accept, but state the claim as conditional on B and subsample scale.
- #10–#12 — accept as-is.
