# Assumption Analysis

**Run**: 2026-08-03-followup-follow-up-spectral-optimizer-redo-pxp-filter-walk-
**Scope note**: This is a follow-up run. The parent's challenge pass (prior/challenge/) already
covered the unchanged parts — era-autocorrelation inference, batch-composition ambiguity,
Feldman long-tail counter-hypothesis, MLP-vs-GBT gap, cut-cascade dynamics. This analysis
targets what is NEW in the corrected design: the p×p SpectralGradientFilter at Numerai scale,
the expanding-window walk-forward with refit, the 0.60× baseline gate, the parameter-space
random-subspace port, the matched 12-trial rank sweep, and the kill criterion that now sits
downstream of all of them. Overlap with the decomposition's priced risks (#1–#8) is avoided;
where a priced risk hides an unpriced framing assumption, only the unpriced part is surfaced.

## Summary

Thirteen load-bearing assumptions found: 5 critical, 5 moderate, 3 background. The most
consequential cluster sits around the run's two new adjudication devices — the mechanism
control (arm C) and the kill criterion — both of which are assumed to mean what they meant in
the parent design after ports that change their semantics: arm C's random subspace moves from
sample space (k/B ≈ 3–50% of gradient energy) to parameter space (k/p ≈ 0.3%), and the kill
criterion's "genuine sweep" condition is assumed satisfied by 12 trials over a search space
several times larger than the parent's. A third independent critical item: the run carries the
parent's NOVEL verdict across an object swap that the parent's own challenge pass (A4/F11)
explicitly classified as a scope change requiring a novelty re-check.

## Critical Assumptions (Low confidence, High impact)

### 1. The norm/k-matched random-subspace control remains a meaningful mechanism control after the port from sample space to parameter space

- **Category**: Methodological / Independence
- **Confidence**: Low
- **Currently assumed because**: Arm C is "the best artifact the parent run produced" and the
  brief says "port it, do not drop it." The matching *design* (same k(t), same update-norm-ratio
  trajectory) is proven twice — but in B×B sample space, where a random k-dim subspace of
  R^B (B≈1024) captures k/B of the gradient energy, i.e. a few percent to half. The port is
  treated as mechanical (#4, P=0.75 prices implementation, not semantics).
- **What changes if wrong**: In parameter space with p ≈ 600k and k ≤ 2048, a random
  orthonormal subspace captures E[‖Pg‖²/‖g‖²] ≈ k/p ≈ 0.3% of gradient energy, and the
  projected direction has cosine ≈ √(k/p) ≈ 0.06 to the true gradient. Matching arm B's
  update-norm-ratio trajectory (kept fraction plausibly 10–90%) then requires scaling the
  projected update by 30–250× in norm — arm C becomes norm-matched near-noise injection, not
  "generic low-rank projection." Two failure directions: (a) if the basis is **fixed**, arm C
  is intrinsic-dimension training (Li et al. 2018 style) with externally forced step norms —
  a regime with its own literature and behavior, unrelated to what arm B does; (b) if the
  basis is **resampled per step**, arm C is approximately scaled-gradient-plus-noise. Neither
  is stated; the fixed/refreshed choice appears only as an aside in #4 ("one-off or
  per-refresh"). If arm C simply trains to garbage, B > C trivially, the mechanism-honesty
  clause ("B indistinguishable from C ⇒ low-rank projection, not spectral selection") can
  never bind, and the control is structurally biased toward affirming the spectral mechanism —
  the exact opposite of its purpose. A WINS-with-mechanism claim would then rest on a vacuous
  control.
- **How to test**: Free, before Step 7: (i) decide fixed vs. refreshed basis in writing;
  (ii) 20-minute local CPU sim — measure arm C's captured-energy fraction and its realized
  norm-amplification factor at k ∈ {8, 512, 2048} for p = 600k, and check whether a
  norm-matched arm C can descend at all on the #3 synthetic task. If captured energy is
  ~k/p as predicted, redesign the control before any cluster spend — e.g. a random *rotation
  of the learned spectral basis* (random k-dim subspace drawn inside the span of recent
  gradients, norm-matched), which preserves "same energy budget, same subspace dimension,
  non-spectral selection" and is the semantics the sample-space original actually had.
- **Relevant evidence**: Parent exp-004/exp-005 nulls demonstrate the matching logic worked
  in sample space; nothing at any scale demonstrates it in parameter space. The decomposition
  itself flags basis memory (5 GB at k=2048) but not energy fraction — the cost was priced,
  the meaning was not.

### 2. Twelve trials per arm constitutes the "genuine rank sweep" that licenses the kill criterion

- **Category**: Methodological / Baseline
- **Confidence**: Low
- **Currently assumed because**: 12 is "the parent's precedent" and matched budgets are the
  DeepOBS standard; the brief says matched-budget tuning is "affordable for the first time"
  so there is "no excuse for an asymmetric budget." Matched, yes — adequate is a separate,
  unexamined question.
- **What changes if wrong**: Arm B's declared space is LR × {5 fixed ranks, effrank, gap} ×
  energy_threshold {0.90, 0.99} × alpha {0.5, 1, 2} (plus, per the criteria, re-tuned step
  count somewhere) — dozens of cells; 12 trials gives ~1–2 LR values per filter config at
  best. The parent pre-mortem's Scenario 4 (equal trial counts ≠ equal distance from optimum
  when spaces differ in dimension) predicted exactly this shape, and the parent's binding
  amendment F12 — downgrade a HURTS to "no evidence under the affordable budget" when
  best-config-on-grid-boundary or non-monotone-sweep signatures appear — was **not carried
  into this run's regenerated criteria**. The consequence is now larger than in the parent:
  the kill criterion is terminal ("line dead for financial tabular regression, STOP"), and
  it is conditioned on "the rank sweep genuinely ran." An under-resolved sweep that returns
  NULL/HURTS fires a permanent kill on an artifact, with no downgrade rule in force.
- **How to test**: Free, pre-registrable now: (i) restore F12's signature checks as binding —
  log per-fold best-config position relative to the grid boundary and sweep monotonicity;
  (ii) pre-specify the 12-trial allocation (e.g. a small number of filter configs × a short
  LR ladder, centered on prior H3/H5/H7 knowledge) rather than leaving the sampling scheme
  implicit; (iii) state in the criteria how a NULL/HURTS with under-exploration signatures is
  reported — kill with a stated caveat, or downgraded verdict. Deciding this after seeing
  results is the thing pre-registration exists to prevent.
- **Relevant evidence**: prior/challenge/pre-mortem.md Scenario 4; limitation-triage F12
  (binding in the parent, absent here); Gorishniy et al. on tuning-budget sensitivity.

### 3. The example-model yardstick is solidly positive and stable on every ~110-era TEST block

- **Category**: Baseline
- **Confidence**: Low–Medium
- **Currently assumed because**: The parent measured +0.0235 aggregate over 255 eras, and the
  gate is defined as a ratio (≥ 0.60×) against the per-fold restriction of that number. The
  decomposition prices whether arm A can *reach* 0.60× (#5, P=0.55) but never asks whether
  the denominator is well-behaved per block.
- **What changes if wrong**: Per-era corr for the example model has sd ~0.02–0.03; over a
  ~110-era autocorrelated block, block means can plausibly range ~0.015–0.03, and a
  bad-regime block could land near zero. The gate then degenerates in both directions: a
  near-zero denominator makes 0.60× trivially passable (gate passes, "baseline demonstrably
  works" is false), and a lucky-block denominator can fail a genuinely healthy baseline,
  burning the EU-5 reserve on a fold where nothing is broken. Either way the run's central
  interpretability device — the thing that distinguishes this run from the parent — silently
  stops measuring what it claims.
- **How to test**: Free and immediate, before protocol freeze: compute the example model's
  mean per-era corr on each *prospective* TEST block from
  `data/v5.0_validation_example_preds.parquet` the moment #8 emits candidate boundaries (this
  touches example preds, not this run's models — no TEST contamination of the comparison).
  If any block's yardstick is below a floor (e.g. ~0.010), pre-register a fallback gate for
  that fold (absolute floor, or ratio against the example model's all-validation mean) —
  chosen now, not after seeing arm A's numbers.
- **Relevant evidence**: Parent exp-001 per-era vectors exist on disk
  (`out/example_per_era_corr.csv`), so this is a 10-minute pandas check; parent noted
  regime drift across recent eras — the same drift that motivated the redo makes the
  per-block yardstick variance non-hypothetical.

### 4. Baseline-gate fixes are independent of the treatment arm's measured feasibility envelope

- **Category**: Independence / Scaling
- **Confidence**: Low–Medium
- **Currently assumed because**: The plan structure is linear: EU-1 measures throughput/VRAM
  and fixes the realized rank grid at one architecture; EU-2 runs the gate; EU-5 fixes the
  baseline if the gate fails, with "larger network" second on the fix list. Nothing connects
  the two.
- **What changes if wrong**: Every EU-1 measurement is conditional on p ≈ 600k. If EU-5
  fixes the gate by widening the MLP (say 2.3k→1024→512, p ≈ 2.9M), then: the p×k basis at
  k=2048 goes from ~5 GB to ~24 GB (plus arm C's basis), s/step at every rank point is
  invalid, the realized rank grid is invalid, and the packing arithmetic (#1) is invalid —
  with no experiment unit left to re-run the pre-flight. The run would face a choice between
  comparing arms at an architecture whose feasibility was never measured, or comparing at
  the small architecture that fails the gate. The gate-fix levers and the treatment envelope
  are coupled through p, and the plan assumes they are not.
- **How to test**: Cheap, inside EU-1: measure arm B s/step and VRAM at *two* architecture
  sizes (the planned one and the largest plausible gate-fix size) — a few extra 100-step
  timings in the same debug job. Alternatively pre-register the constraint the other way:
  gate fixes are restricted to levers that do not change p (steps, LR schedule, target
  transform, regularization, feature handling), with "larger network" explicitly triggering
  a documented re-preflight in EU-5's budget.
- **Relevant evidence**: Decomposition #6's own arithmetic (5 GB basis at 600k params) makes
  the 5× scaling consequence a one-line computation; parent F12 signatures suggested the
  parent baseline wanted *more* capacity/regularization room, so the "larger network" lever
  is likely to be reached for, not hypothetical.

### 5. The parent's NOVEL verdict covers the p×p streaming object

- **Category**: Scope
- **Confidence**: Low
- **Currently assumed because**: Step 1 carried novelty over on the reasoning that "the
  contribution remains the transfer verdict... nothing in the feedback changes the novelty
  position." But the parent's novelty assessment differentiated against GAF specifically via
  **full similarity-matrix eigendecomposition with an MP threshold** — properties of the B×B
  object. The parent's own assumption analysis (A4) and binding amendment F11 stated in
  writing that a swap to the streaming rank-k covariance variant is "a SCOPE CHANGE requiring
  a novelty re-check," naming K-FAC lineage and DP-PMLF as nearer neighbors to the streaming
  variant. This run performs exactly that swap — correctly, per the user's feedback — and
  carries the novelty file unchanged.
- **What changes if wrong**: The nearest-neighbor set for a streaming top-k
  gradient-subspace projector is different and denser: GaLore-style low-rank gradient
  projection (SVD of recent gradients, project, update in subspace), momentum-subspace and
  low-pass-filter methods, K-FAC-lineage covariance preconditioners. None of these was
  searched against, because the parent's search was framed around per-sample MP-spectral
  consensus. The transfer-verdict framing ("the contribution is the verdict, not the
  optimizer") survives, but the differentiation section of the write-up and the "unoccupied
  niche" language do not automatically. Risk is to the paper's positioning, not the
  experiment's validity — but it is the exact risk the parent flagged and this run waved
  through.
- **How to test**: Zero compute: a targeted mini-search (GaLore and successors,
  gradient-subspace/momentum-subspace optimizers 2024–2026, applied to regression/finance)
  before write-up, and one paragraph repositioning the mechanism relative to
  memory-motivated low-rank projection (same operator, different purpose and different
  claim). If a close neighbor applies low-rank gradient projection to noisy regression, the
  novelty positioning narrows to the protocol + verdict, which the criteria already treat
  as the primary contribution.
- **Relevant evidence**: prior/challenge/assumption-analysis.md A4;
  prior/challenge/limitation-triage.md F11; novelty-assessment.md differentiators are
  MP-spectral-specific.

## Moderate Assumptions (Medium confidence or Medium impact)

### 6. "All usable history" monotonically helps the gate

- **Category**: Methodological / Scope
- **Confidence**: Medium
- **Currently assumed because**: The parent's failure was too little data (150/574 eras), so
  the brief mandates the opposite pole: ALL usable eras, no era subsampling for headline
  runs. The corrective framing assumes more history is at worst neutral.
- **What changes if wrong**: Numerai v5 train reaches back ~two decades; under regime
  non-stationarity, uniformly weighted old data can *depress* recent-era performance —
  practitioners commonly downweight or truncate history. If so, the mandated protocol is
  itself a gate risk, and the listed fix levers (features, steps, network, LR schedule,
  target transform, regularization) exclude the natural remedy. Note also the expanding
  window makes fold 1 the *smallest*-data fold: the gate-first-on-fold-1 strategy is
  conservative only if data monotonically helps — the same assumption.
- **How to test**: Free clarification now: state in the protocol that era-recency *weighting*
  (sample weights, not era removal) is an authorized gate-fix lever and does not violate the
  no-subsampling mandate. Optionally, the fold-1 VALID sweep can include one recency-weighted
  trial as a probe at zero extra budget.

### 7. The VALID-selected configuration transfers across the refit boundary

- **Category**: Methodological
- **Confidence**: Medium
- **Currently assumed because**: The refit-from-scratch on TRAIN+embargo+VALID at
  VALID-selected hyperparameters is the protocol's centerpiece fix, imported from standard
  walk-forward practice.
- **What changes if wrong**: Two unstated sub-assumptions: (a) hyperparameters (especially
  step count / stopping point, which the criteria say must be re-tuned and therefore must
  live inside the 12-trial space) selected at training-set size N remain right at N + ~104
  eras; (b) the refit has **no held-out data** before TEST — early stopping is impossible
  without leakage, so the stopping rule must be fully determined on VALID (fixed steps or
  scaled steps). If the transfer is poor, *both* arms degrade at refit, which mostly hurts
  the gate (arm A) rather than biasing B−A — but a systematic filter-specific interaction
  (e.g. the filter's warmup/EMA horizon vs. a rescaled step budget) would bias the
  comparison. The step-count scaling rule is currently unspecified anywhere.
- **How to test**: Free: pre-register the refit stopping rule (e.g. steps scaled by
  rows_refit/rows_train at fixed batch size) in protocol.json before TEST is touched; log
  refit loss curves so under/overtraining at refit is visible post hoc.

### 8. The alpha=0 identity passing means the integration is verified

- **Category**: Methodological
- **Confidence**: Medium
- **Currently assumed because**: The identity is "free" and by-construction, and the brief
  elevates it to the trust precondition for all comparisons.
- **What changes if wrong**: The identity exercises only the no-op path. Hard top-k
  projection, adaptive effrank/gap, energy-threshold, and soft-alpha reweighting code paths
  are all untouched by it; a bug there passes the identity and silently corrupts arm B in
  every trial. The parent's C1 amendment (planted-spike synthetic test: filter keeps a
  planted low-rank signal, rejects i.i.d. noise) verified the mechanism computes what it
  claims; no analog exists in this run's #3 quick test ("loss falls" is much weaker).
  Relatedly, the parent's C3 cosine diagnostic (filtered vs. unfiltered update) was dropped:
  without it, a NULL verdict cannot distinguish "the filter changed the trajectory and it
  didn't matter" from "VALID selected a near-identity config (high rank, soft alpha,
  soft_residual=True) and B ≈ A by construction." Both are legitimate NULLs but they mean
  different things, and the kill-criterion write-up should say which occurred.
- **How to test**: Cheap, inside the existing #3 local smoke: add a planted-subspace
  synthetic check (filter at hard top-k recovers a planted dominant direction) and log the
  per-step cosine between filtered and unfiltered updates plus kept-norm fraction in all
  arm-B runs; report the selected config's distance-from-identity per fold.

### 9. Three walk-forward folds are quasi-independent confirmations under the 2-of-3 rule

- **Category**: Independence
- **Confidence**: Medium
- **Currently assumed because**: The decision rule ("≥ 2 of 3 folds, CI excluding zero, same
  sign") is framed as replication across folds; the decomposition notes gate correlations
  but not verdict correlations.
- **What changes if wrong**: Expanding windows are nested (fold 3's training data contains
  fold 1's entirely), seeds are shared, and the three TEST blocks are contiguous slices of
  one recent regime. The (B−A) estimates across folds are positively correlated, so 2-of-3
  agreement carries less evidential weight than three independent replications — a
  regime-specific artifact can clear the rule. This inflates confidence in WINS/HURTS
  verdicts (and in the kill criterion's finality) rather than biasing their direction.
- **How to test**: Free at write-up: report the cross-fold correlation of per-era (B−A)
  where blocks are adjacent, and state the verdict as "consistent across 3 nested folds
  covering [date range]" rather than implying independent replication. No design change
  warranted within budget — this is a claim-calibration item.

### 10. The per-sample-consensus interpretive frame transfers to a temporal covariance object

- **Category**: Theoretical
- **Confidence**: Medium
- **Currently assumed because**: The literature synthesis (carried from the parent) frames
  the mechanism via Coherent Gradients / per-sample agreement / GAF, and the Feldman
  long-tail counter-hypothesis is about suppressing rare *samples'* gradients.
- **What changes if wrong**: `SpectralGradientFilter`'s covariance is an EMA (decay 0.99,
  ~100-step horizon) of **batch-mean gradient** outer products across *steps* — it measures
  temporal persistence of mean-gradient directions, not cross-sample agreement within a
  batch. "Directions many samples agree on" becomes "directions the optimizer has recently
  and persistently moved in" — closer to a momentum subspace. The Feldman story (rare-sample
  signal suppressed) and the parent's era-factor concern both need re-derivation in
  step-space before being used to explain whatever verdict emerges; a mechanism narrative
  imported from the sample-space frame could caption the result wrongly (the parent
  pre-mortem's Scenario 2 shape, one level up). This does not touch the verdict machinery —
  only its explanation.
- **How to test**: Free at write-up time: describe the mechanism as temporal
  mean-gradient-subspace filtering; check which interpretive claims from the synthesis
  survive that description. The kept-subspace diagnostics from #8 above (cosine, kept-norm
  fraction, selected-rank trajectory) are the evidence base for any mechanistic sentence.

## Background Assumptions (High confidence, Low impact)

### 11. Era capacity and example-preds coverage suffice for 3 folds

- **Category**: Data/Resource. 3×(110+96+8) plus fold structure fits inside the parent's
  measured 643+ usable validation eras, and all VALID/TEST blocks land in the
  example-preds-covered region. Verified arithmetically by #8's quick test. Acceptable;
  just include an explicit "all TEST eras present in example preds" assert alongside THE
  assertion.

### 12. Paired seeding is implementable across arms

- **Category**: Methodological. "Same seed, same data order" requires that arm B/C's extra
  RNG consumption (basis init, subspace draws) comes from a **separate generator** from the
  data-order stream, else pairing silently breaks. One line of seed discipline; worth a
  comment in the harness, not a risk.

### 13. Cluster torch 2.5.1 runs the filter code unmodified

- **Category**: Data/Resource. Covered by the EU-1 on-cluster identity re-assert already in
  the plan. Acceptable.

## Assumption Dependency Map

- **#1 → the mechanism claim**: if the arm C port is semantically broken, (B−C) and (C−A)
  are uninformative and the honesty clause is vacuous; #8's dropped diagnostics then remove
  the *only other* window into mechanism, so #1 and #8 failing together leave any verdict
  mechanistically uncaptioned — at which point #10's frame-transfer risk becomes the
  write-up's default failure mode.
- **#2 → the kill criterion**: the kill criterion inherits whatever the sweep-adequacy
  assumption is worth; #7 feeds it too (step-count selection consumes trials from the same
  budget of 12).
- **#3 and #4 are the gate's two hidden flanks**: #3 corrupts the gate's meaning (denominator),
  #4 fires after a gate failure (fix levers). #6 can *cause* the gate failure that triggers #4.
  All three sit underneath the decomposition's priced P=0.55 without being priced in it.
- **#5 is a switch**: independent of all experimental outcomes; flips only the positioning of
  the write-up.

## Recommendations

**Test before proceeding (all free or folded into existing local work):**
1. Compute the example model's per-block mean corr on the prospective TEST blocks as soon as
   #8 emits boundaries; pre-register the degenerate-yardstick fallback now (#3).
2. Decide arm C's basis semantics (fixed vs. refreshed vs. random-rotation-within-gradient-span)
   in writing, and run the 20-minute captured-energy/norm-amplification sim before porting
   code (#1). If k/p energy is confirmed, change the control design, not just its
   implementation.
3. Restore parent F12 as binding: log grid-boundary and monotonicity signatures per sweep;
   pre-specify the 12-trial allocation and what an under-exploration NULL/HURTS is reported
   as (#2).
4. Pre-register the refit stopping rule in protocol.json (#7), and add the planted-subspace
   correctness check plus filtered-vs-unfiltered cosine logging to the #3 smoke (#8).
5. In EU-1, time arm B at the largest plausible gate-fix architecture too, or restrict gate
   fixes to p-preserving levers in writing (#4).
6. State that era-recency weighting is an authorized gate-fix lever compatible with the
   no-subsampling mandate (#6).

**Plan revisions suggested:**
- #1 is the only item that plausibly changes an arm's *design* rather than its logging or
  pre-registration; it should be resolved before Step 7 finalizes the arm C spec.
- #5 costs one targeted search plus a paragraph; do it before write-up, not after review.

**Acceptable risks:**
- #9 (fold correlation) — accept; calibrate the claim language.
- #10 (interpretive frame) — accept for execution; gate the write-up's mechanism sentences on
  the #8 diagnostics.
- #11–#13 — accept with the one-line asserts noted.
