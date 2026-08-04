# Success Criteria

**Run**: 2026-08-03-followup-follow-up-spectral-optimizer-redo-pxp-filter-walk-
**Status**: ROUND 2 — regenerated after the Step 6 challenge loop-back (verdict
MAJOR_REVISIONS). This file REPLACES round 1 and is the **frozen audit anchor
for Step 10**: the audit judges the run against what is written here, not
against the parent's criteria, not against round 1, and not against anything
decided after TEST data is touched. Every binding element of round 1 is
preserved in force; the challenge's A-series amendments (A1–A8, step 6
decision in `state.md`) are integrated as binding criteria throughout —
principally in the gate semantics, the sweep specification, the inference
spec, the arm C design, the verification requirements, and the kill
criterion, which is now power- and scope-qualified. The parent run's success
criteria were anchored to the wrong optimizer (the per-sample B×B
`SpectralConsensusFilter`) and a broken temporal split (~5.4-year train→test
gap, no refit, baseline below its own pre-registered sanity band); nothing in
them binds this run.

## The Motivating Question (directional)

Does the p×p **SpectralGradientFilter** (repo root of `~/pyg/optimizers`,
`spectral_filter.py` — verified present 2026-08-03, class at line 52,
`filter_grad()` at line 303) improve out-of-sample predictive performance
versus tuned AdamW on Numerai v5 data?

**The deliverable is a WINS / HURTS / NULL verdict** on this question,
delivered under the pre-registered protocol below. Positive, negative, and
null verdicts are equally valuable outcomes; only an *uninterpretable* run
(gate failed unfixably, sweep not genuinely run, protocol assertion never
fired) is a failure. The verdict taxonomy — including the qualified forms a
NULL or HURTS can take when power, rank-grid scope, or tuning-budget
resolution limit what the evidence can support — is defined in full under
"Decision rule" and "Kill criterion" below.

The parent run's audited negative (−0.00527 mean per-era corr, CI
[−0.00886, −0.00181]) is a fact about the B×B variant on a broken split. Its
target-independence theorem depends on row normalization and does **not**
bind `SpectralGradientFilter`; it may be cited only as a scoped side-note
about the per-sample variant. This run's outcome is genuinely uncertain in
both directions.

## State of the Art

### Current Best Methods / Reference Points

| Method | Source | Key Metric | Value | Year |
|--------|--------|------------|-------|------|
| Numerai example model (official reference predictor) | `data/v5.0_validation_example_preds.parquet` | mean per-era numerai_corr (parent's test eras 0971–1225) | +0.0235 | 2025 |
| Tuned AdamW MLP (parent run, broken protocol — 150/574 eras, no refit) | parent exp-003/exp-004 | mean per-era numerai_corr | +0.0064 (below parent's own sanity band [0.0087, 0.0522]) | 2026 |
| B×B SpectralConsensusFilter (parent run — wrong object, broken split) | parent exp-004 + audit | Δ(filter − AdamW) per-era corr | −0.00527, CI [−0.00886, −0.00181] | 2026 |
| Muon (best optimizer on tabular MLPs) | `gorishniy2026benchmarkingoptimizers` | consistently beats AdamW across 17 tabular datasets | qualitative ranking | 2026 |
| Gradient Agreement Filtering | `chaubard2024gradientagreementfiltering` | reduced memorization in distributed image classification | domain-specific; no financial results | 2024 |
| RMT/Marchenko-Pastur covariance cleaning | `bouchaud2009rmtfinancereview` lineage | improved portfolio risk estimates (returns covariance, not gradients) | qualitative | 2009– |

### SOTA Summary

There is no academic SOTA for "optimizer-level gradient-covariance spectral
filtering on financial regression" — the niche is unoccupied across the
28-reference literature base (novelty verdict: NOVEL, carried from parent,
with a caveat: the B×B→p×p object swap requires a targeted novelty
mini-search against GaLore-style low-rank gradient projection,
momentum-subspace methods, and the K-FAC lineage before write-up; risk is to
positioning, not experimental validity). The relevant reference points are
therefore: (a) the **Numerai example model**, the domain's
practitioner-standard yardstick, which scores +0.0235 mean per-era corr on
the parent's test eras and whose per-fold score on this run's TEST eras
defines the baseline gate; (b) **tuned AdamW**, the credible optimizer
baseline per the matched-tuning-budget standard (DeepOBS, Gorishniy et al.);
and (c) the **parent run's audited negative**, which is scoped to the B×B
variant and does not pre-answer this run's question.

Two literature findings calibrate expectations in both directions. Against
the filter: Adam-family optimizers already carry noise robustness in
heavy-tailed gradient regimes, and Feldman's long-tail memorization result
predicts that consensus filtering could suppress rare-but-genuine financial
signal. For the filter: Coherent Gradients, GAF, and prior findings H3/H5
(adaptive rank groks sparse parity; rank ≤ 4 destabilizes, ~10 fastest)
show the mechanism has real traction when the solution is genuinely
low-dimensional. The question is a transfer verdict, and it is open.

The contribution, per the novelty assessment, is **the transfer verdict,
not the optimizer**: a rigorous answer to whether gradient-covariance
spectral filtering helps low-SNR financial tabular regression — including,
explicitly, a well-evidenced negative.

## Benchmarks

### Standard Evaluation

| Benchmark/Dataset | Metrics | Typical Range | Notes |
|-------------------|---------|---------------|-------|
| Numerai v5 (train + validation parquets, ~6.3 GB, already on local disk in parent `data/` — copy/symlink, do not re-download) | mean per-era numerai_corr (primary); mean per-era Spearman, corr-Sharpe (secondary) | example model +0.0235 on parent's test eras; competitive edges are small (~0.01–0.05 per-era corr) | Era structure (5-day spacing, 20-day target horizon) mandates purging/embargo; no academic literature — protocol comes from practitioner sources |
| 3-fold expanding-window walk-forward (this run's protocol) | per-fold paired differences (B−A), (B−C), (C−A) with 95% moving-block bootstrap CIs and per-fold MDE reported alongside | — | TRAIN / embargo E=4 / VALID (hp selection only) / embargo E=4 / TEST; refit from scratch on train+embargo+valid at VALID-selected hps; TEST touched exactly once per fold for the comparison (gate retries are the one logged exception — see gate semantics) |

Known issues this protocol addresses: naive temporal splits leak via the
20-day overlapping target (hence embargo E = ceil(20/5) = 4); large
train→test gaps without refit produce broken baselines (the parent's
failure); per-era corr is autocorrelated across eras, so plain bootstrap
CIs are invalid (hence moving-block bootstrap with per-fold block length);
the three folds are nested, not independent — claim language is calibrated
accordingly (see inference spec).

### Required Baselines

For this work to deliver an interpretable verdict, it must compare against:

1. **Arm A — tuned AdamW** (12 trials on VALID per fold) — the credible
   control; an optimizer claim on tabular data without a tuned-AdamW
   comparison at matched budget is not publishable (Gorishniy et al.,
   DeepOBS standard).
2. **Arm C — matched-invariant random-basis mechanism control** — the
   control that separates "spectral selection matters" from "any low-rank
   projection does this". **Re-derived for parameter space (amendment A7)**:
   the matching invariants are the kept-rank trajectory k(t), the
   update-norm-ratio trajectory, **and the basis-rotation rate**. The
   preferred design is a **random orthogonal rotation of arm B's own
   realized basis** — matching everything except the spectral identity of
   the retained directions. The parent's sample-space (B×B) matching design
   is the ancestor, not the spec: a fixed random parameter-space subspace at
   k/p ≈ 0.3% captured energy is NOT an acceptable implementation (it is
   norm-matched noise injection, and would make the honesty clause vacuous).
   A ~20-minute local CPU simulation — captured-energy fraction and realized
   norm-amplification factor at k ∈ {8, 512, 2048} with p ≈ 600k — is
   **required before porting any arm C code**, and its output determines the
   final control design within the invariants above. Arm C is reported **in
   absolute terms** (its own loss curves and TEST scores), not only as
   differences — a straw control must be visible in its own row of the table.
3. **Numerai example model** (per-fold TEST-era mean per-era corr from
   `data/v5.0_validation_example_preds.parquet`) — not an arm but the
   yardstick for the baseline gate, with the denominator semantics defined
   under the gate section (A5).
4. **Sanity assertions (arm D, not an arm)**: (i) `weighting="soft",
   alpha=0, soft_residual=True` must reproduce plain AdamW (bit-for-bit in
   fp64, or indistinguishable in fp32) before any comparison is trusted;
   (ii) a seeded zero-predictor control; (iii) **a planted-subspace
   correctness check (amendment A8)**: the filter at hard top-k must recover
   a planted dominant gradient direction on a synthetic task — this
   exercises the filtering code paths the alpha=0 identity leaves untouched.

Muon is a desirable secondary baseline per the literature but is **not
required** within this run's 5-experiment / <30-GPU-min-per-job budget; if
budget does not permit it, it is future work, not a criterion.

## Pre-Registered Protocol (frozen before TEST is touched)

The **protocol** is pre-registered and frozen before any TEST data is
evaluated: splits, embargo, metrics, bootstrap procedure (including seed
handling), decision rule, baseline gate (including denominator fallback,
near-miss bands, fix ladder, and retry semantics), arm-B trial allocation,
under-exploration signatures, kill-criterion qualifications, refit stopping
rule — everything in this section and the ones that follow, with realized
numeric boundaries and mappings recorded in `protocol.json`. A filter
**operating point is explicitly NOT pre-registered**: rank is a
hyperparameter tuned on VALID, never a frozen constant. (Freezing an
operating point chosen from a diagnostic proxy was the parent's fatal
process error; the pre-registration machinery was good — the freeze line
moves, the machinery stays.)

Amendments A1–A8 below are wording and specification changes made at zero
compute, before `protocol.json` freezes and before any TEST touch — the
moment at which such changes are legitimate. After freeze, they are as
binding as everything else here.

### Splits and the protocol validity assertions

- 3-fold expanding-window walk-forward. Per fold: TRAIN = all usable eras
  up to T; embargo E = 4; VALID = next ~96 eras (hyperparameter selection
  ONLY); embargo E = 4; TEST = next ~110 eras. The three TEST blocks are
  contiguous, non-overlapping, and cover the most recent portion of the
  data (coverage ≥ 0.95). Exact boundaries computed from the realized
  usable-era list and written to `protocol.json` — not hardcoded — together
  with the **raw-era ↔ usable-index mapping (A8)**, so the assertion below
  is checkable against era-index gaps.
- Before TEST evaluation, **refit from scratch** on TRAIN + embargo +
  VALID at the VALID-selected hyperparameters. TEST is evaluated exactly
  once per fold for the optimizer comparison (gate retries are separately
  logged — see gate semantics). The **refit stopping rule is pre-registered
  in `protocol.json` before TEST is touched (A4)**: the refit has no
  held-out data, so early stopping is impossible without leakage; the rule
  is deterministic given VALID-stage quantities (default: refit steps =
  selected-trial steps scaled by rows_refit/rows_train at fixed batch
  size). Refit loss curves are logged so under/over-training at refit is
  visible post hoc.
- **THE assertion of the run**, hard-asserted in code and printed in the
  log for every fold:

  ```
  assert min(test_eras) - max(refit_train_eras) == E + 1   # == 5 with E = 4
  ```

  The test period begins one embargo after the refit training data ends.
  A run that reports results without this assertion having fired for every
  fold is not answering the question.
- **Companion assertion (A8)**, alongside THE assertion for every fold:
  every TEST era of every fold is present in
  `data/v5.0_validation_example_preds.parquet` — an off-by-one in the fold
  boundaries would otherwise silently break the gate denominator.
- Headline runs use ALL usable eras up to each fold boundary (no era
  subsampling) and the full v5.0 feature set, unless VRAM genuinely forces
  a reduction — in which case the constraint is documented and reported as
  a limitation. Step count is re-tuned for the larger training sets
  (parent's 2000 steps @ B=1024 must not be inherited); how it is re-tuned
  is specified under "Matched tuning budget" below (A4).

### Baseline gate (hard gate, full semantics — A5)

- **Denominator, computed before protocol freeze**: for each *prospective*
  TEST block, the Numerai example model's mean per-era numerai_corr
  restricted to that block is computed **locally, before `protocol.json`
  freezes**, from the parent's `out/example_per_era_corr.csv` /
  `data/v5.0_validation_example_preds.parquet`. This touches example
  predictions only, not this run's models — no TEST contamination. The
  per-fold denominators are recorded in `protocol.json`.
- **Degenerate-yardstick fallback, pre-registered now**: the gate
  denominator for a fold is `max(example-model per-fold mean per-era corr,
  0.010)` — an absolute floor at 0.010. A fold whose raw example-model mean
  falls below 0.010 is flagged **low-signal yardstick** in all reporting,
  and its gate result carries that flag. (A near-zero raw denominator would
  otherwise make the gate trivially passable while "baseline demonstrably
  works" is false.)
- **GATE**: tuned AdamW (arm A) must reach **≥ 0.60× the (floored)
  denominator** on the fold's TEST eras. If the gate fails on a fold, **no
  optimizer comparison runs on that fold**. The gate outcome and realized
  ratio are reported explicitly for every fold, pass or fail.
- **Near-miss and structural-gap bands, pre-registered now**:
  - realized ratio ≥ 0.60 → gate passes.
  - realized ratio in **[0.50, 0.60)** → near-miss: **one targeted fix**
    (the highest-priority applicable rung of the ladder below), one
    re-check, then **accept the outcome** — pass or fail, no further repair
    spend on that fold.
  - realized ratio in **[0.45, 0.50)** → the pre-specified fix ladder runs
    within the reserve unit's budget, subject to its stopping rule.
  - realized ratio **< 0.45** → structural gap: **no repair**. The reserve
    unit buys **diagnosis plus a data-scaling learning curve** for the
    baseline (mean per-era corr vs training-era count) — converting the
    gate failure into quantitative evidence on assumption 3 ("the parent's
    shortfall was protocol-induced, not intrinsic"), which is the one
    question this failure mode can still answer. Twelve fix trials will not
    close a 15-point ratio gap.
- **Baseline-fix ladder, pre-specified with per-rung budgets and a stopping
  rule** (ordering per the parent's F12 signatures, which showed the
  baseline under-regularized at its grid boundary):
  1. Regularization (weight decay, dropout) — 4 trials.
  2. Network size — 4 trials, **available only if the pre-flight
     experiment unit also timed arm B at that larger architecture**;
     otherwise this rung is skipped (see p-preservation below).
  3. LR schedule and step count — 2 trials.
  4. Era-recency **weighting** (sample weights, not era removal) — 1
     trial. This is an **authorized gate-fix lever, explicitly compatible
     with the no-subsampling mandate** (weighting is not subsampling).
  5. Target transform / feature handling — 1 trial.
  - **Stopping rule**: stop at the first rung whose best trial reaches
    ratio ≥ 0.60; hard cap 12 fix trials per fold; futility stop if two
    consecutive rungs improve the realized ratio by < 0.02 combined.
- **p-preservation constraint**: gate fixes are restricted to levers that
  do not change the parameter count p (steps, LR schedule, target
  transform, regularization, feature handling, era-recency weighting)
  **unless** the pre-flight unit also timed arm B (throughput, VRAM,
  realized rank grid) at the larger architecture — because every
  feasibility measurement is conditional on p, and comparing arms at an
  architecture whose feasibility was never measured is not permitted.
- **Retry semantics (leakage control)**: a gate re-check after a baseline
  fix is a second TEST touch for arm A. Therefore: gate retries select on
  the **gate ratio only** — never on any comparison quantity; after any
  baseline fix, the comparison hyperparameters (all arms) are **re-selected
  on VALID**; and the **TEST-touch count is logged per fold and reported**
  in the write-up. This bounds and discloses the bias a partially
  TEST-selected baseline could introduce into (B−A).

### Matched tuning budget and rank sweep (A4, A3, A8)

- **12 trials per arm** on VALID, per fold. Arm B's search space **must
  include the learning rate** (filtering changes the effective step size).
  Rank is swept on VALID, never frozen. `normalize="none"` per prior
  finding H7. Measured seconds/step reported for both arms (the p×p filter
  is ~2× Adam, which is what makes the matched budget affordable this
  time).
- **Pre-registered arm-B trial allocation — staged design over a pruned
  space (A4)**. Twelve uniform trials cannot resolve the brief's full
  enumeration (~42 discrete configs × LR); an unresolvable sweep would
  fail the "genuine sweep" clause by construction. Therefore:
  - **Stage 1 (7–8 trials)**: one trial per rank/adaptive grid point at a
    transferred LR, centered on the parent's empirical priors (H3: rank ≤ 4
    destabilizes, ~10 fastest; H5: adaptive effective-rank was the only
    parity-grokking variant). Grid points: fixed ranks from the realized
    log grid (target {8, 32, 128, 512, 2048}, capped at what the pre-flight
    measures as feasible for p) plus `adaptive="effrank"`.
  - **Stage 2 (4–5 trials)**: refine LR and alpha around the stage-1
    winner.
  - **Pruning, documented under the brief's feasibility clause**:
    `adaptive="gap"` and one `energy_threshold` value (keep 0.90, drop
    0.99) are dropped from the swept space; alpha enters only at stage 2.
    A resolvable sweep over a smaller space beats an unresolvable sweep
    over the full enumeration; the pruning is stated in the write-up.
  - The realized allocation (exact grid, transferred LR, stage boundaries)
    is written into the experiment plan / `protocol.json` **before TEST is
    touched**.
- **Step count is NOT inside the 12 trials (A4)**: the converged training
  step count is fixed by the pre-flight experiment unit's **full-length
  arm-A convergence run** (which also collapses the ~500×-extrapolated
  cost anchor and provides an early proxy gate ratio). The refit stopping
  rule (above) is likewise outside the trial budget.
- **Realized rank grid and adaptive-rank logging (A2)**: the realized grid
  (after any feasibility capping) is recorded in `protocol.json`, and the
  **realized k(t) trajectory of every adaptive configuration is logged in
  every run**, so the condition "the feasibility cap binds the adaptive
  arms too" is checkable from artifacts, not memory. The kill criterion's
  scope is tied to the realized grid (see kill criterion).
- **Under-exploration signatures, pre-registered as binding (A3 —
  reinstating the parent's F12 guard)**. The following are computed and
  reported for every fold's arm-B sweep:
  1. **Grid-boundary signature**: the VALID-selected arm-B configuration
     sits at an edge of the realized grid (minimum or maximum swept rank,
     or an endpoint of the LR or alpha refinement range).
  2. **Non-monotone/high-variance signature**: the stage-1 VALID score
     across the rank grid is non-monotone with a range smaller than ~2× the
     across-seed standard deviation at fixed config (i.e., the sweep cannot
     rank its own grid points).
  3. **LR-shift signature**: arm B's selected LR differs from arm A's
     selected LR by more than one grid step (≥ 4× multiplicatively),
     indicating the transferred-LR stage-1 centering was off.
  If a HURTS verdict emerges **and any signature fires**, the verdict is
  **downgraded to "no evidence of benefit under the affordable tuning
  budget"** — reported as such, and the kill criterion does not fire (see
  kill criterion). The signatures and this downgrade rule are frozen now,
  before any TEST touch.
- **Declared limitation (A8)**: `decay` and `warmup` are fixed at repo
  defaults (0.99, 100) across ~10k-step schedules on regime-shifting data
  and are not in the swept space — the covariance-memory horizon is
  untested. This is declared a limitation NOW, not discovered at write-up.
  Step 5 may lift it by swapping one pruned grid dimension for
  `decay ∈ {0.99, 0.999}` under the staged design; if it does not, the
  limitation stands and is reported.

### Metrics and inference (A1, A6)

- **Primary**: mean per-era numerai_corr on TEST.
- **Secondary**: mean per-era Spearman; corr-Sharpe (mean/sd across eras).
- **Paired seeds**: **target 5, floor 3, per arm per fold** (A6). Seeds are
  raised to 5 if experiment-unit packing permits — this is the only
  authorized upward power lever; if packing does not permit, ≥ 3 remains
  the binding floor (round 1's requirement). Same seed and same data order
  across arms so differences are paired per era; arm B/C's extra RNG
  consumption (basis init, subspace draws) must come from a separate
  generator so pairing does not silently break.
- **95% moving-block bootstrap CIs** per fold, block length from the
  fold's own lag-1 autocorrelation of the per-era corr series (recomputed
  per fold, not inherited from the parent's measurements). Bootstrap RNG
  fixed and stated; CI stability checked across ≥ 2 RNG seeds before any
  stability claim (the parent's Sharpe CI was bootstrap-RNG fragile).
- **Seed-variance handling, stated explicitly (A6)**: the headline CI uses
  **hierarchical resampling** — moving-block resampling over era blocks of
  per-era paired differences, with seed-level resampling nested inside —
  so seed variance is folded into the CI rather than averaged away.
  Additionally, the ratio of between-seed variance of the fold-mean (B−A)
  to the era-blocked variance is computed and reported; if seed variance is
  demonstrably negligible (< ~10% of era variance), that check is the
  reported justification and the era-block CI may serve as headline, with
  the hierarchical CI reported alongside.
- **Minimum detectable effect (MDE), standing table (A1)**: for every fold
  and every reported CI, the realized per-fold MDE — operationally, the
  95% CI half-width of the seed-averaged fold-mean paired difference under
  the same hierarchical moving-block bootstrap — is reported **next to the
  CI as a standing table**. This makes the power-limited/adequately-powered
  distinction auditable rather than narrative, and it feeds the kill
  criterion's power qualification.
- Report the paired differences **(B−A), (B−C), (C−A)** per fold, each
  with 95% CI and MDE.
- **Fold non-independence, claim calibration (A6)**: the three folds are
  nested (expanding windows share training data, seeds, and one recent
  regime). The **cross-fold correlation of per-era (B−A)** is computed and
  reported, and all claim language reads "**consistent across 3 nested
  folds covering [date range]**" — never "independent replications".
- Numerical robustness: the per-step (k+1)×(k+1) eigh is wrapped in a
  CPU-fp64 fallback from the start (parent audit Finding 5: fp32 cuSOLVER
  eigh fails stochastically under rank collapse); every firing is logged
  and the count reported.

### Verification depth (arm D + A8)

Before any comparison is trusted:

- **Alpha=0 identity**: `weighting="soft", alpha=0, soft_residual=True`
  reproduces plain AdamW bit-for-bit in fp64 (or indistinguishably in
  fp32). This verifies the no-op path only, so it is supplemented by:
- **Planted-subspace correctness check**: the filter at hard top-k recovers
  a planted dominant gradient direction on a synthetic task — exercising
  the top-k / adaptive / energy-threshold code paths the identity leaves
  untouched.
- **Zero-predictor control**: seeded, as in round 1.

During all arm-B runs:

- **Per-step filtered-vs-unfiltered cosine and kept-norm-fraction logging**
  in every arm-B run, and the **selected configuration's
  distance-from-identity reported per fold**. Without these, a NULL cannot
  distinguish "the filter changed the trajectory and it didn't matter"
  from "VALID selected a near-identity configuration and B ≈ A by
  construction" — both are legitimate NULLs, but the write-up must say
  which occurred, and any mechanistic sentence in the write-up is gated on
  these diagnostics.

### Decision rule (pre-registered)

- **WINS** if (B−A) > 0 with 95% CI excluding zero on ≥ 2 of 3 folds,
  same sign.
- **HURTS** if (B−A) < 0 with 95% CI excluding zero on ≥ 2 of 3 folds,
  same sign — subject to the A3 under-exploration downgrade above.
- **NULL** otherwise — subject to the A1 power qualification below.
- Effect sizes with CIs and MDEs are reported regardless of category.
- **Mechanism honesty, symmetric (A7)**: if B beats A but B is
  statistically indistinguishable from C, the conclusion is "low-rank
  projection helps; spectral selection is not the active ingredient" —
  that is what gets claimed, not the spectral mechanism. Symmetrically, a
  "B ≫ C" mechanism claim requires evidence that C was a **fair control**
  (the rotation-rate/invariant matching of A7, evidenced by C's absolute
  loss curves and the basis-rotation diagnostics); absent that evidence,
  the claim downgrades to "adaptive low-rank projection helps".

### Kill criterion (power- and scope-qualified — A1, A2, A3)

The kill criterion exists so a clean negative ends the line without
epicycles — and its qualifications exist so it cannot fire on an artifact
of power, feasibility capping, or tuning-budget resolution. The parent's
central failure was a safeguard that verified execution while the
conclusion went unlicensed; these qualifications are that lesson applied
one level up.

**The line is declared dead for financial tabular regression — write the
clean negative and STOP (no fourth epicycle, no new dataset, no
full-scale-rerun rescue) — only if ALL of the following hold:**

1. **Execution validity** (round 1, unchanged): the baseline gate passed on
   the folds used; the rank sweep genuinely ran at matched budget per the
   pre-registered staged allocation; and the result is NULL or HURTS on
   ≥ 2 of 3 folds. A negative on a failed gate or a skipped/truncated sweep
   is uninterpretable, not a finding.
2. **Power qualification (A1)** — applies to NULL: a NULL is terminal only
   if the realized per-fold MDE ≤ **0.005** on ≥ 2 of the gated folds
   entering the decision rule. Otherwise the verdict is **"NULL
   (power-limited): |B−A| bounded within ±MDE; line not declared dead"**,
   and the write-up carries a resourced decisive-test ask in Future Work
   (what design, at what compute, would resolve ±0.005).
3. **Rank-grid scope (A2)**: the kill verdict is automatically scoped to
   the realized rank grid ("dead for k ≤ K_max on this problem"). If the
   realized **K_max < 512**, a NULL/HURTS outcome reports **"no evidence at
   feasible ranks"** plus a **costed future-work ask** for the untested
   rank regime, instead of killing the line. The logged adaptive-arm k(t)
   trajectories are the evidence for whether the cap bound the adaptive
   arms too.
4. **Under-exploration guard (A3)** — applies to HURTS: if any
   pre-registered signature fired, the HURTS is downgraded to "no evidence
   of benefit under the affordable tuning budget" and does not kill the
   line.

When all conditions hold, the clean negative IS a success outcome of this
run. When condition 1 holds but 2, 3, or 4 does not, the run still delivers
its verdict in the corresponding qualified form — informative, publishable,
and honest about what the design could see — with the follow-up path stated
as a costed ask rather than forbidden.

## Publishability Criteria

### Target Venues

- **Primary**: NeurIPS/ICLR optimization workshop (e.g., OPT / Optimization
  for Machine Learning) — the contribution is an optimizer transfer verdict
  under a rigorous matched-budget protocol; workshops in this family value
  well-characterised negatives and protocol contributions.
- **Secondary**: ACM ICAIF or a quantitative-finance ML venue — the
  era-purged walk-forward protocol documentation is itself a contribution
  given the absence of academic Numerai literature.
- **Workshop/fallback**: an ML-evaluation or reproducibility workshop
  (e.g., NeurIPS Datasets & Benchmarks track workshops) — the paired
  spectral-vs-random-basis control design and the protocol repair story
  (what the parent got wrong and how the redo fixed it) stand alone.

A main-conference ML paper (ICML/NeurIPS main track) would require breadth
this run's compute budget does not permit (multiple datasets, Muon and GBT
baselines, architecture variety); that is future work, not a criterion.

### Evidence Thresholds

**Minimum viable (workshop paper / informative internal verdict)**:
- Baseline gate passed (arm A ≥ 0.60× the floored denominator) on ≥ 2 of 3
  folds, with the protocol assertion (`min(test_eras) −
  max(refit_train_eras) == 5`) and the example-preds coverage assertion
  fired and printed for every fold used, and TEST-touch counts reported.
- Rank sweep genuinely executed on VALID per the pre-registered staged
  12-trial allocation (arm B's space including LR) on every gated fold,
  with the under-exploration signatures computed and reported.
- Alpha=0 identity and planted-subspace checks passed before any comparison
  was trusted; zero-predictor control run.
- A verdict delivered per the pre-registered decision rule — WINS, HURTS,
  NULL, or one of the pre-registered qualified forms (power-limited NULL;
  scope-limited "no evidence at feasible ranks"; budget-limited "no
  evidence of benefit under the affordable tuning budget") — with (B−A)
  per-fold effect sizes, 95% hierarchical moving-block bootstrap CIs, and
  the standing per-fold MDE table reported regardless of category.

**Solid contribution (strong workshop / short venue paper)**:
- All of the above on all 3 folds (gate passed 3/3), with ≥ 3 (target 5)
  paired seeds per arm per fold.
- Arm C run on the same folds under the A7 invariants, with (B−C) and
  (C−A) reported and C's absolute performance shown — the mechanism
  attribution (spectral selection vs generic low-rank projection) resolved
  either way, with fair-control evidence for any B ≫ C claim.
- Bootstrap CI stability demonstrated across ≥ 2 RNG seeds; seed-variance
  handling reported; cross-fold (B−A) correlation reported; eigh-fallback
  firing counts reported; measured seconds/step for both arms reported;
  adaptive-arm k(t) trajectories and distance-from-identity diagnostics
  reported.
- If negative: the kill criterion invoked cleanly through all four
  conditions, with the negative connected to the literature's predicted
  failure modes (Feldman long-tail; prior weak-signal finding) — a
  boundary-delineating result for coherence-amplifying optimizers.

**Strong contribution (best-case for this run's scope)**:
- A WINS verdict where B beats both A and C with CIs excluding zero on
  ≥ 2 of 3 folds, with fair-control evidence — evidence that *spectral
  selection specifically* (not generic low-rank projection) transfers to
  low-SNR financial regression; first such evidence in an unoccupied niche.
- Or: a HURTS/NULL verdict of equal rigor plus a mechanistic account
  (selected-rank trajectories, filtered-vs-unfiltered cosine and kept-norm
  diagnostics, VALID-vs-TEST sweep behaviour, tail-era breakdown)
  explaining *why* the transfer fails — converting the negative into a
  characterisation of when gradient-consensus filtering helps.

### What Counts as an Informative Outcome

Positive, negative, and null results are treated as equally valuable. An
outcome is informative and publishable if:

- The verdict (WINS, HURTS, NULL, or a pre-registered qualified form) was
  produced under the pre-registered protocol: gate passed on the folds
  used, both protocol assertions fired per fold, rank swept on VALID per
  the staged matched-budget allocation, TEST touched once per fold for the
  comparison with all gate-retry touches logged and reported — these
  conditions apply identically whether the effect is present, absent, or
  reversed.
- Effect sizes with valid (hierarchical moving-block, RNG-stable) CIs
  **and per-fold MDEs** are reported for (B−A), (B−C), (C−A) per fold
  regardless of the verdict category.
- The mechanism question is answered honestly and symmetrically: any
  B-over-A claim is qualified by the B-vs-C comparison, and any B-over-C
  claim is qualified by the fair-control evidence.
- A NULL or HURTS that satisfies all four kill conditions triggers the kill
  criterion and is written up as a clean negative — a successful, complete
  outcome of this run, not a failure mode. A qualified NULL/HURTS
  (power-limited, scope-limited, or budget-limited per A1/A2/A3) is
  likewise informative and publishable: it bounds the effect, states
  exactly what the design could see, and carries a resourced decisive-test
  ask — it simply does not declare the line dead.
- A terminal gate failure is informative as the pre-registered
  **baseline-construction negative**: "the gate was unreachable under this
  budget", stated plainly with the realized ratios, the fix-ladder
  trajectory, and — for a structural (< 0.45) gap — the data-scaling
  learning curve converting the failure into evidence on assumption 3.
  This is a different and weaker finding than a verdict, and is labeled as
  such.

An outcome is **not** informative (the run's only true failure modes) if:
the baseline gate failed on ≥ 2 folds and the pre-specified ladder and
bands were not followed or reported; or the rank sweep was truncated below
the matched budget without the truncation being the pre-registered
feasibility capping; or the protocol assertions did not fire; or TEST was
touched for selection outside the logged gate-retry semantics.

## Minimum Viable Contribution

A verdict on the motivating question — WINS / HURTS / NULL (or its
pre-registered qualified form) per the decision rule — delivered against a
baseline that demonstrably works (gate ≥ 0.60× the floored example-model
denominator passed per fold), with rank tuned on VALID via the staged
allocation rather than guessed, on a split where TEST begins exactly one
embargo (5 eras) after the refit training data ends, with the per-fold MDE
table making the strength of every claim auditable. If only 2 of 3 folds
pass the gate, a verdict on those 2 folds with the third fold's gate
failure reported explicitly still meets this bar (the decision rule's
"≥ 2 of 3" then reads over the gated folds, and this limitation is
stated). Seeds are reduced before folds if compute forces a cut (floor 3),
raised to 5 if packing permits, and either adjustment is reported.

## Compute Feasibility Constraint (binding on all criteria)

All criteria above must be achievable on the MATS Slurm free `compute`
partition: L40 GPUs, 1 GPU per job, experiments targeting under ~30
GPU-min each, max 5 experiments. This fits: the p×p filter is ~2× Adam and
the parent's AdamW runs were ~3 s; the cost drivers are the full v5.0
feature set and re-tuned longer schedules on expanding-window training
sets — which is exactly why the pre-flight unit's full-length arm-A
convergence run (A4) replaces the ~500×-extrapolated cost anchor with a
measured one before fold jobs are sized. Anything that does not fit (Muon
baseline, second architecture, more folds, extra seeds beyond 5, other
datasets, rank points above the measured feasibility cap) is **future
work, not a criterion**, and its absence is not a Step 10 audit finding
against the run — though a realized K_max < 512 changes what a negative is
allowed to claim (A2). The 5-experiment cap is read as a cap on
**submitted experiment units**, with short trials batched inside jobs;
Step 5 must size the decomposition to this. If a fold will not fit, seeds
are reduced before folds (floor 3), and the reduction is documented.

## Risks to Success

- **Baseline gate unreachable within budget** (pre-mortem scenario 1,
  ~half the failure mass): the 0.60× threshold assumes the parent's
  shortfall was protocol-induced rather than intrinsic. Mitigations, now
  binding: per-fold denominators computed locally before protocol freeze
  with the degenerate-yardstick floor; the gate checked before any
  comparison spend; the pre-specified fix ladder with per-rung budgets,
  stopping rule, near-miss band, and structural-gap cutoff; the pre-flight
  proxy gate ratio from the full-length arm-A run. If still unreachable,
  the run reports the baseline-construction negative plus the data-scaling
  learning curve — an informative, if weaker, finding.
- **NULL by power, killed by rule** (pre-mortem scenario 2): ~110
  autocorrelated TEST eras per fold may only resolve effects ≥ ~0.004–
  0.008. Mitigations, now binding: the A1 power qualification on the kill
  criterion, the standing MDE table, seeds raised to 5 where packing
  permits, and the component-level power simulation feeding the frozen
  wording before any TEST touch.
- **Feasibility-capped rank grid stops covering the hypothesis**
  (pre-mortem scenario 3): the useful-rank region at p ≈ 600k may lie
  above what is affordable per step. Mitigations, now binding: the A2
  kill-scope floor (K_max ≥ 512), realized-grid recording, adaptive k(t)
  logging, and pre-flight timing of eigh-every-N-steps / GPU-fp32-eigh
  variants (named documented variants if ever used).
- **Under-explored arm-B sweep produces an artifact HURTS**: mitigated by
  the A4 staged allocation (resolvable by construction) and the A3
  signature-triggered downgrade — the kill criterion cannot fire on a
  budget artifact.
- **Arm C matches the wrong invariants** (pre-mortem scenario 5): a
  parameter-space port of the sample-space control is norm-matched noise
  injection. Mitigations, now binding: A7's re-derived invariants
  (k(t) + norm ratio + basis-rotation rate; preferred rotated-own-basis
  design), the required pre-port CPU simulation, absolute reporting of C,
  and the symmetric honesty clause.
- **Expanding-window training sets exceed the ~30 GPU-min job envelope**:
  mitigation — the pre-flight full-length convergence run sizes fold jobs
  from measurement; reduce seeds before folds; document any feature-set
  reduction as a limitation; shard construction reuses parent code.
- **12 trials × 3+ arms × 3 folds strains the 5-experiment cap**:
  mitigation is batching (one Slurm job runs many short trials
  sequentially) and the experiment-unit reading of the cap, sized in
  decomposition (Step 5).
- **Numerical instability of the streaming rank-1 SVD / per-step eigh over
  longer schedules**: mitigation is the mandated CPU-fp64 eigh fallback
  with logged firing counts (parent audit Finding 5 patch), plus the
  alpha=0 identity and planted-subspace checks as integration canaries.
- **Bootstrap validity on autocorrelated per-era series**: block length is
  recomputed per fold from the realized lag-1 ACF, seed variance is
  handled hierarchically or shown negligible, RNG is fixed and stated, and
  stability is checked across ≥ 2 RNG seeds — no inherited parameters.
- **Selection leakage**: all selection on VALID; TEST evaluated once per
  fold for the comparison after refit; gate retries constrained and logged
  per the A5 retry semantics; the protocol (this file + `protocol.json`)
  frozen before TEST is touched. The Step 10 audit checks the timestamps
  and logs against this file.
- **Fold non-independence inflates apparent replication**: accepted as a
  claim-calibration item (A6) — nested-fold language and the reported
  cross-fold correlation, no design change.
- **A same-idea preprint appearing mid-run / nearest-neighbor drift from
  the object swap** (novelty caveats): does not change the run's
  execution; the targeted mini-search (GaLore-style projection,
  momentum-subspace optimizers, K-FAC lineage) runs before write-up, the
  search is dated, and the write-up positions against GAF explicitly
  rather than claiming a vacuum — with the mechanism described as
  **temporal mean-gradient-subspace filtering** (EMA over steps), not
  per-sample consensus, and interpretive claims gated on the A8
  diagnostics.
