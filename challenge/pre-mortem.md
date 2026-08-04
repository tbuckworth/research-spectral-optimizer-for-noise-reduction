# Pre-Mortem Analysis (Round 2)

**Scope note**: This is the challenge round-2 pre-mortem, written against the ROUND-2
success criteria (A1–A8 integrated) and ROUND-2 decomposition (B1–B4 integrated).
Round 1's five scenarios were largely converted by the amendments into pre-registered
qualified outcomes — that conversion is itself the subject of several scenarios below.
The question this pass asks is: given the *amended* plan, how does the run still end
without the answer it was commissioned to deliver — including failure modes the
amendments introduced?

## Setting

It is six months from now. The research on "Spectral Optimizer redo: p×p filter,
walk-forward split, rank sweep" is complete. The run did not deliver the decisive
WINS/HURTS/NULL verdict it was commissioned — for the second time on this question —
to produce. Working backward, here is how that happened.

## Scenarios (ranked by priority)

### 1. The gate was unreachable, and the proxy ratio mis-called it [Likelihood: High | Severity: High]

**What happened**: The run bet on assumption 3 — that the parent's 27%-of-example-model
baseline was protocol-induced, not intrinsic. It was partly intrinsic. The gate asks a
12-trial-tuned MLP to reach ≥ 0.60× the Numerai example model (a boosted-tree ensemble,
the domain's strongest public reference) on the *most recent* ~330 eras — the hardest,
lowest-signal regime in the dataset, exactly where practitioner MLPs underperform GBTs
most. The parent's best tuned MLP hit 0.0196 on older, easier eras; 0.60× ≈ 0.014
absolute on recent eras was close, but the realized fold-1 ratio came in at 0.51 — the
near-miss band. One targeted regularization rung moved it to 0.56. Per the
pre-registered band, that outcome was accepted: fail. Fold 2 landed at 0.47, and the
ladder's futility stop fired after two rungs improved the combined ratio by 0.015. Two
failed gates of three: the run terminated with the pre-registered baseline-construction
negative — its second consecutive report about the baseline rather than the filter.

A secondary mechanism made it worse: the EU-1 proxy gate ratio, computed on a VALID
slice, read 0.63. The VALID slice sits 8+ eras earlier than TEST and had a friendlier
regime; the proxy's false pass meant the reserve unit was not pre-positioned for
baseline work, and fold-1's near-miss arrived as a surprise after EU-2 was already
sized for the full three-arm comparison. The proxy was pre-registered as an early
warning but its own calibration (VALID-slice vs TEST-block regime shift) was never
checked — it was a thermometer nobody had tested against the thing it predicts.

**Root cause**: Assumption 3 is the run's dominant untested premise (the decomposition
prices it at 0.55 and the amendments deliberately made its failure *cheaper*, not less
likely), and the one instrument added to buy early warning — the VALID-slice proxy —
measures a different distribution than the gate it forecasts.

**Early warning signs**:
- Before any GPU: the per-fold example-model denominators (computed locally from
  `out/example_per_era_corr.csv` per A5) show the recent TEST blocks are
  high-denominator/high-volatility relative to the parent's tuning block — the gate's
  absolute bar in per-era corr is then quantifiable weeks before arm A exists.
- EU-1: proxy ratio in [0.45, 0.65] — anywhere near the boundary means the proxy's
  VALID-vs-TEST regime offset (which can be computed for the *example model* for free,
  since its per-era corr exists on both slices) decides the forecast, not the ratio.
- Fold 1: realized ratio in [0.45, 0.60) — the bands then govern; the question is
  whether the reserve is still intact when they do.

**Mitigation**:
- Calibrate the proxy at zero cost: compute the example model's own VALID-slice vs
  TEST-block mean per-era corr for every fold from the existing CSV, and report the
  proxy *with* that offset applied. This turns an uncalibrated forecast into a
  calibrated one before EU-1 is even submitted.
- Pre-commit the reserve-allocation decision to the *calibrated* proxy: calibrated
  proxy < 0.60 → EU-5 is provisionally earmarked for the ladder before EU-2 launches,
  and EU-2's phase-2 (arms B/C) is made conditional in the sbatch structure, not just
  in prose.
- If fold-1 lands in the near-miss band, choose the targeted rung using the parent's
  F12 signatures *recomputed on this run's arm-A sweep* (grid-boundary position of the
  selected config), not the parent's memory of them.

**Pivot indicator**: Calibrated EU-1 proxy < 0.50, or fold-1 realized ratio < 0.45 —
at that point stop treating this as a comparison run: the deliverable is the
data-scaling learning curve and the baseline-construction negative, and the remaining
budget should be spent making *that* finding sharp (learning curve + regime analysis)
rather than keeping folds 2–3 alive.

**Load-bearing assumption**: "The parent's baseline shortfall was protocol-induced,
not intrinsic to tuned MLPs on recent-era Numerai data" (state.md assumption 3) — plus
the unstated sub-assumption that a VALID-slice ratio predicts a TEST-block ratio.

---

### 2. EU-1 collapsed under its amendment cargo, and the sizing chain collapsed with it [Likelihood: High | Severity: Medium]

**What happened**: Round-2 EU-1 carried: largest-shard build (int8, 4–6M rows),
s/step for arms A and B across five rank points at *two* architectures, eigh-every-N
and GPU-fp32-eigh variant timings, VRAM peaks, 500-step stability at grid extremes,
an on-cluster alpha=0 re-assert, adaptive k(t) and rotation-rate logging, and — the
B1 centerpiece — one full-length arm-A convergence run with a VALID-slice proxy gate
ratio. It was submitted with `--qos=debug` under the 2-hour ceiling. The circularity
was structural: the convergence run's duration is exactly the unknown quantity EU-1
exists to measure, so no `--time` could be chosen correctly in advance. Shard build
took 35 minutes, the two-architecture rank sweep took 40, and the convergence run was
still climbing on the VALID slice when the job hit the wall. No plateau, no measured
steps-to-convergence, a truncated proxy ratio.

Downstream, everything the amendments had routed through that measurement was starved:
#1's packing arithmetic reverted to extrapolating the truncated curve (the exact
~500×-extrapolation failure B1 existed to eliminate); the A4 step count and the refit
stopping rule's scale went into the frozen `protocol.json` as extrapolations; and the
resubmitted EU-1 continuation consumed the reserve's slack, so when fold-1's gate came
in at a near-miss (scenario 1), the ladder had no unit left. The two highest-priority
risks fired *together* because the amendment that mitigated one consumed the buffer
for the other.

**Root cause**: The B1 amendment concentrated eight measurement objectives — one of
unbounded a-priori duration — into a single pre-sized job, recreating inside EU-1 the
atomic-job fragility that B2 was busy removing from the fold jobs.

**Early warning signs**:
- Local, pre-submission: a back-of-envelope steps-to-plateau estimate from the
  parent's loss curves scaled by rows says the convergence run alone projects > 60
  min — then the 2 h debug window cannot hold the full cargo.
- In-flight (first 30 min of EU-1): shard build + first-architecture timings have
  consumed > 45 min — the convergence run will be truncated.

**Mitigation**:
- Restructure EU-1 internally as ordered, individually-persisted phases: (1) shard
  build → persist; (2) timings/VRAM/variants at both architectures → persist a
  complete timing table; (3) identity re-assert + stability; (4) convergence run
  LAST, checkpointed every N steps with the VALID-slice score series streamed to NFS.
  A wall-clock kill then costs only the tail of phase 4, and the checkpoint makes the
  continuation a resume, not a redo.
- Do not use `--qos=debug` for EU-1 as scoped, or split it: EU-1a (phases 1–3,
  debug QOS, < 2 h) and EU-1b (convergence run, normal queue, `--time=6h`,
  resumable). Two submissions — the packing plan must decide *now* whether that
  second submission is inside EU-1's unit accounting or pre-charged to the reserve,
  rather than discovering the charge mid-flight.
- Pre-register the fallback anchor: if the convergence run is truncated, the step
  count entering `protocol.json` is the checkpointed lower bound plus a stated
  extrapolation rule — declared as such, so the audit sees an anchored estimate, not
  a silent regression to guessing.

**Pivot indicator**: EU-1 (or EU-1b) ends twice without a VALID-slice plateau — stop
buying the measurement; freeze the protocol on the declared lower-bound-plus-rule
anchor and let the refit loss curves (already logged per A4) carry the post-hoc check.

**Load-bearing assumption**: "One `--qos=debug` pre-flight job can deliver every
measurement the 11-item protocol freeze requires, including a full-scale convergence
run whose duration is the thing being measured."

---

### 3. The freeze locked in a defect, and the pre-registration machinery turned it into the verdict [Likelihood: Medium | Severity: High]

**What happened**: The amendments roughly quadrupled the pre-registered surface: an
11-item checklist, every item filled from a single-shot local sim or one debug job, in
autonomous mode, with the challenge loop capped and no legitimate revision after
freeze. One frozen quantity was wrong, and the machinery did what frozen machinery
does. The specific defect: the A4 stage-1 design centers arm B's sweep on a
*transferred* LR from arm A. But filtering changes the gradient norm delivered to
AdamW (the kept-norm fraction at low rank is well below 1), so arm B's optimal LR sat
far from arm A's. Stage-1 rankings across the rank grid were scrambled by the
mis-centered LR — every grid point handicapped by a different amount — stage 2 refined
around a wrong winner, and the selected config's LR ended ≥ 4× from arm A's. That is
precisely A3's LR-shift signature: it fired, as designed, on every fold.

The result was the pre-registered downgrade cascade doing exactly what it was frozen
to do: the (B−A) < 0 result on 2 of 3 folds — which may have been a *true* HURTS —
was downgraded to "no evidence of benefit under the affordable tuning budget". The
kill criterion could not fire. The run ended with a verdict the plan could have
predicted at freeze time from the kept-norm diagnostics in EU-1's own short arm-B
runs: the signature's firing was foreseeable *before* any fold job, and nothing in
the frozen protocol used that foresight. A parallel version of the same failure sat
in the refit stopping rule — steps × rows_refit/rows_train assumes convergence scales
linearly in rows; the logged refit loss curves later showed fold-3's refit stopped
short of plateau, contaminating both the gate ratio and (B−A) on the largest fold —
visible post hoc, unfixable post freeze.

**Root cause**: The pre-registration surface grew faster than the evidence available
to fill it. Each frozen rule is only as good as its single-shot input, and the A3/A4
interaction has a foreseeable self-triggering mode: a mis-centered transferred LR
*causes* the LR-shift signature that then voids the verdict.

**Early warning signs**:
- EU-1's short arm-B runs: kept-norm fraction at the mid-grid ranks well below ~0.7 —
  the transferred LR will be mis-centered, and the LR-shift signature is on track to
  fire before any fold job exists.
- Fold-1 stage 1: VALID scores non-monotone across the rank grid with range under the
  across-seed sd (A3 signature 2) — the sweep cannot rank its grid at the transferred
  LR.
- Fold-1 refit: loss slope at the stopping-rule step visibly nonzero — the linear-rows
  scaling is off.

**Mitigation**:
- Spend one of stage 1's 7–8 trials as a pre-registered LR probe: the modal rank point
  at transferred-LR × (1/kept-norm-fraction measured in EU-1). If the probe beats the
  transferred-LR twin, stage 1's centering is corrected *by a frozen rule*, not by
  discretion — this closes the self-triggering loop between A4 and A3 at a cost of
  one trial.
- Freeze a deterministic refit-extension rule alongside the stopping rule: if refit
  loss slope over the last 5% of steps exceeds a stated threshold, extend once by a
  stated factor and log it. Pre-registered now, it is a rule; discovered later, it is
  a violation.
- Treat EU-1's kept-norm diagnostics as a freeze *input*, not just a run-time log:
  item 4 of the checklist (transferred LR) should be written as a function of them.

**Pivot indicator**: Two or more A3 signatures fire on fold 1 — stop before submitting
folds 2–3. Two more folds of auto-downgraded results add nothing; the reserve is
better spent on a re-centered fold-1 VALID sweep (selection on VALID only — no
TEST-touch issue) so that at least one fold's comparison is unimpeached.

**Load-bearing assumption**: "Every quantity the 11-item freeze requires can be set
correctly from one local sim pass plus one pre-flight job — in particular, arm A's
tuned LR is an adequate center for arm B's stage-1 sweep."

---

### 4. The run completed cleanly and delivered a triple-hedged non-answer [Likelihood: High | Severity: Medium]

**What happened**: Nothing broke. The gate passed on 2 of 3 folds (fold 2 near-miss,
accepted-fail, reported). EU-1's cubic-eigh wall capped the realized grid at
K_max = 128 even under the timed variants — the A2 scope wording went into
`protocol.json` the day the grid was realized, exactly as designed. The #3 power sim
had returned FAIL (recent-window ACF near the example model's 0.763 shrank the
effective era count; the hierarchical seed term widened CIs further), so the A1
wording was load-bearing from the start; realized per-fold MDEs came in at
0.006–0.009 against the 0.005 bar. The verdict, delivered exactly per the frozen
protocol: **"NULL (power-limited, scope-limited): |B−A| bounded within ±0.007 at
feasible ranks k ≤ 128, on 2 gated nested folds; line not declared dead"** — with a
standing MDE table, logged k(t) trajectories, and a resourced decisive-test ask.

Every safeguard worked. And the motivating question — does the p×p filter help on
noisy financial data — stood exactly as open as it did before the parent run. This
run exists because the parent's answer "did not count"; this answer, while honest and
pre-registered, does not count *as an answer* either. It counts as a bound and a
costed ask for a third run. The plan's own arithmetic said this was the expected
outcome: P ≈ 0.06 for an unqualified verdict, and the qualification triggers (gate
margin, power at ~110 autocorrelated eras, rank-grid feasibility) are not independent
— they all descend from the same constraint, the free-partition/5-unit compute
envelope, so they fire together far more often than the product suggests.

**Root cause**: The A1/A2/A3 qualifications protect the verdict's *interpretability*
against artifacts — the correct response to the parent's failure — but nothing in the
amended design added *decisiveness*: no lever raised power (5 seeds narrows the seed
term, not the era term, and era count per fold is fixed by the 3-fold layout), and no
lever guaranteed grid coverage. The design guarantees the run cannot lie; it does not
make the run able to speak.

**Early warning signs**: Both are visible **before any TEST touch**, which is what
makes this scenario actionable rather than fated:
- #3's local power sim FAILs its B4 threshold (P(detect ±0.005) < 0.6 on ≥ 2 folds).
- EU-1 realizes K_max < 512.
  If both hold at freeze time, the frozen protocol is provably incapable of any
  terminal verdict — WINS aside — and the run knows it while redesign is still legal.

**Mitigation** (all legitimate only pre-freeze, which is exactly when the warnings
arrive):
- Pre-register a pooled secondary estimand: the era-level paired (B−A) series
  concatenated across gated folds under one hierarchical moving-block bootstrap
  (~220–330 eras), reported alongside the per-fold rule. It cannot overturn the
  per-fold decision rule, but it converts "three underpowered bounds" into "one
  adequately-powered bound", and its MDE plausibly clears 0.005 when the per-fold ones
  do not.
- Add a pre-freeze branch point to the checklist: if the power sim FAILs *and* EU-1
  caps the grid below 512, re-scope before freezing — the pre-registered option being
  2 folds with proportionally longer TEST blocks (more eras per fold buys the era-term
  power the seed lever cannot), with the fold reduction reported. Trading one nested
  replication for a decision-capable design is a choice the criteria's cut order
  already gestures at (reduce folds last) — it should be a considered branch, not an
  overflow response.
- If the branch is not taken, say so at freeze time in `protocol.json`: "this design's
  modal outcome is a bound, not a verdict" — so the write-up leads with the MDE table
  by plan rather than by apology.

**Pivot indicator**: At the freeze checkpoint: power sim FAIL + K_max < 512 + the
pooled estimand also projecting MDE > 0.005. All three together mean no affordable
version of this design can return a terminal verdict — the honest moves are the
re-scope branch or an explicit decision to run for the bound and title the ask.

**Load-bearing assumption**: "An interpretable qualified outcome satisfies the brief."
The brief's own history (a parent run whose rigorous answer 'did not count') says the
user's bar is decisiveness, not just interpretability.

---

### 5. Arm C was either noise or a clone, and the mechanism claim collapsed either way [Likelihood: Medium | Severity: Medium]

**What happened**: The B3 sim ran as required and delivered its verdict on candidate
(a): a Haar-rotated basis is a uniformly-random k-subspace at each instant — captured
energy ≈ k/p ≈ 0.3%, norm-matching means 30–250× amplification. Noise injection;
correctly rejected, as the round-1 challenge predicted. The design fell to candidate
(b): random rotation *within the span* of B's tracked covariance factorization. The
sim's pass criteria all cleared: k(t) matched by construction, norm ratio ≈ 1 (the
span carries the energy), rotation rate matched, and C descended within 2× of arm A.
What the sim did not test: candidate (b)'s control condition is *almost the treatment*.
Rotating within the retained span preserves exactly the subspace that spectral
selection selected — C differs from B only in which basis of the *same* filtered
subspace weights the update. On TEST, (B−C) was statistically zero on every fold, and
the pre-registered honesty clause did its work: "low-rank projection helps; spectral
selection is not the active ingredient." But that conclusion was baked in by the
control's construction, not discovered about the world — the exact mirror of the
round-1 defect, where a too-weak C made the honesty clause vacuous in the other
direction. The mechanism-attribution contribution — the run's distinctive claim
beyond (B−A) — was lost either way: candidate (a) too far from B, candidate (b) too
close, and the sim's pass criteria only policed the "too far" side.

**Root cause**: In parameter space there may be no control that simultaneously shares
B's captured-energy budget and is genuinely spectrally null — and the B3 sim's
acceptance test (match invariants + descend) checks only one tail of that dilemma.

**Early warning signs**:
- In the sim itself: candidate (b)'s captured-energy fraction ≈ B's kept-norm fraction
  *and* small principal angles between C's realized subspace and B's — the "clone"
  signature, computable in the same 20 minutes.
- In EU-1 short runs: per-step cosine between C's and B's updates persistently high
  (> ~0.8) — C is shadowing B.

**Mitigation**:
- Add a positive control *for the control* to #2's pass criteria: on the #5
  planted-subspace synthetic task — where spectral selection provably matters and
  hard top-k provably recovers the planted direction — an acceptable arm C must
  perform measurably *worse* than arm B. A candidate that matches the invariants,
  descends, and still fails the planted task is a fair control; one that matches B on
  the planted task is a clone and is rejected. This closes the untested tail of the
  dilemma at zero GPU cost, and the acceptance test itself goes into the frozen
  protocol (item 10).
- If no candidate passes both tails, invoke the criteria's existing fallback honestly
  and *early*: drop C, scope the run to (B−A), report the sim as the evidence that a
  fair parameter-space control was not constructible, and spend C's refit-seed budget
  on raising seeds toward 5 (the A6 power lever).

**Pivot indicator**: No candidate passes both the invariant-matching test and the
planted-task discriminability test in the sim — decide *then* (pre-freeze, pre-port)
to run without C, rather than carrying a control whose verdict is predetermined into
three fold jobs.

**Load-bearing assumption**: "A parameter-space control exists that matches k(t),
norm ratio, and rotation rate while remaining spectrally null — and the B3 sim as
specified would notice if it didn't."

---

## Cross-Cutting Themes

1. **Everything routes through one measurement pass into an irreversible freeze.**
   The amendments moved nearly every safeguard's parameters into `protocol.json`,
   filled from single-shot inputs: one debug job (EU-1) feeds checklist items 3, 4,
   5, 7, 11; two local sims feed items 6, 9, 10. Scenarios 2, 3, and 5 are all the
   same failure at different checklist items: a single-shot input is wrong or
   truncated, the freeze happens anyway (autonomous mode has no revision loop), and
   pre-registration converts the defect into either an artifact verdict or a
   protocol violation. The freeze checklist needs the property the fold jobs got
   from B2: per-phase persistence and a defined resume/fallback for every input.

2. **The amendments bought interpretability; nothing bought decisiveness.** A1/A2/A3
   ensure the run cannot report an artifact as a verdict — the correct lesson from
   the parent. But the qualification triggers (gate margin, per-fold power, grid
   feasibility) all descend from the same compute envelope and are strongly
   positively correlated: the plan's 0.06 unqualified-verdict figure is a product of
   correlated terms, and the modal clean outcome is a multiply-qualified bound. For
   a run commissioned because the previous answer "did not count", that is the
   central systemic risk — scenario 4 is not a malfunction of the plan but its
   expected behaviour, knowable at freeze time.

3. **The instruments themselves are uncalibrated.** The two early-warning devices the
   amendments added — the VALID-slice proxy gate ratio (scenario 1) and the B3 sim's
   acceptance test (scenario 5) — each measure a proxy whose relationship to the
   target (TEST-block ratio; fair-control discriminability) was never itself checked.
   Both calibrations are free (a pandas job on an existing CSV; one extra sim check).

## Summary Risk Profile

| Scenario | Likelihood | Severity | Priority | Mitigable? |
|----------|-----------|----------|----------|------------|
| 1. Gate unreachable; proxy mis-calls it | High | High | Critical | Partially — bands/ladder cap the cost; assumption 3 itself is only testable by running; proxy calibration is free |
| 2. EU-1 collapses under amendment cargo | High | Medium | High | Yes — phase-ordered persistence + EU-1a/EU-1b split, decided pre-submission |
| 3. Freeze-locked defect (LR transfer → A3 self-trigger; refit rule) | Medium | High | High | Yes — LR-probe trial, refit-extension rule, kept-norm as a freeze input; all pre-freeze wording |
| 4. Clean run, triple-hedged non-answer | High | Medium | High | Partially — pooled estimand + pre-freeze re-scope branch raise decisiveness; the compute envelope's power ceiling remains |
| 5. Arm C noise-or-clone dilemma | Medium | Medium | Medium | Yes — planted-task discriminability added to the sim's pass criteria; honest early drop if it fails |

## Top Recommendations

1. **Make the freeze inputs resumable and calibrated before anything is submitted.**
   Restructure EU-1 as ordered, individually-persisted phases with the convergence run
   last and checkpointed (or split EU-1a/EU-1b with the unit accounting decided now);
   calibrate the proxy gate ratio with the example model's own VALID-vs-TEST offset
   from the existing CSV. Both are zero-GPU-cost changes to scenario territory worth
   roughly half the failure mass.
2. **Close the two self-defeating loops in the frozen protocol.** (a) One stage-1
   trial becomes a kept-norm-corrected LR probe so the A4 transferred-LR centering
   cannot mechanically trigger the A3 LR-shift signature; (b) add the planted-task
   discriminability requirement to arm C's acceptance test so the honesty clause
   cannot be satisfied by a clone. Both are wording changes, legitimate only now.
3. **Add the pre-freeze decisiveness checkpoint.** If the #3 power sim FAILs its B4
   threshold *and* EU-1 realizes K_max < 512, the design provably cannot return a
   terminal verdict: pre-register the branch (pooled cross-fold secondary estimand;
   optionally 2 longer-TEST folds instead of 3) and take it deliberately at freeze
   time — or record, in `protocol.json`, the explicit decision to run for a bound.

## Residual Risk

After all mitigations, three risks remain and should be stated plainly. First, the
baseline gate: assumption 3 is genuinely ~50/50 and no amount of protocol machinery
changes it — the amendments only make its failure cheap and informative. A
baseline-construction negative (a second consecutive report about the baseline, not
the filter) remains a fully live outcome. Second, the power ceiling: ~110
autocorrelated eras per fold, with era-level variance the seed lever cannot touch, may
bound decisiveness regardless of design; the pooled estimand and the 2-fold branch
mitigate but do not remove this — the free-partition compute envelope, not the
protocol, sets the ceiling. Third, by the plan's own arithmetic the modal outcome is a
qualified finding, not a terminal verdict: even executed flawlessly, this run is more
likely to bound the effect and price the decisive experiment than to deliver the kill
or the win itself. That is acceptable only if the user accepts that "the decisive
test" may be the *third* run's job — and the write-up should be planned, from the
start, to make that third run's design and cost its most valuable output if the
qualifications fire.
