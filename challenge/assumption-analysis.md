# Assumption Analysis (Round 2)

**Run**: 2026-08-03-followup-follow-up-spectral-optimizer-redo-pxp-filter-walk-
**Scope note**: This is the round-2 challenge pass after the MAJOR_REVISIONS loop-back. It replaces the round-1 analysis. Round-1 findings that amendments A1–A8 / B1–B4 have addressed are not re-listed; this pass (a) verifies the amendments are actually implemented in the round-2 documents, and (b) surfaces assumptions that remain or were newly introduced *by* the amendments.

## Amendment Implementation Check (requested verification)

All twelve amendments are verifiably present in the round-2 artefacts, not merely claimed:

| Amendment | Located at | Status |
|---|---|---|
| A1 power-qualified kill + standing MDE table | criteria: kill criterion cond. 2; metrics/inference section | Implemented |
| A2 kill-scope floor (K_max < 512) + k(t) logging | criteria: kill cond. 3; sweep section | Implemented |
| A3 under-exploration signatures + HURTS downgrade | criteria: sweep section (3 signatures); kill cond. 4 | Implemented (one operational gap — see Moderate #8) |
| A4 staged 12-trial allocation, pruning, steps outside the 12, refit stopping rule | criteria: sweep section; splits section | Implemented (minor arithmetic gap — see Background #13) |
| A5 gate semantics (local denominators, 0.010 floor, bands, 5-rung ladder 4/4/2/1/1, stopping rule, p-preservation, retry semantics, weighting lever) | criteria: baseline-gate section | Implemented |
| A6 hierarchical bootstrap, seeds 5-target/3-floor, nested-fold language, cross-fold correlation | criteria: metrics/inference section | Implemented |
| A7 arm C invariants (k(t), norm ratio, rotation rate), rotated-own-basis, required CPU sim, absolute reporting, symmetric honesty | criteria: baselines §2 + decision rule; decomposition #2 | Implemented |
| A8 planted-subspace check, cosine/kept-norm/distance-from-identity logging, coverage assertion, era mapping, decay/warmup limitation | criteria: verification + splits sections; decomposition #5, #9 | Implemented |
| B1 EU-1 convergence run + proxy gate ratio + 2nd-architecture + eigh-variant timings | decomposition #8, #1, EU-1 row, change log | Implemented |
| B2 resumability component before submission | decomposition #7 (new), EU-2/3/4 requirements | Implemented |
| B3 arm C re-derived + pre-port sim + rotation diagnostic | decomposition #2 (rewritten), #5, #8 | Implemented |
| B4 power-sim consequence wired to mandatory A1 wording | decomposition #3 fail branch; freeze checklist item 6 | Implemented |

The decomposition additionally exercised the delegated decay decision (decay=0.999 stage-2 probe) and honors the change-log indexing. The regeneration is genuine, and the round-2 P_success drop (0.09 → 0.06 unqualified) is honest accounting, not drift.

## Summary

Thirteen assumptions found: 4 critical, 6 moderate, 3 background. The critical cluster is new in kind: round 1's defects were *missing machinery* (no F12 guard, unspecified allocation, unexamined denominator); the amendments built that machinery, and the remaining risk now lives in **what the machinery itself quietly assumes** — that a globally fixed step count is fair to both arms, that a fair parameter-space control exists at all between two degeneracies, that the 0.005 MDE floor (which is a ~50%-power half-width, and ~35% of the gated baseline's own score) makes a "clean kill" mean what it sounds like, and that a transferred-LR stage-1 sweep ranks ranks correctly. Three of the four feed the same construct — "genuine sweep at matched budget" — which is exactly the construct the kill criterion's condition 1 certifies.

## Critical Assumptions (Low confidence, High impact)

### 1. Arm B converges on arm A's schedule: a step count fixed by the arm-A convergence run is fair to the filtered arm

- **Category**: Methodological / Baseline
- **Confidence**: Low
- **Currently assumed because**: A4 (correctly) pulled the converged step count out of the 12-trial budget and fixed it via EU-1's **arm-A** full-length convergence run. Matched steps reads as fairness — the same compute for both arms — and the amendment's attention was on cost-anchoring, not on whose convergence curve defines "converged".
- **What changes if wrong**: The filter discards gradient energy every step (kept-norm fraction < 1 by construction, especially at low rank), and adds a covariance-warmup phase. If filtered training converges slower than plain AdamW — the default expectation for any projection method — then arm B is evaluated systematically short of its own plateau at every rank, biasing (B−A) toward HURTS/NULL. None of the three A3 signatures detects this: B's best config can sit mid-grid with a sane LR and still be under-trained. The kill criterion's condition 1 ("sweep genuinely ran at matched budget") would certify a comparison whose budget was matched in steps but not in convergence — the parent's failure shape (a safeguard verifying execution while the conclusion goes unlicensed) one level down.
- **How to test**: Free, pre-freeze: log VALID-score-vs-step curves for every arm-B trial (the infrastructure exists — EU-1 already monitors plateau for arm A) and pre-register a fourth under-exploration signature: *arm B's selected config still improving at step cutoff (VALID slope over the last X% of steps > threshold)* → same downgrade path as A3. Alternatively/additionally, calibrate the claim language now: the verdict is "at matched step budget", stated as such. EU-1 can also cheaply run one short arm-B config alongside the arm-A convergence run and compare plateau fractions.
- **Relevant evidence**: The filter's own H3 finding (rank ≤ 4 destabilizes, ~10 *fastest*) is evidence that convergence speed is strongly rank-dependent — i.e., a single global step count interacts with the swept variable itself.

### 2. A fair parameter-space arm C exists between two degeneracies — and candidate (b) as written may be a no-op

- **Category**: Methodological / Theoretical
- **Confidence**: Low
- **Currently assumed because**: A7/B3 correctly killed the round-1 fixed-subspace port (captures ~k/p ≈ 0.3% energy → noise injection) and specified invariants plus three candidates with a required sim. The amendment assumes at least one candidate threads the needle; the sim tests capture-energy and descent but not distinctness from B.
- **What changes if wrong**: The candidates bracket a possibly empty middle. Candidate (a) (Haar rotation of B's basis) is, instant-by-instant, a uniformly random k-subspace — the decomposition itself concedes it may capture ~k/p energy, i.e., degenerate toward noise injection. But candidate (b) has the *opposite* degeneracy, unstated: **projection onto a subspace is basis-invariant**, so a "random rotation within the span of B's tracked factorization" that stays inside the retained k-dimensional span produces *identical* updates to arm B — C ≡ B, and (B−C) ≈ 0 by construction, which would spuriously "prove" spectral selection is not the active ingredient. Candidate (b) is only meaningful if it rotates within a strictly larger ambient history space (r > k) and then truncates to k — and how much larger r is determines where C sits on the B↔noise axis. If no candidate is simultaneously (i) energy-viable, (ii) invariant-matched, and (iii) *distinct from B*, the mechanism-attribution deliverable is unavailable, and — worse than unavailable — a degenerate C in either direction produces a confident wrong mechanism verdict rather than a missing one.
- **How to test**: Extend the already-required B3 CPU sim with a **distinctness criterion**: C's per-step update must differ from B's by more than seed-level noise (e.g., mean principal angle between kept subspaces above a pre-registered floor, and per-step update cosine to B's update below a ceiling), alongside the existing captured-energy and descent checks. Pre-register now what gets claimed if the sim shows the middle is empty ("no fair parameter-space control exists at this k/p; mechanism attribution out of scope"), so the fallback is a scoped statement rather than a straw control.
- **Relevant evidence**: The decomposition's own risk note on (a); basic linear algebra for (b); parent run's straw-control experience (pre-mortem scenario 5) showing degenerate controls fail toward false confidence, not toward visible breakage.

### 3. MDE ≤ 0.005 means the kill is adequately powered — but the MDE is a ~50%-power half-width, and 0.005 is ~35% of the gated baseline's own score

- **Category**: Methodological / Scope
- **Confidence**: Low
- **Currently assumed because**: A1 imported 0.005 as the power floor (echoing the parent's −0.00527 effect size), and operationalized MDE as the realized 95% CI half-width — the natural quantity the bootstrap already produces.
- **What changes if wrong**: Two stacked calibration gaps. (i) A CI excludes zero when the estimate exceeds the half-width, so a true effect exactly at the half-width is detected with ~50% probability — "MDE ≤ 0.005" licenses a kill under roughly coin-flip power at the boundary effect, which is weaker than the phrase "adequately powered NULL" suggests (B4's sim threshold P ≥ 0.6 is consistent with this, but the criteria text and the sim threshold are never connected). (ii) The floor itself: the gate targets arm A ≈ 0.014 (0.60 × 0.0235), so a clean kill bounds only |B−A| < 0.005 ≈ **35% relative improvement**. In a domain the criteria themselves describe as trading on edges of 0.01–0.05, an optimizer worth a true +0.003/era (~20% relative) survives every fold's CI, satisfies all four kill conditions, and the line is declared "dead for financial tabular regression". The parent challenge flagged the analogous threshold-relativity problem (its item 7); the amendment carried the absolute number over without re-deriving it against the realized baseline.
- **How to test**: Free, pre-freeze: state in `protocol.json` what the kill's bound *means* in relative terms (e.g., "dead ⇔ any true effect ≥ X% of realized arm-A per-era corr excluded at ~50% boundary power"), or re-anchor the terminal-NULL condition on the quantity B4 already computes — P(detect | true effect ±0.005) per fold — rather than the half-width. At minimum, rename the reported quantity ("CI half-width (≈50%-power MDE)") so the audit and the write-up cannot equivocate.
- **Relevant evidence**: Standard power arithmetic; criteria's own SOTA table (0.0235 yardstick, 0.60 gate); parent exp-003 half-width 0.00219 at 255 eras scaling to ~0.0033–0.005 at 110 eras — i.e., the design will plausibly sit *exactly at* the boundary where this calibration gap matters most.

### 4. A transferred LR ranks the rank grid correctly at stage 1

- **Category**: Methodological
- **Confidence**: Low
- **Currently assumed because**: A4's staged design needs stage 1 cheap (one trial per grid point), so all rank points share one transferred LR; LR refinement is deferred to stage 2 *around the winner only*. The pruning was justified as "a resolvable sweep over a smaller space beats an unresolvable sweep" — true, but resolvability was evaluated as trial-count arithmetic, not as estimator bias.
- **What changes if wrong**: The filter's effective step size shrinks with the kept-norm fraction, which shrinks with k. At a fixed transferred LR (presumably tuned for unfiltered arm A), low-rank configs run at systematically smaller effective LR than high-rank/near-identity configs — stage 1 then ranks "which rank tolerates arm A's LR" rather than "which rank is best at its own LR", predictably selecting large-k/near-identity winners. Stage 2 refines LR only for that winner, so mis-ranked low-rank points never get their fair LR. The A3 LR-shift signature fires only on the *winner's* LR shift — a winner selected close to identity will show *no* shift, so the guard is silent precisely when the confound succeeded. Downstream: a NULL with small distance-from-identity (which A8 will dutifully report as "VALID selected a near-identity configuration") could be an artifact of stage-1 LR transfer, not a fact about the filter.
- **How to test**: Free, pre-freeze, using diagnostics A8 already mandates: scale the transferred LR per grid point by the (measurable in EU-1) expected kept-norm fraction at that rank — one line of arithmetic per trial — or pre-register "stage-1 winner within one grid step of identity-distance minimum" as an additional under-exploration signature feeding the A3 downgrade.
- **Relevant evidence**: Criteria's own rationale for putting LR in arm B's space ("filtering changes the effective step size") — the assumption contradicts the design's own stated premise, one stage earlier.

## Moderate Assumptions (Medium confidence or Medium impact)

### 5. A low-signal-flagged fold is still a valid vote in the ≥2/3 decision rule

- **Category**: Methodological / Scope
- **Confidence**: Medium
- **Currently assumed because**: A5's floor (denominator = max(example, 0.010)) fixed the trivially-passable-gate problem, and the flag was specified as a reporting attribute.
- **What changes if wrong**: The floor creates the mirror problem: on a genuinely low-signal fold (example model at, say, 0.005), the gate demands arm A ≥ 0.006 — *beating the domain yardstick* — so the fold can fail its gate for regime reasons, not baseline-construction reasons, yet it still counts toward "gate failed on ≥ 2 folds → baseline-construction negative". Conversely a flagged fold that scrapes past the floor contributes a vote to WINS/HURTS from a period where per-era corr is mostly noise. The criteria say the flag "carries" into reporting but never say how a flagged fold enters the decision rule or the gate-failure tally.
- **How to test**: Free and immediate: component #9 computes all three denominators before freeze — if none lands near 0.010, pre-register that fact and the assumption is retired for this run; if any does, pre-register the flagged fold's role (full vote / reported-but-excluded / triggers the fold-substitution question) *before* the number can motivate the choice.
- **Relevant evidence**: Example model 0.0235 on the parent's 255-era block is an average over regimes; the three ~110-era blocks cover the most recent (hardest, per the decomposition) data, so a low block is plausible, not hypothetical.

### 6. The EU-1 proxy gate ratio (VALID slice) predicts the TEST-era gate ratio

- **Category**: Methodological
- **Confidence**: Medium
- **Currently assumed because**: B1 wanted the gate bit one unit earlier, and a VALID-slice ratio is the only thing EU-1 can measure without touching TEST.
- **What changes if wrong**: The ratio model/example is itself regime-dependent (the parent's baseline hit 83% of its sanity band on older eras and 27% on the broken protocol's eras). A healthy proxy that false-negatives the warning costs the plan its early-warning value and EU-2 gets sized on optimism; a pessimistic proxy could trigger premature baseline-fix spend. Impact is bounded — the real gate still runs, fold-1-first — so this degrades efficiency, not validity.
- **How to test**: Report the proxy alongside the realized fold-1 ratio (both will exist) so the proxy's calibration is itself a measured quantity for folds 2–3 decisions; do not let the proxy trigger anything other than sizing caution.
- **Relevant evidence**: Parent's era-dependence of baseline quality; decomposition #4's own "hardest regime" note.

### 7. Hierarchical bootstrap over 3–5 seeds yields valid CIs

- **Category**: Methodological
- **Confidence**: Medium
- **Currently assumed because**: A6 fixed the seed-variance *omission*; the resampling scheme was specified without noting that seed-level resampling over n=3–5 units estimates the seed component on 2–4 degrees of freedom.
- **What changes if wrong**: Bootstrap over so few second-level units is known to under-cover — the seed component of the CI will be noisy and typically anti-conservative, i.e., the headline CI is somewhat too narrow exactly when the verdict is close. The negligibility check (seed var < ~10% of era var) is the effective mitigation; the residual risk is the case where seed variance is non-negligible *and* estimated on 2 dof.
- **How to test**: Already half-built: component #3's sim runs at 3 and 5 seeds — add a coverage check (does the nominal 95% hierarchical CI cover the known injected effect ~95% of the time?) to the sim's outputs; if coverage is materially below nominal at 3 seeds, that is a concrete, pre-freeze argument for the 5-seed packing option beyond "more power".
- **Relevant evidence**: Standard small-cluster bootstrap literature; the parent's bootstrap-RNG-fragile Sharpe CI shows this design is already operating near CI-stability limits.

### 8. A3 signature 2 is computable when it is needed

- **Category**: Methodological (operational)
- **Confidence**: Medium
- **Currently assumed because**: The signature ("stage-1 range < ~2× the across-seed standard deviation at fixed config") was written as a statistical condition without tracing where its inputs come from in the staged design.
- **What changes if wrong**: Stage 1 runs **one** trial per grid point — there is no across-seed sd at sweep time; the only multi-seed data arrives at refit (different data size, selected config only). A pre-registered guard whose inputs don't exist as written gets reinterpreted post hoc, which is precisely the failure mode pre-registration exists to prevent — and this guard gates the kill criterion (condition 4).
- **How to test**: Free: pre-register the sd source now (e.g., across-seed sd of the arm-A refit at the same fold, or one deliberately duplicated stage-1 trial at the H3-prior rank ~32 — costs one of the 12 trials but makes the guard real).
- **Relevant evidence**: Criteria sweep section vs. A4 stage-1 definition — the two subsections are internally inconsistent on this point.

### 9. Arm C's design, frozen from a synthetic-stream sim plus short-run EU-1 diagnostics, transfers to full-schedule real gradients

- **Category**: Scaling
- **Confidence**: Medium
- **Currently assumed because**: B3 requires the sim before porting code, and EU-1's short arm-B runs feed real rotation-rate diagnostics — the design freeze then happens before any full-schedule arm-B run exists.
- **What changes if wrong**: If the real basis-rotation rate or effective rank drifts over the full schedule (plausible: covariance EMA equilibrates slowly at decay 0.99–0.999 over ~10k steps), the frozen C matches early-run invariants and drifts off the A7 invariants late in training — degrading the fair-control evidence the symmetric honesty clause requires.
- **How to test**: The invariants are all logged in production arm-B runs (A8/B3); pre-register a *post hoc invariant-match report* (realized principal-angle and norm-ratio match between B and C over the full schedule, per fold) as the fair-control evidence the honesty clause consumes — this converts the assumption into a measured quantity without new compute.
- **Relevant evidence**: Decomposition #6's own note that decay-EMA interaction with longer horizons is untested.

### 10. A clean kill on one dataset, one architecture, one p kills "the line … for financial tabular regression"

- **Category**: Scope
- **Confidence**: Medium
- **Currently assumed because**: The kill wording was inherited from the follow-up brief and qualified along the power (A1), rank (A2), and budget (A3) axes — but not the dataset/architecture/scale axis, which the criteria elsewhere explicitly list as future work ("multiple datasets, … architecture variety").
- **What changes if wrong**: The verdict claims a generality the design cannot support: one Numerai-v5 MLP at p ≈ 600k. Low probability of misleading anyone technical, but it is the one remaining place the frozen criteria overclaim relative to their own compute-feasibility section.
- **How to test**: Free wording fix at freeze: "dead for this problem class as instantiated (Numerai v5 tabular regression, MLP, p ≈ 600k, k ≤ K_max)" — the A2 scoping sentence already does exactly this for rank; extend the same move to the other axes.
- **Relevant evidence**: Criteria's Publishability section (main-conference breadth explicitly out of scope).

## Background Assumptions (High confidence, Low impact)

### 11. The B4 power sim's inputs (parent per-era diff vectors from the B×B variant on the broken protocol) transfer to this run's variance structure

Self-correcting: the sim only calibrates expectations and write-up prominence; the verdict consumes the *realized* per-fold MDE. Worst case is a mis-calibrated prior on which qualified outcome is modal. Accept; note the provenance in the sim's output.

### 12. Cross-fold step-count semantics under expanding windows

The converged step count comes from one EU-1 convergence run, but TRAIN grows across folds (fold-3 ≫ fold-1). The refit rule scales within-fold (rows_refit/rows_train); whether the *trial* step count scales across folds is unstated. A one-line rule at freeze ("trial steps scale by rows_train_fold/rows_train_EU1", or explicitly fixed) closes it. Low impact — affects both arms symmetrically within a fold (though see Critical #1 for the asymmetric residue).

### 13. Stage-1 trial-count arithmetic

Stage 1 is stated as 7–8 trials over 6 enumerated points ({8, 32, 128, 512, 2048} + effrank). The 1–2 unexplained trials are presumably slack for capping, but an auditor checking "genuine sweep per the pre-registered allocation" needs the enumeration and the count to match. Trivial fix in `protocol.json`.

## Assumption Dependency Map

- **#1, #4, #8 → the "genuine sweep at matched budget" construct → kill condition 1 and the A3 guard.** These three are the same defect class the round-1 challenge found in the trial *count* — now in the trial *content*. If any fails, a HURTS/NULL passes all four kill conditions while being a budget/schedule artifact: A3's signatures are blind to under-training (#1), blind to mis-ranked unpicked ranks (#4), and possibly not computable as written (#8). The kill criterion's four-condition structure is only as strong as the signatures feeding condition 4.
- **#2 stands alone** but is two-sided: degeneracy toward B fabricates "spectral selection doesn't matter"; degeneracy toward noise fabricates "spectral selection matters" (the symmetric honesty clause catches the second only if the fair-control evidence of #9 exists). #9 is the evidence channel for #2's clause.
- **#3 compounds with #7**: an anti-conservative CI (#7) shrinks the reported half-width, making the A1 condition (MDE ≤ 0.005) *easier* to satisfy exactly when it shouldn't be — the two calibration gaps push the same direction, toward an overconfident kill.
- **#5 depends on realized denominators** (#9-the-component, testable pre-freeze); if all three denominators land comfortably above 0.010, it retires completely.
- **#6, #11, #12, #13** are leaves — efficiency and bookkeeping, no downstream verdict coupling.

## Recommendations

**Test/fix before protocol freeze (all zero-compute wording or logging changes; the freeze is the last legitimate moment):**
1. (#1) Add the fourth under-exploration signature — arm B's selected config still improving at step cutoff — to the A3 list, and/or calibrate claim language to "at matched step budget". Log VALID-vs-step for all arm-B trials (infrastructure exists).
2. (#2) Add the distinctness criterion (principal-angle floor and update-cosine ceiling vs B) to the required B3 sim, and pre-register the empty-middle fallback wording.
3. (#3) Connect the kill's power condition to B4's P(detect) quantity, or rename the reported half-width honestly and state the kill's bound in relative terms against realized arm A.
4. (#4) Scale stage-1 transferred LR by measured kept-norm fraction per rank (EU-1 measures it), or add the near-identity-winner signature.
5. (#8) Name signature 2's sd source in `protocol.json`.
6. (#5) After #9 computes denominators: pre-register the flagged-fold's role in the decision rule (or record that no fold is flagged).
7. (#10, #12, #13) One-line wording fixes at freeze: kill-scope instantiation axes; cross-fold trial-step rule; stage-1 enumeration = count.

**Suggest plan revision (small, contained):**
- (#7) Add the coverage check to component #3's existing sim; if 3-seed coverage is materially sub-nominal, that finding upgrades the 5-seed packing option from "power lever" to "validity lever" in #1's arithmetic.
- (#9) Pre-register the post hoc invariant-match report as the fair-control evidence consumed by the symmetric honesty clause.

**Acceptable risks (no change):**
- #6 (proxy-ratio calibration — heuristic only, real gate still runs fold-first), #11 (power-sim input provenance — self-correcting), and the previously-accepted round-1 residuals (fold correlation as claim calibration, torch-pin compatibility via EU-1 re-assert, novelty mini-search deferred to pre-write-up per C1).

None of the four critical items requires compute or a design rethink; all are wording, logging, or sim-extension changes that fit inside the protocol-freeze checklist the decomposition already gates EU-2 on. That is the same shape as the round-1 fixes — which is itself the observation worth carrying: this plan's failure modes concentrate not in its experiments but in the fine print of its safeguards, and the freeze checklist is where every one of them is cheapest to close.
