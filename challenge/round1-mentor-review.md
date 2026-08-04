# Mentor Review

## Overall Assessment

This is a well-designed redo: it fixes exactly the two defects that invalidated the parent run (wrong object, broken split), pre-registers the protocol and decision rule, buys the dominant uncertainty (the baseline gate) first and cheapest, and honestly prices P_success at ~0.09/0.23 rather than flattering itself. Its biggest strength is that the verdict machinery is symmetric and the kill criterion has teeth. Its biggest weakness is that the "genuine rank sweep at matched budget" — the run's raison d'être — is under-specified: 12 trials cannot resolve the ~42-point discrete config space × learning rate the criteria enumerate, and the plan carries no guard (the parent's own F12 amendment) against a HURTS verdict that is an under-exploration artifact, which the kill criterion would then convert into a terminal "line is dead" claim.

## What a Senior Researcher Would Do Differently

1. **Pre-register the allocation of arm B's 12 trials over its search space, and prune the space to fit.** The criteria list fixed rank {8, 32, 128, 512, 2048} + adaptive {effrank, gap} + energy_threshold {0.90, 0.99} + alpha {0.5, 1, 2} + LR — 42+ discrete combinations before LR is touched. Twelve uniform trials over that space is not a rank sweep; it is a lottery. Fix with a staged design stated in the experiment plan before TEST is touched: e.g., stage 1 = 7–8 trials, one per rank/adaptive setting at a transferred LR (prior H3/H5 justify centering: adaptive effrank was the only parity-grokking variant; rank ≤ 4 destabilizes); stage 2 = 4–5 trials refining LR (and alpha) around the stage-1 winner. Drop `gap` or one energy_threshold if needed — a resolvable sweep over a smaller space beats an unresolvable sweep over the brief's full enumeration, and the brief's own language ("capped at what is feasible") licenses this if documented.

2. **Carry forward the parent's F12 under-exploration guard.** The parent's triage bound this and the regenerated criteria dropped it: if a HURTS verdict emerges *and* arm B's best config sits on the grid boundary, or the sweep is non-monotone/high-variance across rank, or arm B's optimal LR shifts sharply vs arm A — pre-register the downgrade to "no evidence of benefit under the affordable tuning budget" rather than HURTS. This matters more now than in the parent because HURTS here triggers the kill criterion, a terminal claim. Zero compute; it is a paragraph in the protocol.

3. **State how the 3 paired seeds enter the bootstrap CI.** Bootstrapping era blocks of seed-averaged paired differences understates uncertainty if seed variance is non-negligible relative to era variance; the CI either needs seed variance folded in (hierarchical or seed-block resampling) or a reported check that seed variance is small. The parent machinery is being reused across a design change (255 → ~110 eras, 3 seeds) and this choice is currently implicit.

4. **Pre-register the gate-fix retry semantics.** The brief mandates re-checking a failed gate on the same TEST eras after baseline fixes. That is a second TEST touch for arm A, and a baseline partially selected on its own TEST ratio biases (B−A) in the HURTS direction (conservative for WINS, anti-conservative for HURTS — the direction the kill criterion fires on). State now: gate retries select on the gate ratio only; comparison hyperparameters are re-selected on VALID after any baseline fix; TEST-touch count logged per fold and reported.

5. **Report the per-fold minimum detectable effect alongside any NULL.** Component #2's power sim expects ~0.0033 half-width at 110 eras — adequate for a 0.005 effect, marginal below it. "The line is dead" on a NULL is only an honest claim if accompanied by "and we could have detected an effect of size X." Cheap: the MDE falls out of the bootstrap already planned.

## What Hasn't Been Examined Yet

- **Filter EMA horizon vs non-stationarity.** `decay=0.99` (≈100-step covariance memory) and `warmup=100` are fixed at repo defaults across ~10k-step schedules on regime-shifting data, and neither is in the sweep space. If the verdict is NULL/HURTS, "covariance memory untested" is the first limitation a reviewer will name. Either swap one grid dimension for decay ∈ {0.99, 0.999} or pre-declare it a limitation now, not at write-up.
- **Example-preds coverage of TEST eras.** The gate divides by the example model's per-era corr from `v5.0_validation_example_preds.parquet`; component #8 should assert every TEST era of every fold exists in that file (the expanding-window TRAIN reaches into the train parquet, and fold boundaries are computed from the usable-era list — an off-by-one here silently breaks the gate denominator).
- **The ==5 assertion under era-index gaps** is handled in component #8's fail branch, but the resolution ("adjust the boundary definition, not the assertion") must land in `protocol.json` with the raw-era ↔ usable-index mapping recorded, since the brief treats the literal assertion as the run's validity certificate.
- **Step-count tuning location.** The brief mandates re-tuned step counts; the plan never says whether steps/epochs sit inside the 12-trial space (consuming trials) or are fixed by a pre-flight convergence check (the better answer). Decide and state it — it interacts directly with item 1 above.

## Simpler Alternatives

The scoping is already close to minimal for a credible answer: one dataset, one architecture, three arms, three folds, gate-first ordering, no GRU rebuild, banked infrastructure. I see no cheaper path that still clears the tuned-baseline/matched-budget bar. The one simplification I would actively make is the search-space pruning in recommendation 1 — it *reduces* complexity and *increases* the sweep's information content simultaneously. Do not add Muon; the criteria are right to defer it.

## Construct Validity / Information Value

The construct is sound. (1) The result is not statable on paper: the parent's target-independence theorem is correctly scoped to the row-normalized B×B variant and does not bind `SpectralGradientFilter`; the literature genuinely predicts both directions (Coherent Gradients/GAF/H5 for, Feldman long-tail and Adam's built-in noise robustness against). (2) The object is the real thing under study — the actual p×p filter at its verified path, integrated via its documented API, with an alpha=0 identity check guarding against testing a broken wrapper — not a proxy. (3) The plan answers the motivating question with a committed WINS/HURTS/NULL verdict under a pre-registered rule, and the arm-C control plus the mechanism-honesty clause prevent the one residual construct trap (claiming "spectral" when "any low-rank projection" explains it). The ==5 assertion and the 0.60× gate are precisely the validity instruments the parent lacked. No construct redesign is called for. The residual validity risks are inferential, not constructive: an artifact HURTS from an unresolvable arm-B sweep, and an underpowered NULL feeding the kill criterion — both fixable by pre-registration, per the recommendations above.

## Key Recommendations

1. **Pre-register arm B's 12-trial allocation as a staged design over a pruned space** (one trial per rank/adaptive point at transferred LR, then LR/alpha refinement), so "genuine rank sweep" is true by construction rather than by hope — and document any space pruning under the brief's feasibility clause.
2. **Reinstate the F12 under-exploration guard as a binding amendment**: grid-boundary / non-monotone-sweep / LR-shift signatures pre-registered, with the HURTS→"no evidence of benefit under affordable budget" downgrade rule stated before any TEST touch — this protects the kill criterion from firing on an artifact.
3. **Close the three inference gaps in the protocol before freezing it**: seed-variance handling in the bootstrap CI, gate-retry semantics (VALID-only re-selection, TEST-touch logging), and per-fold MDE reporting attached to any NULL.

## Verdict

MAJOR_REVISIONS

The construct passes every information-value check and the design directly answers the motivating question — no rethink is warranted, and most of the plan is ready to run. But one load-bearing element must change before compute is spent: as written, the matched 12-trial budget cannot resolve the enumerated arm-B space, so the run risks failing its own criteria clause ("rank sweep genuinely executed") or, worse, delivering an artifact HURTS that the kill criterion converts into a terminal claim with no pre-registered downgrade path. All required fixes are zero-compute protocol amendments (trial allocation, F12 guard, three inference clarifications) and should be folded into the pre-registered protocol before EU-2, mirroring how the parent run's triage bound its amendments.

Relevant files: success-criteria.md (missing F12-style downgrade rule and trial-allocation spec), decomposition.md (components #2, #5, #8 carry the affected quick tests), prior/challenge/limitation-triage.md (F12, the amendment to reinstate).
