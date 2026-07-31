# Pre-Mortem Analysis

## Setting

It is six months from now. The research on "Spectral Optimizer (for noise reduction) on Financial Timeseries Data" has been completed. The central hypothesis did not hold — the study did not produce a clean, defensible verdict that the Spectral Optimizer helps (or hurts, or provably doesn't help) on low-SNR financial prediction. The scenarios below reconstruct how that happened. They deliberately avoid re-litigating the decomposition's component risks (#1 spectral engagement, #2 vmap, etc., already carry honest P values and fallbacks); each scenario is a systemic failure that passes the component gates individually and emerges from their interaction.

## Scenarios (ranked by priority)

### 1. The verdict thresholds were unreachable in the regime the budget forced [Likelihood: High | Severity: High]

**What happened**: To fit the <30-minute job cap and 32G RAM, the pipeline subsampled Numerai eras and features. The tuned-AdamW baseline landed at mean per-era corr 0.007 — inside the sanity band [0.005, 0.05], so component #8 passed its gate. The power analysis (#3) bootstrapped from that baseline's per-era corr vector and reported an achievable CI half-width of ~0.005 using all available validation eras, a marginal pass. The main seeded comparison then produced a filter-vs-no-filter difference of +0.001 with a paired CI of [-0.004, +0.006].

That interval satisfies none of the three verdict categories: it does not exclude zero (no "helps"/"hurts"), and it does not exclude improvements ≥0.005 (no equivalence-style "doesn't help"). The pre-authorized fallback — "an honestly-scoped effect estimate with calibrated uncertainty" — was executed, but the resulting paper read as "we ran a small experiment and could not tell," which is exactly the outcome the success criteria's teeth were designed to prevent. The deeper problem: with a baseline at 0.007 rather than the assumed 0.02, the 0.005 practical-significance threshold silently became a ~70% relative-effect requirement instead of the intended ~20-25%. No single gate caught this because each gate checked its own band: #8 checked "in [0.005, 0.05]", #3 checked "CI half-width ≤ ~0.005", and nobody checked the ratio between the two.

**Root cause**: The verdict thresholds were calibrated in absolute corr units against practitioner performance on the *full* dataset, while the budget forced a subsampled regime with lower absolute signal. The pass criteria for the baseline gate and the power gate were individually satisfiable in a configuration where the verdict definitions were jointly unsatisfiable.

**Early warning signs**:
- Baseline lands in the bottom third of the sanity band (mean per-era corr < 0.012) at gate #8 — visible in the very first array job, before any experiment slot is spent.
- The #3 bootstrap's detection probability for a +0.005 effect is below ~0.8, or the achievable half-width is within 20% of the threshold (a "marginal pass") — visible in the pre-experiment gate.

**Mitigation**:
- Now: make the practical-significance threshold *relative* to the realized baseline (e.g., ≥25% of the baseline's mean per-era corr, with 0.005 as a cap, not a floor), and add a joint gate: proceed to the main comparison only if the #3 bootstrap shows ≥80% probability of reaching *some* verdict category given the realized baseline level.
- If the warning appears: spend budget on more eras before more seeds (pre-authorized as nearly free); reduce feature subsampling before era subsampling, since features drive per-run cost less than they drive signal.
- If neither restores power, re-scope the primary endpoint *before* running the main comparison (e.g., corr-Sharpe difference, or a directional sign test across eras), rather than discovering the endpoint is unreachable afterward.

**Pivot indicator**: After gate #8 and #3, computed jointly: if P(any verdict reachable) < 0.6 under the realized baseline corr and era count, stop and redesign the endpoint — do not proceed to spend the 5 slots on a comparison that cannot conclude.

**Load-bearing assumption**: That a subsampled-Numerai baseline would land near the middle of the practitioner band (~0.02), keeping 0.005 a "20-25% relative" effect. The subsample plausibly lands much lower, and every verdict definition inherits the miscalibration.

---

### 2. The filter engaged — as a trivial mean-gradient smoother — and the null's mechanistic story was wrong [Likelihood: Medium | Severity: High]

**What happened**: The engagement diagnostics passed cleanly from the first run: 1-3 eigenvalues above the MP bulk, 30-60% of gradient norm passed. The main comparison returned a null, and the paper reported the designed interpretation: "the mechanism engaged but did not improve generalization in the low-SNR regime." A reviewer then pointed out what the diagnostics actually showed. For MSE regression with a shallow MLP on weak signal, the per-sample gradient of the output layer is (residual_i × activation_i); the B×B similarity matrix is dominated by one large eigendirection — essentially the batch-mean gradient — with everything else in the bulk. Keeping "directions many samples agree on" reduced, in this regime, to "keep approximately the mean gradient, discard per-sample deviations": functionally equivalent to gradient smoothing / a larger effective batch, an effect that tuned AdamW's learning-rate search absorbs entirely. The null was real, but it was a null about *variance reduction that LR tuning replicates*, not about coherence amplification distinguishing signal from noise directions.

This was hard to anticipate because the diagnostics were designed to rule out the opposite failure (a no-op filter), and "eigendirections kept strictly between 0 and B, norm fraction in [0.1, 0.9]" is nearly *guaranteed* to pass in a low-SNR regression: there is always a mean-gradient spike above the MP bulk. The pass criterion for component #1 could not distinguish "the mechanism found coherent signal structure" from "the spectrum has the trivial rank-1 structure every regression batch has." The MNIST label-noise success had many class-conditional coherent directions; financial regression may have exactly one, and it is the one plain SGD already follows.

**Root cause**: The engagement diagnostics are necessary but not sufficient. They measure that filtering *happened*, not that the retained subspace differs from what the unfiltered mean gradient already computes. A verdict's interpretation rested on a diagnostic that passes for trivial reasons in this loss/architecture regime.

**Early warning signs**:
- In the first diagnostics run (gate #1): the number of above-bulk eigenvalues is consistently 1-2 and the cosine similarity between the filtered update and the plain mean-gradient update is > ~0.95. That cosine is a one-line addition to the diagnostics and is the single most informative number in the study.
- The filter-on arm's optimal learning rate in the tuning sweep shifts systematically relative to filter-off (a signature that the filter is acting as an LR/batch-size rescaling).

**Mitigation**:
- Now: add "filtered-update vs mean-gradient cosine" and "filtered-update vs filter-off-update cosine" to the logged diagnostics, and add a cheap control arm to the ablation: plain AdamW at doubled effective batch (or gradient-EMA smoothing). If the spectral filter is indistinguishable from that control, the correct conclusion is an equivalence reduction, not a transfer verdict.
- If the warning appears: report the reduction *as the finding* — "in low-SNR regression the MP-spectral filter degenerates to mean-gradient smoothing; the coherence-amplification mechanism has at most rank-1 structure to work with on Numerai." That is a genuine mechanistic contribution and arguably more interesting than the null, but only if the diagnostics were logged to support it.

**Pivot indicator**: If the update-cosine to the plain mean gradient exceeds ~0.95 for all threshold modes across both tested batch sizes in the gate-#1 run, stop treating the study as "filter vs baseline" and redesign around characterizing the degeneracy (spectra vs the MNIST-label-noise spectra), reallocating the sequence-arm slot to that comparison.

**Load-bearing assumption**: That "eigendirections kept ∈ (0, B) and norm fraction ∈ [0.1, 0.9]" implies the mechanism is doing *selective, non-trivial* work — i.e., that engagement diagnostics validate the mechanistic interpretation of whatever verdict follows.

---

### 3. Overlapping Numerai targets broke era independence, and the CI machinery was invalid [Likelihood: Medium | Severity: Medium]

**What happened**: The entire statistical design — paired per-era differences, bootstrap CIs, the #3 power analysis — treated eras as independent observations. Numerai's main targets are forward returns over horizons (~20 trading days) that span multiple weekly eras, so adjacent eras have mechanically overlapping target windows and strongly autocorrelated per-era scores. The naive bootstrap over eras understated the CI width by a substantial factor. Two consequences surfaced late: first, the #3 power gate passed optimistically (the bootstrap resampled autocorrelated eras as if exchangeable); second, when the results audit recomputed the CI with a block bootstrap, the "helps"-direction interval that had excluded zero no longer did. The headline verdict retracted to no-verdict after all experiment slots were spent. A small embargo (1-2 eras) also let target-window overlap leak between train and validation, nudging the baseline corr upward and contributing to Scenario 1's miscalibration in the opposite direction.

This was foreseeable — the practitioner literature the protocol was drawn from discusses purging *because* of overlapping targets — but the plan operationalized the purge (a data-split concern) without carrying the same fact into the inference machinery (an analysis concern). The two live in different components (#9 builds the split; #3 does the power analysis) and the dependency between them was never stated.

**Root cause**: Era autocorrelation from overlapping target windows was handled in the data split but not in the statistical inference. The paired per-era design cancels era-level *level* variance but does not restore independence of era-level *differences*.

**Early warning signs**:
- Lag-1 autocorrelation of the baseline's per-era corr vector > ~0.2 (computable for free the moment gate #8's output exists, before #3 runs).
- Embargo shorter than the target horizon in eras (a checklist item at #9: target horizon in days ÷ era spacing in days = minimum embargo).

**Mitigation**:
- Now: fix the embargo to ≥ ceil(target_horizon / era_spacing) eras; specify in the analysis plan that all CIs use a moving-block bootstrap (block length ≥ target-horizon overlap) or a HAC-adjusted paired test; make the #3 power simulation use the same block structure.
- If the warning appears after #8: re-run the power gate with blocks before committing experiment slots — the era count needed may grow, which is still nearly free.

**Pivot indicator**: If the block-bootstrap CI half-width exceeds the verdict threshold even at the full available era count, this collapses into Scenario 1's pivot: redesign the endpoint before spending slots.

**Load-bearing assumption**: That per-era scores (and per-era paired differences) are exchangeable units for bootstrap inference. With ~20-day targets on weekly eras, they are not.

---

### 4. Matched trial counts, unmatched tuning adequacy — the negative verdict was an under-tuning artifact [Likelihood: Medium | Severity: Medium]

**What happened**: The protocol matched tuning *budgets* — same number of trials, same search-space scope per arm — per the DeepOBS standard. But the spectral arm's search space is genuinely larger (threshold mode × filter strength × batch size, which interacts with the B×B spectrum, on top of LR/wd), and its per-trial cost is 2-5x AdamW's. Under the 30-minute job cap, both pressures were resolved the same way: the spectral arm got the same 8-12 trials over a larger space at coarser resolution, with batch size frozen at whatever gate #7 had timed. The comparison returned "hurts" with a CI excluding zero. The verdict was formally clean and passed every internal check — matched trials, ≥3 seeds, engaged diagnostics — but it was substantially an artifact of the treated arm sitting further from its optimum than the baseline sat from its. The filter-on/filter-off ablation at identical base hyperparameters, which was supposed to be the clean isolation, inherited the same problem: "identical base config" means the base config was tuned for the *unfiltered* dynamics, which is a handicap for the filtered arm, not a control.

This is the mirror image of the strawman failure the plan guards against (under-tuned baseline flattering the method) and is harder to see because the guard-rail — matched budgets — is what produces it. The known direction of publication bias in optimizer papers made everyone vigilant about unfair advantage to the new method; unfair handicap produces an equally wrong negative verdict.

**Root cause**: "Matched trial count" is not "matched distance from each arm's optimum" when the arms have different-dimensional hyperparameter spaces and different per-trial costs under a shared wall-clock cap.

**Early warning signs**:
- The spectral arm's tuning-sweep results are non-monotone or high-variance across its threshold-mode/strength grid (the sweep is not resolving its space), while AdamW's LR curve is smooth — visible in the tuning experiment itself, before the seeded comparison.
- Best-found spectral configs sit on the boundary of the searched grid (classic under-exploration signature).

**Mitigation**:
- Now: pre-register the spectral arm's hyperparameter priors from the prior repo's experience (the code is the author's own — transfer its known-good threshold modes/strengths as the search center, and document that transfer as part of the matched-budget argument). Define the tuning match in *compute*, not trial count, and state this choice in the protocol.
- If the warning appears: spend the spare slot (or the designated-cut budget) on a second, refined spectral sweep centered on the best region before running the seeded comparison; report the sweep resolution honestly as a limitation if it cannot be afforded.

**Pivot indicator**: If a "hurts" verdict emerges and any of: best config on grid boundary, non-monotone sweep, or optimal-LR shift between arms (Scenario 2's warning) is present — downgrade the claim from "hurts" to "no evidence of benefit under the affordable tuning budget" before write-up, and say why.

**Load-bearing assumption**: That equal trial counts over each arm's declared search space deliver comparable tuning adequacy, so a negative delta reflects the mechanism rather than asymmetric optimization.

---

### 5. The cut cascade executed as designed and left an unpublishable core [Likelihood: Medium | Severity: Medium]

**What happened**: Nothing failed catastrophically; the pre-registered cut order simply fired in sequence. The sequence arm dropped at gate #2 (P was only 0.35 — this was the expected outcome, not a surprise). Budget arithmetic at gate #4 came in tight because the spectral arm's per-trial cost ran at the high end, so Muon was cut, then the GAF-style ablation — both designated cuts, executed correctly. The result was the minimum viable configuration: Numerai, one MLP, tuned AdamW vs filter-on, 3 seeds, null verdict. Each cut was individually pre-authorized; jointly, they removed every element that made a null *interesting*: no architecture-consistency statement, no evidence the spectral machinery differs from the cheap GAF heuristic, no Muon context, and (per Scenario 2) no control separating the filter from batch-size effects. TMLR reviewers did not dispute the rigor; they disputed the significance — a single-architecture, single-dataset null on the author's own unpublished optimizer, with the differentiating ablation (the one the novelty assessment called "the single strongest way to convert related-work-exists into a sharp contribution") cut for budget. The workshop fallback accepted it, but the deliverable landed well below what the ~0.30 "informative publishable outcome" estimate implied, because that estimate priced component failures and not the value lost through authorized cuts.

**Root cause**: The cut order was optimized to protect statistical validity (seeds, matched tuning) but not to protect the *contribution's identity*. For a likely-null outcome, the GAF ablation is not an optional garnish — it is what distinguishes "the spectral machinery specifically" from "any consensus filtering," which is the paper's stated novelty axis.

**Early warning signs**:
- Gate #4 arithmetic shows the plan fits only with both Muon and GAF cut — known within the first day of the pre-experiment gate, before any slot is spent.
- Gate #2 fails (expected at P=0.35), freeing 1-2 slots — the moment at which the freed budget's allocation decision determines the paper's ceiling.

**Mitigation**:
- Now: re-order the cut list conditional on the expected verdict direction: if early diagnostics point to a null, the GAF ablation outranks the sequence arm (it converts a null into a comparative mechanistic statement; a second architecture merely replicates the null). Write this conditionality into the plan rather than the static order.
- If the warning appears: the GAF filter is cheap (pairwise checks, no eigendecomposition, less compute than the spectral arm) — it can often share an array job with the main comparison rather than costing a full slot; check this in the #4 arithmetic explicitly.

**Pivot indicator**: If at gate #4 the plan cannot fit the main comparison plus at least one mechanism-discriminating ablation (GAF-style or the mean-gradient-smoothing control from Scenario 2), stop and re-scope the tuning sweep (fewer trials, declared) rather than cutting the ablation — a slightly less-tuned comparison with a discriminating ablation outranks a well-tuned comparison that cannot say what its null means.

**Load-bearing assumption**: That the minimum viable configuration (Numerai + MLP + 2 arms) clears the workshop bar *with a null result*. It clears it with a positive result; a null with no discriminating ablation is near the floor of publishable.

## Cross-Cutting Themes

1. **Budget-driven subsampling silently changes the scientific regime.** Scenarios 1, 4, and 5 all descend from the same pressure: the <30-min/5-slot budget forces subsampling and cuts, and the absolute-units verdict thresholds, the tuning adequacy, and the ablation set were all calibrated for a regime the budget doesn't deliver. The gates check feasibility in the subsampled regime but not whether the subsampled regime still tests the hypothesis.
2. **Diagnostics validate execution, not interpretation.** Scenarios 2 and 3 share a shape: a check designed to rule out one failure (no-op filter; leaky split) passes, and its passing is then treated as licensing a stronger claim (mechanism meaningfully engaged; eras are independent units) that it never tested.
3. **The negative direction is under-guarded.** The plan's rigor machinery (matched budgets, sanity bands, strawman avoidance) is oriented against *false positives*. Scenarios 2, 3, and 4 each produce a false or uninterpretable *negative* — the direction this project, with its "negative results are full deliverables" framing, will publish without hesitation.

## Summary Risk Profile

| Scenario | Likelihood | Severity | Priority | Mitigable? |
|----------|-----------|----------|----------|------------|
| 1. Verdict thresholds unreachable in the forced regime | High | High | Critical | Yes (relative thresholds + joint gate) |
| 2. Trivial engagement; null's mechanistic story wrong | Medium | High | High | Yes (cosine diagnostics + smoothing control) |
| 3. Era overlap invalidates CI machinery | Medium | Medium | Medium | Yes (embargo sizing + block bootstrap) |
| 4. Unmatched tuning adequacy → artifact "hurts" | Medium | Medium | Medium | Partially (priors transfer; compute-matched budgets) |
| 5. Cut cascade leaves unpublishable null core | Medium | Medium | Medium | Yes (conditional cut order; cheap GAF co-scheduling) |

## Top Recommendations

1. **Add a joint go/no-go gate after #8 and #3**: convert the 0.005 practical-significance threshold to a relative one (≥~25% of the realized baseline corr, capped at 0.005), size the embargo to the target horizon, run the power simulation with a block bootstrap, and require P(some verdict reachable) ≥ 0.6 before spending any experiment slot. This addresses Scenarios 1 and 3 for zero GPU cost.
2. **Extend the gate-#1 diagnostics with two cosines** (filtered update vs mean gradient; filtered vs unfiltered update) and add a mean-gradient-smoothing / larger-batch control arm to the ablation. This is the difference between a null that means something and a null with a wrong mechanistic caption (Scenario 2).
3. **Make the cut order conditional on verdict direction and cost-shape**: if diagnostics point toward a null, protect the GAF-style (or smoothing-control) ablation ahead of the sequence arm, and check whether it can share an array job before treating it as a full slot (Scenarios 4, 5).

## Residual Risk

After all mitigations, the dominant residual risk is irreducible: the low-SNR regime may simply not contain enough separable gradient-coherent structure, at affordable batch sizes and era counts, for *any* affordable design to resolve a three-way verdict at practical significance. The mitigations convert most bad outcomes into honest, mechanistically-annotated findings, but they cannot guarantee those findings clear more than the workshop bar — the decomposition's own ~0.07 estimate for the full clean verdict is consistent with this pre-mortem, and roughly a 1-in-4 to 1-in-3 chance remains that six months from now the deliverable is a carefully-caveated "could not determine" at a venue below the primary target. That risk is acceptable only because both the plan and the success criteria explicitly price a well-characterized boundary/degeneracy finding as a deliverable; it should be accepted knowingly, not discovered at write-up.
