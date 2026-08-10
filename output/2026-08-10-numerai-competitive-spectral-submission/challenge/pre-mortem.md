# Pre-mortem: Competitive Spectral Submission

Assume the study has finished and failed: the optimizer comparison is not credible, the offline winner does not transfer, and the resulting artifact is not competitive. The plan has strong provenance, sealing, metric-parity, and claim-boundary protections. The main remaining risks are experimental-design risks rather than runtime risks. Spectral runtime overhead is irrelevant to fairness; the two arms must instead match completed configurations, training examples and optimizer updates per configuration, folds, and seeds.

## Ranked failure scenarios

### 1. The comparison budget was matched by accelerator time rather than statistical opportunity

- **Likelihood:** High
- **Severity:** High
- **Combined risk:** Critical
- **What failed:** AdamW and spectral were called “equal-budget” even though the spectral arm completed fewer configurations or received different numbers of examples, updates, folds, or seeds because each spectral update was slower.
- **Root cause:** Component 9 requires completed-trial count *and accelerator time* to agree within 10%, while the compute plan allocates about 60 accelerator-hours to AdamW and 120 to spectral and says either configurations or accelerator-hours may bind. Wall-clock/GPU time is not a valid fairness unit here. It can truncate the slower arm, alter promotion depth, or create unequal selection evidence despite nominally careful accounting.
- **Early warning signs:** Different counts of fully completed configurations; configurations stopped at different update counts; fold or seed results missing disproportionately in one arm; “equal compute” invoked to justify fewer spectral evaluations; results changing when compared at a common update checkpoint.
- **Mitigation:** Delete accelerator-time and per-step-overhead conditions from components 3, 8, and 9 as scientific stop or fairness rules. Before either HPO begins, freeze an arm-level ledger with the same number of completed configurations, the same training examples and optimizer updates at every fidelity, and the same folds and seeds. Run all promoted configurations to those endpoints. Track runtime only for logistics and packaging, never to determine comparative evidence.

### 2. Asymmetric screening killed spectral before it received the same selection opportunity as AdamW

- **Likelihood:** High
- **Severity:** High
- **Combined risk:** Critical
- **What failed:** Spectral was rejected by a three-setting pilot while AdamW was allowed a broad 30–40-configuration multi-fidelity search, so the negative result measured unequal search procedures rather than optimizer quality.
- **Root cause:** Component 8 imposes an early spectral-only gate (`2 of 3` positive settings, best delta at least `0.0005`, and random-subspace separation), whereas AdamW proceeds through a much larger HPO after only a basic learning sanity check. The pilot is noisy at era-level effect sizes and constitutes asymmetric optional stopping. Freezing architecture and non-optimizer choices from AdamW is sensible for attribution, but it does not justify giving spectral fewer completed optimizer configurations.
- **Early warning signs:** The conclusion says “spectral failed” after only three settings; one setting is mildly positive but misses an arbitrary pilot threshold; spectral gets fewer full-fidelity fold-seed evaluations; pilot outcomes are used to avoid the predeclared matched search.
- **Mitigation:** Make the pilot an integrity test only (finite updates, correct controls), not an efficacy gate. Predeclare matched optimizer searches with equal completed configuration counts and identical fidelity schedules, examples, updates, folds, and seeds. If resource availability limits the study, reduce both arms symmetrically before seeing outcomes. Reserve efficacy decisions for the matched search and sealed comparison.

### 3. Multiple targets and post-processing routes produced a winner by hidden multiplicity

- **Likelihood:** Medium
- **Severity:** High
- **Combined risk:** High
- **What failed:** A favorable result was selected from `target_cyrusd_20`, `target_ender_20`, a target ensemble, the separate Ender-60 track, raw predictions, several neutralization levels, and benchmark blends, but its confidence interval and threshold treated that route as if it had been the only test.
- **Root cause:** The plan says routes will be predeclared but does not designate one primary optimizer estimand and one exact primary target before HPO, nor specify multiplicity control across target, post-processing, and blend candidates. Using the same OOF folds to choose optimizer settings and transformations further amplifies winner's curse.
- **Early warning signs:** The “primary” target changes after development scores appear; the target ensemble or blend weights lack a frozen formula; many route-level intervals are reported but only the best route drives the conclusion; spectral wins on one target while losing on the others; bootstrap inference ignores route selection.
- **Mitigation:** Name one primary optimizer comparison now—for example, standalone predictions on one exact 20D released target—with one aggregation rule and one directional hypothesis. Treat every other target and all neutralized/blended routes as secondary. Freeze the complete candidate count and formulas, apply family-wise or false-discovery control to confirmatory secondary claims, and report the full route matrix regardless of outcome. Use benchmark blends only for the separate submission-worthiness decision, not to establish that the optimizer improved.

### 4. Repeated walk-forward HPO overfit the 574 development eras despite the sealed validation set

- **Likelihood:** Medium
- **Severity:** High
- **Combined risk:** High
- **What failed:** The selected AdamW and spectral configurations exploited idiosyncrasies of only three development folds, then both effects disappeared on official validation.
- **Root cause:** Thirty to forty configurations per arm, architecture choices for AdamW, fidelity promotion, spectral parameters, stopping rules, and post-processing are all selected from a small number of highly dependent era sequences. A moving-block bootstrap quantifies dependence for a fixed comparison; it does not undo adaptive selection over many configurations. Requiring low/high-fidelity rank correlation also does not establish out-of-regime robustness.
- **Early warning signs:** Large gaps between best and median configuration; unstable winners under removal of one era block or fold; broad seed variance; selected settings sit at search-space boundaries; low-fidelity rank correlation is acceptable overall but finalists reverse across folds; OOF gains concentrate in one regime.
- **Mitigation:** Add nested temporal selection inside the 574 eras: inner folds tune each arm, while outer development folds estimate the performance of the entire selection procedure. Select a final configuration by a predeclared stability rule, not maximum pooled OOF score. Report leave-one-block/fold sensitivity and selection-adjusted outer-fold deltas. Keep the official validation reveal for the single final confirmation.

### 5. The historical winner learned the wrong target for the live tournament

- **Likelihood:** High
- **Severity:** Medium
- **Combined risk:** High
- **What failed:** A model passed the Diagnostics-compatible and benchmark-relative gates but produced weak live CORR20v2/MMC because released proxy targets did not preserve optimizer rankings or useful signal for unreleased `target_cyrus_20`.
- **Root cause:** Neither `target_cyrusd_20` nor `target_ender_20` is the live payout target. The plan correctly states this limitation, but “submission-worthy” thresholds such as 95% of Ender or BMC `>=0.002` may still create unwarranted confidence. Optimizer rankings can reverse under target shift, and blending against Ender can reward historical benchmark orthogonality that does not transfer forward.
- **Early warning signs:** Spectral effects disagree in sign across released 20D targets; gains are concentrated in old validation eras; the chosen route depends heavily on benchmark residualization; forward predictions drift in distribution or benchmark correlation; early resolved rounds do not match the offline direction.
- **Mitigation:** Require sign and practical consistency across the predeclared released 20D targets as a robustness condition for promotion, while retaining one primary target for inference. Rename the offline gate “eligible for prospective test,” not “submission-worthy” in a competitive sense. If upload is later authorized, run frozen unstaked AdamW and spectral submissions in parallel for enough resolved rounds to compare paired live outcomes; do not update the optimizer conclusion from a few rounds.

## Priority changes to the plan

1. Replace every accelerator-time, steps/hour, and spectral-overhead fairness criterion with exact matching on completed configurations, examples, optimizer updates, folds, and seeds. Keep only memory feasibility and the separate Numerai upload-runtime contract.
2. Remove the spectral efficacy pilot as a stop rule. Give both arms the same predeclared multi-fidelity schedule and completed search opportunity.
3. Declare one exact primary target and standalone optimizer estimand; enumerate secondary target and post-processing tests and add multiplicity handling.
4. Evaluate the full HPO-and-selection procedure with nested outer development folds, including fold/block sensitivity, before the sealed reveal.
5. Recast offline promotion as eligibility for a prospective live test and require robustness across released 20D proxies; reserve competitive claims for paired resolved forward rounds.
