# Assumption Analysis

## Bottom line

The plan is unusually careful about temporal leakage, target naming, benchmark alignment, and the distinction between historical Diagnostics and live reputation. Its central comparison is nevertheless not yet valid as written. The decomposition repeatedly treats accelerator time and spectral runtime overhead as fairness constraints, despite the governing constraint that runtime overhead is irrelevant. The fair estimand is performance after matched **completed configurations, training examples or optimizer updates, folds, and seeds**. Matching accelerator-hours would give the slower spectral method fewer optimization opportunities and confound optimizer quality with implementation cost.

Three other assumptions could also overturn the conclusion: the AdamW-first procedure freezes shared choices in AdamW's favour; the pilot and HPO reuse the same development folds too adaptively to support the stated bootstrap claims; and the proposed blend/BMC gate can select a benchmark complement without establishing that the neural model is competitive on the unreleased live target. These are repairable before execution.

## Assumptions that require revision

### 1. Equal wall-clock or accelerator time is a valid fairness criterion

**Verdict: rejected.** Success criterion 3 says “equal trial/compute accounting”; decomposition component 9 requires accelerator-time within 10%; component 8 permits the “same accelerator-hour allowance”; component 3 imposes a 3x overhead and 500-steps/hour gate; and the compute plan gives AdamW about 60 accelerator-hours versus 120 for spectral before saying the comparison budget is matched. These rules are internally inconsistent and violate the explicit experimental constraint.

Runtime may be measured as an engineering characteristic, but it must not determine scientific promotion, stopping, search allocation, architecture, rank, update cadence, or the number of completed trials. The primary ledger should match, per arm:

- the same number of completed hyperparameter configurations at each fidelity;
- the same training examples seen, or equivalently the same predeclared optimizer-update schedule when batch sizes match;
- the same fold assignments and evaluation eras;
- the same number of seeds, paired by initialization and batch stream where possible; and
- the same treatment of failed configurations, with replacement rules declared in advance.

Spectral-only controls are mechanism-validation experiments and should be reported separately from the matched two-arm selection budget. More elapsed time for spectral is acceptable. OOM or numerical failure remains a feasibility failure; slowness alone does not.

### 2. Freezing architecture and training choices after AdamW HPO yields a neutral optimizer comparison

**Verdict: doubtful.** The plan searches learning rate, weight decay, clipping, schedule, warm-up, stopping, width/depth, dropout, normalization, and batch construction for AdamW, then freezes all non-spectral choices and tunes only filter parameters. That answers “does the filter improve the AdamW-optimal recipe?” It does not answer the stated procedure-level question of whether spectral optimization beats AdamW under the same search budget. Spectral filtering can change suitable learning rates, clipping, regularization, warm-up, capacity, and stopping horizons.

Choose and predeclare one estimand:

1. **Wrapper effect:** pair spectral with every evaluated AdamW configuration and add a fixed, small spectral parameter grid. Conclusions are restricted to adding the filter to AdamW-tuned pipelines.
2. **Best complete procedure:** give both arms the same architecture/training search space, completed-configuration count, examples/updates, folds, and seeds; spectral additionally searches its filter parameters inside that fixed count.

The second matches the current objective. A useful compromise is a shared set of sampled base configurations evaluated in both arms, with spectral parameters assigned by a predeclared nested design. Do not let AdamW consume architecture search outside the accounting while spectral inherits only one winner.

### 3. Era bootstrap intervals remain confirmatory after repeated fold-level selection

**Verdict: unsupported as written.** The same 574 training eras support sanity checks, multi-fidelity ranking, spectral pilot promotion, full HPO, target-track choice, neutralization, blending, and multiple threshold gates. A moving-block bootstrap over the selected model's era differences captures temporal sampling variation conditional on selection; it does not account for winner's curse from searching many configurations and transformations on those eras. An 80% lower bound above zero on development folds is therefore not evidence that selection is “robust.”

Use nested temporal evaluation: each outer development fold must select hyperparameters using only earlier inner folds, then generate one untouched outer-fold prediction. Pool only these outer-fold paired differences for promotion. Lock target ensemble, neutralization grid, blend family, block length, and tie rules before outer predictions. Treat the final official validation as the sole confirmatory test and the development intervals as descriptive or promotion-only, with no inferential wording.

### 4. “Official validation” is a homogeneous, untouched proxy for current deployment

**Verdict: only partly true.** It is untouched by this project if the seal works, but it spans hundreds of historical eras and likely multiple market regimes. A single mean CORR and block-bootstrap interval can be dominated by old regimes that are less relevant to a current live submission. Conversely, selecting a recent window after reveal would invalidate the seal.

Predeclare regime and recency diagnostics before reveal: fixed chronological subperiods, rolling 52-era effects, sign consistency, concentration of cumulative advantage, and a fixed recent-era window. Keep the full-period paired mean as primary, but require the claimed submission route not to depend on one old subperiod. This is a robustness diagnostic, not permission to select the best window.

### 5. Positive historical BMC or a benchmark blend establishes plausible live competitiveness

**Verdict: too strong.** The plan correctly notes that `target_cyrus_20` is unavailable historically, yet its “submission-worthy” gate allows BMC against Ender predictions on released targets to carry the decision. BMC is conditional on the chosen benchmark and historical scoring target. Positive BMC on `target_cyrusd_20`, `target_ender_20`, or `target_ender_60` may reflect residual structure that does not transfer to live Cyrus-20, and blend improvement can be mechanically easy when a weak but imperfectly correlated predictor is given a tiny fitted weight.

Define “offline submission candidate” rather than “submission-worthy” for this gate. Require a predeclared minimum blend weight or economically meaningful effect size, consistent sign across outer folds/seeds and fixed target tracks, and comparison with simple diversity controls such as a rank-matched random subspace and an equally correlated noise/control predictor. Reserve “competitive” for resolved prospective rounds.

## Ranked pre-mortem

| Rank | Failure scenario | Likelihood | Severity | Combined risk |
|---:|---|---|---|---|
| 1 | Spectral receives fewer completed trials or updates because it is slower, and an AdamW win is misreported as an optimizer-quality result | High | High | Critical |
| 2 | AdamW-optimal shared hyperparameters suppress spectral performance, so the experiment rejects the filter under an architecture/training regime selected for its control | High | High | Critical |
| 3 | Repeated adaptive selection on the 574 eras produces optimistic development intervals and a winner that fails on the sealed validation | High | High | Critical |
| 4 | A historically positive BMC/blend result does not transfer to the unreleased Cyrus-20 live target, yielding a submission candidate with poor forward reputation | Medium | High | High |
| 5 | Aggregate validation improvement is concentrated in obsolete eras or one seed, while the headline mean and bootstrap interval conceal deployment instability | Medium | High | High |

### Scenario 1: Runtime-matched comparison undertrains spectral

**What failed:** Spectral completed fewer configurations or saw fewer examples before the accelerator-hour cap, then lost to a better-searched AdamW arm.

**Root cause:** The plan conflated computational efficiency with statistical fairness and used both trial count and accelerator time as binding criteria.

**Early warnings:** Unequal completed-configuration counts; spectral jobs stopped mid-fidelity; unequal update counts; architecture/rank reductions made solely to satisfy steps/hour; trial replacement decisions differing by arm.

**Mitigation:** Delete time-based scientific gates. Match completed configurations, updates/examples, folds, and seeds. Continue spectral runs for as long as needed on free compute. Report runtime only in a separate engineering table.

### Scenario 2: The AdamW-first freeze bakes in control-specific choices

**What failed:** The filter appears neutral or harmful because clipping, learning rate, dropout, capacity, and training horizon were optimized for unfiltered AdamW gradients.

**Root cause:** The spectral arm gets filter HPO but no symmetric opportunity to choose interacting shared hyperparameters.

**Early warnings:** Spectral winners lie at the edge of strength, warm-up, or rank ranges; training curves are still improving at the frozen stop; spectral is unusually sensitive to clipping or learning rate in pilot runs; paired deltas reverse for non-winning AdamW base configurations.

**Mitigation:** Evaluate a paired shared design or run equal-budget complete-procedure searches. Report wrapper-effect and best-procedure estimands separately if both are retained.

### Scenario 3: Development selection overfits eras rather than rows

**What failed:** Strong pilot and HPO deltas disappear on official validation despite apparently positive block-bootstrap intervals.

**Root cause:** Millions of rows create a false sense of sample size, while many adaptive decisions are selected using only 574 temporally dependent era units.

**Early warnings:** Large rank changes across folds; winners change under modest block-length choices; one fold supplies most of the gain; low/high-fidelity Spearman instability; selected blends sit on grid boundaries.

**Mitigation:** Use nested outer walk-forward predictions for all selection-performance estimates, predeclare all adaptive families, and show configuration-selection stability. Do not interpret a conditional bootstrap as correcting HPO multiplicity.

### Scenario 4: Historical benchmark complementarity fails live

**What failed:** The frozen blend improves Ender-relative historical metrics but has no MMC/BMC advantage on forward Cyrus-20 rounds.

**Root cause:** Historical target proxies and benchmark residuals are not the live payout target, and the plan's submission gate places too much weight on proxy BMC.

**Early warnings:** Gains vary in sign among `target_cyrusd_20`, `target_ender_20`, and the target ensemble; positive BMC requires a very small blend weight; the model's prediction variance or correlation structure shifts sharply in recent eras.

**Mitigation:** Weaken the offline claim, require cross-target and recent-period consistency, freeze multiple unstaked forward comparators, and decide competitiveness only after enough resolved rounds.

### Scenario 5: A mean improvement masks regime and seed fragility

**What failed:** The selected optimizer has a positive full-period mean but worse current-regime performance, deeper concentrated drawdowns, or gains attributable to one initialization.

**Root cause:** Mean CORR plus an era bootstrap is treated as sufficient despite optimizer-by-seed interactions and nonstationarity.

**Early warnings:** Median seed delta is non-positive; leave-one-seed-out conclusions reverse; cumulative delta comes from a short historical interval; rolling 52-era effects are predominantly negative late in validation.

**Mitigation:** Make seed the replication level as well as era the temporal unit. Require positive median seed effect, disclose all seed-specific deltas, run leave-one-seed-out and fixed subperiod analyses, and characterize heterogeneous results as a tie.

## Required plan changes before execution

1. Replace every accelerator-time, wall-clock, steps/hour, and 3x-overhead fairness or stopping rule with matched completed configurations, examples/updates, folds, and seeds. Keep only memory, numerical correctness, and eventual completion as feasibility gates.
2. Rewrite component 9 so fairness requires exact or predeclared near-exact equality in completed configurations at each fidelity, update/example counts, folds, and seeds. Failed trials must use the same classification and replacement policy in both arms.
3. State whether the primary estimand is wrapper effect or best complete procedure. For the latter, give both optimizers the same shared hyperparameter search envelope and configuration count.
4. Add nested outer walk-forward evaluation inside the 574 training eras. Use outer-fold predictions—not HPO-fold winner scores—for pilot promotion, blend selection, and development uncertainty summaries.
5. Freeze target ensemble weights, neutralization levels, blend grid, bootstrap block rule, seed aggregation, recency window, and regime diagnostics before generating outer-fold or official-validation predictions.
6. Rename the offline gate to “offline submission candidate.” Require effect-size and cross-seed/cross-period consistency; reserve competitive or leaderboard-quality language for resolved forward submissions.
7. Separate mechanism controls from head-to-head HPO accounting. Strength-zero must verify implementation equivalence; random and shuffled subspaces test mechanism; neither should reduce the number of completed spectral configurations in the primary comparison.
8. Remove the hard stop that abandons the causal spectral comparison merely because tuned AdamW fails to beat the old recipe by exactly 0.001. A defensibly tuned AdamW can still be the valid control even if the old point estimate is hard to exceed; baseline competitiveness and optimizer comparison are distinct questions.

With these changes, the plan would support a defensible causal comparison on historical v5.3 data. It still could not establish live competitiveness without prospective resolved rounds, a limitation the existing plan otherwise handles correctly.
