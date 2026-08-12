# Multi-fidelity protocol and pre-validation amendment

The base protocol below was frozen before completion of the first screen. The primary selection statistic is mean exact
standalone Numerai CORR. BMC, Sharpe, runtime and official validation do not select configs.
The primary official benchmark is `v53_lgbm_ender60`, matched to the generic target because the
pinned `target` is byte-identical to `target_ender_60`. Ender20 is retained only for explicitly
labelled secondary 20-day-target analyses.
Every train-to-validation boundary purges exactly 16 eras, matching the 60-day main-target
horizon. The final models used for sealed historical validation train on eras 0001--0558 and
purge eras 0559--0574 before validation begins. They are not silently reused as the eventual
production-live refit; after model selection and sealed evaluation, production models may be
refit without further hyperparameter selection on all then-resolved labeled history.

Development search is performed once, wholly before the first untouched outer block:

1. Screen all 40 config IDs in both arms on the earliest eligible inner fold, seed 0, for 5,000
   updates. Each AdamW/spectral pair has identical architecture, batches, examples and updates.
2. Rank each arm independently. Nominate its top 12 by mean CORR, then worst-fold CORR, then lower
   config ID. Form the union of nominated IDs. Run **both** arms for every union ID on every inner
   fold, seed 0, for 20,000 updates. The union rule preserves paired exposure even when the arms
   nominate different architectures or batch sizes.
3. Nominate the top two per arm independently at 5,000 and 20,000 updates. Confirm each paired
   union at its observed budget on both inner folds and seeds 0, 1 and 2. Nominate the top one per
   arm from each observed fidelity for a paired 100,000-update scout on inner fold 1, seed 0. Force
   the separately frozen high-rank source control into the 20,000 confirmation and 100,000 scout.
4. Scout every GPU-admitted high rank at 20,000 updates on both inner folds, seed 0. Promote the
   top two ranks (or all if fewer than two are admitted) by mean CORR, worst-fold CORR and lower
   ID. Give ordinary and high-rank finalists full two-fold, three-seed coverage at their relevant
   budgets. Assemble only candidates with exactly equal fold/seed coverage within a budget and
   select one `(config ID, update budget)` per arm by mean CORR, worst-cell CORR, lower config ID
   and lower budget.
5. Freeze those two winners before any outer score exists. Evaluate the same arm/config/budget
   pairs, seeds 0, 1 and 2, on all three chronological untouched outer blocks. Outer outcomes
   estimate generalization and optimizer difference; they never trigger reselection.

After all three outer folds, report the paired optimizer difference with a circular moving-block
bootstrap using eight-era blocks, 10,000 samples and seed 20260810. Select the benchmark blend from
the concatenated fixed-config outer predictions only. Refit the unchanged winners on all eligible
pre-validation train eras. Freeze configs, seeds, arm-specific training horizons, scoring,
benchmark blend and analysis before constructing or scoring the official validation shard.

The three models used for that sealed historical score remain immutable evidence and are never
relabelled as production models. After the one-time sealed evaluation, the winning frozen
procedure may be refit, without any further selection, on frozen train plus only those validation
rows whose main target is resolved. These separately named production refits are the only models
eligible for the forward unstaked live bundle. Their data manifest, evaluation dependency,
signatures and hashes are audited independently.

## Base-search memory admission (2026-08-11)

The paired base search is frozen before outcomes are observed. Every spectral draw is
screened using the same conservative allocation bound as the high-rank extension: four
fp32 `p × rank` allocations, 16 bytes per model parameter, and four float64
`rank × rank` workspaces must occupy at most 85% of a 48 GiB L40. A draw outside that
safety margin is not automatically called impossible: it requires a completed GPU probe
that reaches the requested rank, passes filter warmup, matches the frozen draw and exact
parameter count, and records peak CUDA memory below physical capacity. If that evidence
is absent after the stage ends, both the AdamW and spectral members of the pair are
excluded before metric-based selection. The admission audit contains no validation score,
so feasibility cannot be selected post hoc from performance.

## High-rank amendment (2026-08-11)

This amendment was made after the ordinary 20,000-update inner-fold search had begun, but before
any outer block or official validation target was revealed. It responds to the pre-existing request
to test global spectral ranks through at least 1,024 and beyond; it is development hyperparameter
search, not a post-test analysis.

1. Using only the ordinary outer-1 inner-fold 20,000-update scores, select the highest-mean-CORR
   spectral architecture that is analytically feasible at rank 2,048. Break ties by worst inner
   cell CORR and then lower config ID. This fixes every setting except rank.
2. Clone that architecture at ranks 512, 1,024, 1,536, 2,048 and 4,096. Give each rank a stable,
   source-specific config ID and retain an identical AdamW search entry as its control.
3. Before admission, analytically screen four fp32 parameter-by-rank allocations, 16 bytes per
   parameter and four float64 rank-squared workspaces against 85% of 48 GiB. GPU-probe every
   analytically feasible rank for enough updates to fill its basis at its actual covariance-update
   cadence and activate filtering. Admit it only if measured peak allocation is at most 45 GiB,
   full rank is realized, and spectral-norm orthogonality error is at most 1e-3. A caught CUDA OOM
   is an audited rejection of that rank; other exceptions stop the workflow.
4. Scout admitted spectral ranks at 20,000 updates on both inner folds, seed 0, enough to fill even
   rank 4,096 at the allowed update cadence. Promote at most two ranks to three-seed confirmation
   at 20,000 and 100,000 updates. Confirm their one unchanged AdamW source through the ordinary
   paired sets rather than duplicating it once per rank.
5. Selection compares only candidates with equal fold/seed coverage within their budget. A
   winning high rank is frozen before and evaluated unchanged on all outer blocks.

The amendment does not use BMC, runtime, any outer score, official validation, live data or
leaderboard outcomes for source/rank selection. Its generated search, source selection, probe
results and exact F2 coverage are immutable hashed artifacts.

Failures are retained. An integrity or deterministic-restart failure is repaired and rerun. OOM is
not silently converted into a smaller model/rank; the exact config is retried on the same free L40
with reduced non-model overhead, otherwise recorded as a failed complete procedure in both paired
arms. Runtime never selects or disqualifies an optimizer.

Artifact cadence is non-scientific: runs retain about 200 diagnostic points and one atomic mid-run
restart checkpoint. Final predictions/results replace the checkpoint on successful completion.
Promoted jobs request 23.5 hours, below the free partition's tested 24-hour QOS ceiling.

## Multi-fidelity promotion amendment (2026-08-12)

This amendment was made before F1 completed, before F2 began, and before any outer or official
validation score existed. Paired development diagnostics showed that 15 of the 17 promoted AdamW
configurations scored lower at 20,000 than at 5,000 updates on the first inner fold. The original
step 3 selected architectures only at 20,000 updates and therefore could discard a strong
5,000-update specialist before training budget was allowed to compete in F2.

To avoid assuming configuration rankings are monotone in training budget, development promotion
preserves independent nominees from both 5,000 and 20,000 updates. This rule uses development CORR
only and was fixed without an outer, official-validation, live or leaderboard outcome. The later
successive-halving amendment below reduces confirmation counts and exposure while preserving that
multi-fidelity principle. The separately selected high-rank source architecture is forced into
both relevant base arms, guaranteeing every high-rank spectral clone has its unchanged source
AdamW control.

## Successive-halving feasibility amendment (2026-08-12)

This amendment was frozen while F1 was running and before F2, high-rank outcomes, any outer score
or official validation existed. Measured F0 runtimes implied that the earlier Cartesian F2 plus
repeated outer searches would require thousands of L40 GPU-hours, making a current leaderboard
comparison stale before completion. Runtime is used only to design this outcome-blind fidelity
schedule; it never ranks or excludes a candidate.

The search now follows steps 3--5 above. Top counts, folds, seeds and tie-breaks are fixed in code.
Every selection input and generated manifest is hashed and audited. Repeated HPO on outer folds 2
and 3 is removed: one pre-outer search is followed by fixed-config walk-forward evaluation on all
three outer blocks, which estimates one frozen procedure without selecting on outer outcomes.

## One-day decision amendment (2026-08-12)

The user stopped the exhaustive search before F1 completed and requested a bounded decision about
whether spectral optimization is a serious practical competitor. All broad-search jobs and their
automatic handoffs were cancelled before this amendment was executed. Completed results and
checkpoints remain immutable evidence, but the partially completed 20,000-update F1 table is not
used to nominate candidates because completion time is configuration-dependent.

The bounded decision uses only the complete paired 5,000-update F0 screen to nominate the best
AdamW configuration and the best spectral configuration independently by mean CORR, worst-cell
CORR, and lower config ID. Their paired union is expected to be IDs 38 and 39. Both arms for both
IDs receive identical 20,000-update confirmation coverage on `outer_1_inner_1` and
`outer_1_inner_2`, seeds 0 and 1; already-complete exact cells may be reused. One winner per arm is
then selected using only those eight equally covered development cells. The winners are frozen
before any outer score exists and evaluated at 20,000 updates on untouched `outer_1`, seeds 0, 1,
and 2. No result from the partially completed broad F1 search, runtime, BMC, outer data, official
validation, live data, or leaderboard outcomes can nominate or reselect a winner.

This amendment is a bounded go/no-go experiment, not the original full leaderboard-certification
protocol. A positive result means spectral merits a later official-validation campaign; a null or
negative result terminates the optimizer comparison. The one-day run does not reveal sealed
official validation and does not create, upload, or stake a live submission.
