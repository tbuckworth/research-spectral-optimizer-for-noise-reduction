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

For each outer fold independently:

1. Screen all 40 config IDs in both arms on the earliest eligible inner fold, seed 0, for 5,000
   updates. Each AdamW/spectral pair has identical architecture, batches, examples and updates.
2. Rank each arm independently. Nominate its top 12 by mean CORR, then worst-fold CORR, then lower
   config ID. Form the union of nominated IDs. Run **both** arms for every union ID on every inner
   fold, seed 0, for 20,000 updates. The union rule preserves paired exposure even when the arms
   nominate different architectures or batch sizes.
3. Re-rank from the 20,000-update folds. Nominate the top four per arm and run both arms on their
   union, on every inner fold and seeds 0, 1 and 2, at 5,000, 20,000 and 100,000 updates. Reuse
   only exact pre-existing cells. Training budget is therefore a development hyperparameter, not
   an assumed constant.
4. Select one `(config ID, update budget)` pair per arm using mean CORR across equal eligible
   inner-fold/seed coverage, then worst-cell CORR, lower config ID and lower budget. Refit each
   selected arm on the outer training eras at its selected budget for seeds 0, 1 and 2 and score
   the untouched outer block.

After all three outer folds, report the paired outer-fold optimizer difference with a circular
moving-block bootstrap using eight-era blocks, 10,000 samples and seed 20260810. For final
train-only selection, take the union of the three outer-fold winners, run both
arms for every nominated configuration/budget pair over all four canonical train-era folds and
seeds 0, 1 and 2, then choose one pair per arm by the same rule. Freeze configs, seeds,
arm-specific training horizons, scoring,
benchmark blends and analysis before constructing or scoring the official validation shard.

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
4. Add admitted spectral ranks to the 100,000-update F2 cells on every inner fold and seeds 0, 1
   and 2. High ranks are not evaluated at shorter budgets that cannot fill and activate their
   basis. Their AdamW configuration is identical across ranks, so train that exact control through
   the ordinary paired union rather than duplicating it once per rank. Rank-specific AdamW copies
   cannot gain a selection advantage from redundant stochastic replicas. F2 arm/config coverage
   is therefore deliberately asymmetric and audited exactly.
5. Every candidate pair has equal fold/seed coverage. Base candidates compete over all three
   budgets; high-rank spectral candidates compete at 100,000 updates only. If a high-rank spectral
   candidate wins an outer fold, evaluate it at that budget on the untouched outer block. The
   matched AdamW source architecture remains in the paired base union.

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
