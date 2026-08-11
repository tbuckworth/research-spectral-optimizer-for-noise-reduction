# Multi-fidelity protocol and pre-validation amendment

The base protocol below was frozen before completion of the first screen. The primary selection statistic is mean exact
standalone Numerai CORR. BMC, Sharpe, runtime and official validation do not select configs.

For each outer fold independently:

1. Screen all 40 config IDs in both arms on the earliest eligible inner fold, seed 0, for 5,000
   updates. Each AdamW/spectral pair has identical architecture, batches, examples and updates.
2. Rank each arm independently. Nominate its top 12 by mean CORR, then worst-fold CORR, then lower
   config ID. Form the union of nominated IDs. Run **both** arms for every union ID on every inner
   fold, seed 0, for 20,000 updates. The union rule preserves paired exposure even when the arms
   nominate different architectures or batch sizes.
3. Re-rank from the 20,000-update folds. Nominate the top four per arm and again run both arms on
   their union, on every inner fold, seeds 0, 1 and 2, for 100,000 updates.
4. Select one config per arm using mean CORR across all eligible inner folds and seeds, then
   worst-fold CORR and lower config ID. Refit on the outer training eras for 100,000 updates at
   seeds 0, 1 and 2 and score the untouched outer block.

After all three outer folds, report the paired outer-fold optimizer difference with a circular
moving-block bootstrap using eight-era blocks, 10,000 samples and seed 20260810. For final
train-only selection, take the union of the three outer-fold winners, run both
arms for every ID over all four canonical train-era folds at 100,000 updates and seeds 0, 1 and 2,
then choose one config per arm by the same rule. Freeze configs, seeds, training horizon, scoring,
benchmark blends and analysis before constructing or scoring the official validation shard.

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
   and 2. Their AdamW configuration is identical across ranks, so train that exact control through
   the ordinary paired union rather than duplicating it once per rank. Rank-specific AdamW copies
   cannot gain a selection advantage from redundant stochastic replicas. F2 arm/config coverage
   is therefore deliberately asymmetric and audited exactly.
5. Rank AdamW and spectral candidates independently over equal fold/seed/update coverage. If a
   high-rank spectral candidate wins an outer fold, evaluate it normally on the untouched outer
   block. The final canonical-fold winner union again runs both arms for every winning ID, restoring
   full cross-arm evaluation before immutable freeze.

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
