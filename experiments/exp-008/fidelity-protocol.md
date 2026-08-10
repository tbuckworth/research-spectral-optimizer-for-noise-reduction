# Frozen multi-fidelity protocol

Frozen before completion of the first screen. The primary selection statistic is mean exact
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

Failures are retained. An integrity or deterministic-restart failure is repaired and rerun. OOM is
not silently converted into a smaller model/rank; the exact config is retried on the same free L40
with reduced non-model overhead, otherwise recorded as a failed complete procedure in both paired
arms. Runtime never selects or disqualifies an optimizer.

Artifact cadence is non-scientific: runs retain about 200 diagnostic points and one atomic mid-run
restart checkpoint. Final predictions/results replace the checkpoint on successful completion.
Promoted jobs request 23.5 hours, below the free partition's tested 24-hour QOS ceiling.
