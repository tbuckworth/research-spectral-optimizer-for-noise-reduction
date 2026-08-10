# Official Numerai contract audit

Retrieved 2026-08-10 from current Numerai documentation, the public NumerAPI at round 1329, `numerai/example-scripts` commit `2447005b3f2af1fd45b883f2e63d1ebb80f75981`, `numerai-tools` commit `54ef5f10c914be411f7cefb938204cb1bd67e847`, and the round-1329 v5.3 dataset snapshot dated 2026-08-07.

## Conclusions that govern the experiment

1. **The current Tournament data version is v5.3.** The downloaded snapshot contains 3,555 `all` features, 780 `medium` features and 42 `small` features. Train contains 2,746,268 rows in 574 weekly eras. Validation contains 4,107,040 rows over eras 0575–1231; 644 eras (through 1218) currently have the generic target and 13 are unresolved.
2. **Official scoring must come from `numerai-tools`.** CORR tie-ranks predictions, Gaussianizes them, raises predictions and centered targets to power 1.5, then computes Pearson correlation. The previous project metric omitted Gaussianization and did not preserve ties, so it is not canonical CORR20v2.
3. **The official benchmark is currently v5.3 Ender.** Public train/validation benchmark files contain `v53_lgbm_ender20` and `v53_lgbm_ender60`. Benchmark predictions are generated with expanding walk-forward models over 156-era prediction blocks, with an 8-era purge for 20D targets and 16 for 60D targets.
4. **The generic released `target` is not a stable synonym for the payout target.** In the round-1329 v5.3 train and resolved validation files, `target` is byte-for-byte `target_ender_60`. The maintained target-ensemble notebook still says that `target` aliases Cyrus, so that prose is stale relative to the current file. Historically, Numerai staff also acknowledged rounds where the generic target alias lagged a payout-target change.
5. **Current live payout definitions remain distinct.** The current definitions page says CORR20v2 and MMC use `target_cyrus_20`, while CORJ60 uses `target_jerome_60`; the public leaderboard exposes separate `corr20V2Rep`, `mmcRep`, `bmcRep` and `corj60Rep`. The released v5.3 file contains `target_cyrusd_20`, not `target_cyrus_20`. Therefore no released historical column has been established as the exact current live payout target.
6. **Offline and live claims must be separated.** Diagnostics evaluate historical validation predictions. The model leaderboard ranks one-year average resolved live scores (`reputation`). BMC in Diagnostics uses the latest benchmark context, while historical live BMC uses the benchmark meta-model available at each old round. Offline CORR/BMC can select a submission candidate, but cannot recreate live Season or reputation.
7. **A competitive artifact may use live benchmark predictions.** Current Model Upload calls `predict(live_features, live_benchmark_models)`. Artifacts run in Python 3.10–3.13 with 4 GB RAM, one CPU and a ten-minute limit. This permits a frozen blend or residual model, provided its implementation fits the runtime contract.
8. **No upload is needed to make the offline comparison.** The public data API and released scoring package are sufficient. A real leaderboard comparison requires an unstaked forward submission and resolved rounds.

## Implication for the objective

There are two honest “comparable numbers,” not one:

- **Historical Diagnostics-compatible evidence:** exact official CORR and BMC on sealed, walk-forward v5.3 predictions, scored against explicitly named released targets and the version-matched Ender benchmark. The target name must be printed beside every result; the generic alias must never hide it.
- **Leaderboard evidence:** prospective CORR20v2, MMC, BMC, CORJ60 and Season/reputation from frozen unstaked submissions. This is the only exact comparison to the current leaderboard.

The HPO should therefore optimize on pre-validation walk-forward folds using named targets, freeze candidate pipelines, open the official validation once for historical comparison, and then prepare parallel forward AdamW/spectral/blend submissions. It should not claim that a historical Ender-60 or Cyrusd-20 score is numerically interchangeable with current live Cyrus-20.

## Primary sources

- [Data](https://docs.numer.ai/numerai-tournament/data)
- [Models and benchmark construction](https://docs.numer.ai/numerai-tournament/models)
- [Scoring and leaderboard reputation](https://docs.numer.ai/numerai-tournament/scoring/)
- [Current score definitions](https://docs.numer.ai/numerai-tournament/scoring/definitions)
- [CORR implementation description](https://docs.numer.ai/numerai-tournament/scoring/correlation-corr)
- [MMC/BMC and the Diagnostics/live distinction](https://docs.numer.ai/numerai-tournament/scoring/meta-model-contribution-mmc)
- [Model Upload runtime](https://docs.numer.ai/numerai-tournament/submissions/model-uploads)
- [Official example scripts](https://github.com/numerai/example-scripts/tree/2447005b3f2af1fd45b883f2e63d1ebb80f75981/numerai)
- [Official scoring source](https://github.com/numerai/numerai-tools/blob/54ef5f10c914be411f7cefb938204cb1bd67e847/numerai_tools/scoring.py)
- [Historical staff confirmation that released alias and payout target can differ](https://forum.numer.ai/t/target-cyrus-new-primary-target/6303?page=2)
