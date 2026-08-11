# Benchmark and Leaderboard Snapshot

Retrieved 2026-08-10, round 1329. These two panels are deliberately not put on one numerical scale.

## Released v5.3 validation benchmark

Exact `numerai-tools` CORR, same IDs/eras, resolved rows only:

| Prediction | Scoring target | Eras | Mean CORR | CORR Sharpe |
|---|---|---:|---:|---:|
| `v53_lgbm_ender20` | `target_cyrusd_20` | 652 | 0.040453 | 1.8708 |
| `v53_lgbm_ender20` | `target_ender_20` | 652 | 0.036772 | 1.9645 |
| `v53_lgbm_ender60` | `target_ender_60` | 644 | 0.052909 | 2.6165 |
| `v53_lgbm_ender60` | `target_cyrusd_20` | 652 | 0.039953 | 1.8176 |

These are historical Diagnostics-compatible benchmark scores. They are much larger than live reputations because the targets, eras, training chronology and aggregation differ.

## Public live model leaderboard

NumerAPI returned the top 1,000 current model rows. Across those rows, median one-year reputations were CORR20v2 0.00916, MMC 0.00396, BMC 0.00638 and CORJ60 0.01050. The 90th percentiles were 0.01458, 0.00664, 0.00977 and 0.01895 respectively. The rank-1 model had CORR20v2 0.01584, MMC 0.01274, BMC 0.01710 and CORJ60 0.01806.

This snapshot describes the competitive destination. A historical validation score cannot be mapped to one of these ranks; only a resolved forward submission can enter this distribution.

The reproducible retrieval code and a newer raw official snapshot are archived under
`leaderboard/2026-08-11/`. The dated raw response is authoritative for subsequent comparisons;
this prose snapshot is retained as the original pre-experiment record.
