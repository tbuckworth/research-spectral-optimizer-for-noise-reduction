# Main-target protocol correction

Date: 2026-08-11

## Error and impact

The first exp-008 searches set the primary target to the auxiliary column
`target_cyrusd_20`. Current Numerai v5.3 documentation and the official example
scripts instead identify the generic `target` column as the main tournament
development objective. Current live CORR20v2 reputation is scored against
`target_cyrus_20`, which is not released in the pinned historical files. The
released `target_cyrusd_20` auxiliary column is not interchangeable with either
the generic `target` or the unreleased live payout endpoint.

The two released columns are materially different. On the pinned v5.3 training
data their row-level correlation is approximately 0.4175702743. Therefore,
auxiliary-target search results cannot be relabelled or used for the final
main-target model selection. The pinned generic `target` is also byte-identical
to `target_ender_60`; consequently, historical main-target CORR is
Diagnostics-compatible evidence, not a reconstructed live CORR20v2 reputation.
Only forward unstaked submissions can establish direct leaderboard comparability.

## Evidence

- Numerai Data documentation describes `target` as the main target and the
  named target columns as auxiliary targets:
  <https://docs.numer.ai/numerai-tournament/data>
- Numerai scoring documentation defines current CORR20v2:
  <https://docs.numer.ai/numerai-tournament/scoring/definitions>
- Official `numerai/example-scripts` commit
  `2447005b` uses v5.3 with `target_col: "target"`.
- Official `numerai-tools` commit `54ef5f10` reports package version 0.6.0,
  matching the project lock.

The pinned v5.3 source hashes remain unchanged; this correction changes which
released column is the primary objective and adds the main target to materialized
shards.

The matching official baseline is `v53_lgbm_ender60`, not
`v53_lgbm_ender20`. The latter was mistakenly used in the first corrected-target
campaign even though both official benchmark columns were already present in the
shards. That campaign's primary standalone CORR remains descriptive, but its BMC,
blend, and benchmark comparisons are excluded from selection and final reporting.

## Artifact disposition

All completed auxiliary-target artifacts are retained as optimizer-method and
training-budget evidence. They are explicitly excluded from final selection,
sealed validation, live-candidate construction, and leaderboard claims.

When the mismatch was confirmed, jobs 8499 through 8577 were cancelled and the
associated promotion/supervisor sessions were stopped. No target-corrected jobs
had started, and no official validation target was revealed.

On 2026-08-12 the benchmark mismatch above was found before F0 promotion. Jobs
8618 through 8697 were cancelled, all controllers were stopped, and 15 completed
result files (including the smoke gate) were retained under
`results-ender20-wrong-benchmark-20260812`. Those model gradients and standalone
CORR values did not depend on the benchmark, but the campaign is excluded wholesale
to avoid mixing Ender20 BMC/blend artifacts with the corrected Ender60 procedure.
Official validation remained sealed.

That Ender60 relaunch still inherited the earlier 20-day split implementation and
therefore purged only eight eras. On 2026-08-12, before the first spectral cell or
any promotion completed, jobs 8727 through 8806 were cancelled and the two result
files (smoke plus one AdamW cell) were retained under
`results-ender60-wrong-8era-purge-20260812`. The corrected main-target protocol uses
16-era purges at every development and sealed-validation boundary.

## Corrected procedure

1. Materialize fresh v5.3 shards containing the main `target`.
2. Repeat train-era-only nested AdamW selection, including update budget as a
   hyperparameter, with the official validation set unavailable.
3. Run the matched spectral search and rank sensitivity on identical splits,
   seeds, examples, and update budgets.
4. Freeze code, configurations, seeds, candidate transform, and model artifacts.
5. Reveal official validation once for the sealed comparison.
6. Build and resource-test an unstaked live candidate. Upload, submission, and
   staking remain outside this procedure unless separately authorized.
