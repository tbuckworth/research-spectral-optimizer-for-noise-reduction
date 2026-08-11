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

## Artifact disposition

All completed auxiliary-target artifacts are retained as optimizer-method and
training-budget evidence. They are explicitly excluded from final selection,
sealed validation, live-candidate construction, and leaderboard claims.

When the mismatch was confirmed, jobs 8499 through 8577 were cancelled and the
associated promotion/supervisor sessions were stopped. No target-corrected jobs
had started, and no official validation target was revealed.

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
