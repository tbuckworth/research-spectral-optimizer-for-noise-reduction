# Planned Experiments

> **Superseded target field (2026-08-11):** this pre-experiment document selected
> `target_cyrusd_20`. That was corrected before the current run to the documented
> v5.3 main `target`; see
> `experiments/exp-008/target-protocol-correction.md`. Historical main-target
> evidence remains distinct from live CORR20v2 because the pinned main target
> equals `target_ender_60`, while live reputation uses unreleased
> `target_cyrus_20`.

## Exp-008A — Integrity and current-data harness

Build a v5.3 train-only shard with explicit feature/target metadata and hashes. Implement whole-era walk-forward splits, exact `numerai-tools` CORR/BMC, corruption tests, sealed validation access, deterministic configs/checkpoints and per-era prediction artifacts. Reproduce the released Ender benchmark snapshot as an integration test.

## Exp-008B — MLP and spectral feasibility

On RTX 3090, verify finite training, memory bounds, restart determinism, strength-zero equivalence, basis orthonormality and optimizer ordering. Compare learned spectral projection with persistent random-orientation, shuffled-history and norm-matched controls. Throughput is logged but not a pass/fail condition.

## Exp-008C — Symmetric nested HPO

Primary target: `target_cyrusd_20`; primary endpoint: standalone exact CORR. Use nested expanding walk-forward selection inside eras 0001–0574 with an 8-era purge. Run 40 AdamW and 40 spectral complete-procedure configurations under identical three-fidelity schedules. Match shared search distributions, completed configurations, examples/updates, folds and seeds. Use MATS free L40 Slurm arrays; use RTX 3090 for smoke and confirmation. Select each procedure only from inner folds and estimate it on untouched outer development folds.

Shared search dimensions: medium/all feature set; MLP width/depth/residual/normalization/activation; batch construction and size; learning rate; weight decay; dropout; schedule/warm-up; clipping; MSE/Huber and predeclared target transformation. Spectral additionally samples rank, decay, warm-up, strength/weighting and update cadence inside its 40 configurations.

## Exp-008D — Freeze and one-time official validation

Freeze selected procedures, seeds, stopping horizon, named target, exact analysis code, fixed subperiods, rolling windows, bootstrap rule and all secondary transformations. Refit on allowed train eras and reveal official validation once. Primary claim uses spectral-minus-AdamW paired CORR with a 95% moving-block-bootstrap interval. Secondary outputs include Ender-20/60 targets, BMC, Sharpe, drawdown, rolling plots, seed sensitivity and benchmark comparisons.

## Exp-008E — Offline candidate and submission bundle

Using train-only nested OOF predictions, predeclare secondary benchmark blends/neutralization and practical effect-size rules. Package the strongest eligible route as an offline submission candidate with a no-upload default, target-free live fixture, conventional prediction command, and Model Upload-compatible callable if it fits the 4 GB/one-CPU/ten-minute contract. Actual upload/submission/staking remains separately gated.

## Autonomous fail-fast agreement

The challenge revisions are accepted. Engineering-integrity failures stop and are repaired before scoring. Null or negative spectral efficacy does not stop its symmetric search once integrity passes; it is a valid result. Official validation stays sealed until both procedures and the analysis are frozen. Free compute is authorized; paid compute and external submission are not.
