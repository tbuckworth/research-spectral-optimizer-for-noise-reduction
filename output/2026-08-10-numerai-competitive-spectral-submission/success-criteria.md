# Success Criteria

## Non-negotiable validity gates

1. **Current contract:** every headline result uses Numerai Tournament v5.3, explicitly named target columns (never an unexplained generic alias), version-matched IDs/eras and benchmark predictions, and the released `numerai_tools.scoring` implementation pinned by version/commit.
2. **No temporal leakage:** all model, feature, loss, optimizer, stopping, blending and neutralization choices are made using expanding walk-forward folds with the target-horizon embargo. Official validation is read only after both optimizer configurations and the analysis script are frozen.
3. **Comparable procedures:** AdamW and spectral arms use identical input rows, features, shared architecture/training search envelope, target construction, batch streams and evaluation eras. Each receives 40 completed configurations under the same fidelity schedule, folds, updates/examples and confirmation seeds. Runtime is recorded but not matched. Failed numerical configurations remain outcomes; infrastructure failures are rerun.
4. **Reproducibility:** immutable data hashes, environment lock, configs, seeds, checkpoints, raw per-era predictions and metrics are retained. A clean command reproduces every table and plot.
5. **Metric parity:** unit tests match Numerai's own CORR and contribution functions, preserve tied ranks, and reject row/era/benchmark misalignment.

## AdamW baseline gate

- Run a predeclared multi-fidelity HPO, not a hand-picked learning-rate check.
- Search training dynamics (learning rate, weight decay, batch construction, schedule, warm-up, clipping and stopping) and model regularization/capacity within a fixed MLP family.
- Promote configurations by robust walk-forward performance, not their best single fold or seed.
- Evaluate the complete AdamW HPO-and-selection procedure on nested outer development folds, then freeze one standalone AdamW model before official-validation access.
- Demonstrate that the selected configuration beats the original study's AdamW recipe on the same inner folds and is not a collapsed or leakage-suspect model.

## Spectral comparison gate

- Give the complete spectral procedure the same 40-configuration, fidelity, fold, update/example and seed budget as AdamW. Its search includes the same shared hyperparameters plus predeclared filter parameters; it is not forced to inherit AdamW-optimal learning dynamics.
- Report paired nested-outer-fold differences against tuned AdamW for exact CORR, with BMC, Sharpe, drawdown and mechanism diagnostics secondary.
- Call spectral an improvement only if its mean paired CORR difference is positive and its block-bootstrap 95% interval excludes zero without a material BMC or drawdown regression. Otherwise report a tie or failure.

## Primary endpoint and offline-candidate gate

The single confirmatory optimizer endpoint is standalone exact CORR on `target_cyrusd_20`, spectral minus AdamW, over the sealed resolved v5.3 official-validation eras. `target_ender_20`, target ensembles, Ender-60, BMC, neutralization and benchmark blends are predeclared secondary analyses and cannot rescue the primary optimizer claim. Development uncertainty comes from nested outer predictions, not a bootstrap of HPO winner scores.

On the untouched official validation period, report the candidate and `v53_lgbm_ender20` benchmark on identical eras for:

- mean CORR20v2, standard deviation, Sharpe, maximum drawdown and cumulative CORR;
- BMC against the official benchmark, prediction correlation with the benchmark, and rolling 52-era stability;
- raw, neutralized and benchmark-blended predictions where those transformations were frozen upstream;
- paired era-block bootstrap intervals for candidate-minus-benchmark and spectral-minus-AdamW.

A model is an **offline submission candidate** only if its train-only nested outer predictions show stable positive signal across folds/seeds and either useful standalone benchmark-relative CORR or reliably positive BMC/blend contribution of predeclared practical size. There is no arbitrary 95%-of-benchmark cutoff. “Competitive” is reserved for resolved prospective live evidence.

## Live-leaderboard gate

- Produce a model-upload-compatible artifact and a conventional live prediction path, each tested against Numerai's current runtime and schema contract.
- Do not upload, submit or stake without explicit user authorization.
- Describe offline performance as Diagnostics-compatible, never as leaderboard reputation.
- After authorization, compare frozen unstaked AdamW, spectral and candidate-blend models prospectively. A leaderboard-quality claim requires resolved live rounds; the current leaderboard's one-year reputation cannot be recreated retrospectively.
