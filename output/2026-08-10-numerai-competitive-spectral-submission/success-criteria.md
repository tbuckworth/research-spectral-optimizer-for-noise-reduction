# Success Criteria

## Non-negotiable validity gates

1. **Current contract:** every headline result uses Numerai Tournament v5.3, explicitly named target columns (never an unexplained generic alias), version-matched IDs/eras and benchmark predictions, and the released `numerai_tools.scoring` implementation pinned by version/commit.
2. **No temporal leakage:** all model, feature, loss, optimizer, stopping, blending and neutralization choices are made using expanding walk-forward folds with the target-horizon embargo. Official validation is read only after both optimizer configurations and the analysis script are frozen.
3. **Comparable predictions:** AdamW and spectral arms use identical input rows, features, architecture search envelope, target construction, batch streams, evaluation eras and post-processing envelope. Search budgets and failed trials are recorded.
4. **Reproducibility:** immutable data hashes, environment lock, configs, seeds, checkpoints, raw per-era predictions and metrics are retained. A clean command reproduces every table and plot.
5. **Metric parity:** unit tests match Numerai's own CORR and contribution functions, preserve tied ranks, and reject row/era/benchmark misalignment.

## AdamW baseline gate

- Run a predeclared multi-fidelity HPO, not a hand-picked learning-rate check.
- Search training dynamics (learning rate, weight decay, batch construction, schedule, warm-up, clipping and stopping) and model regularization/capacity within a fixed MLP family.
- Promote configurations by robust walk-forward performance, not their best single fold or seed.
- Freeze one standalone AdamW model and one permitted benchmark-blend/post-processing recipe before official-validation access.
- Demonstrate that the selected configuration beats the original study's AdamW recipe on the same inner folds and is not a collapsed or leakage-suspect model.

## Spectral comparison gate

- Start from the frozen AdamW pipeline and tune only predeclared spectral/filter parameters under an equal trial/compute accounting.
- Report paired per-era differences against tuned AdamW for exact CORR, BMC, Sharpe and drawdown across folds and confirmation seeds.
- Call spectral an improvement only if its mean paired CORR difference is positive and its block-bootstrap 95% interval excludes zero without a material BMC or drawdown regression. Otherwise report a tie or failure.

## Offline competitiveness gate

On the untouched official validation period, report the candidate and `v53_lgbm_ender20` benchmark on identical eras for:

- mean CORR20v2, standard deviation, Sharpe, maximum drawdown and cumulative CORR;
- BMC against the official benchmark, prediction correlation with the benchmark, and rolling 52-era stability;
- raw, neutralized and benchmark-blended predictions where those transformations were frozen upstream;
- paired era-block bootstrap intervals for candidate-minus-benchmark and spectral-minus-AdamW.

A candidate is **submission-worthy** if it is either (a) competitive in standalone CORR with the official benchmark, or (b) has reliably positive BMC and improves a frozen blend with the benchmark, while passing stability and integrity gates. This is deliberately relative to the current benchmark rather than an obsolete absolute CORR threshold.

## Live-leaderboard gate

- Produce a model-upload-compatible artifact and a conventional live prediction path, each tested against Numerai's current runtime and schema contract.
- Do not upload, submit or stake without explicit user authorization.
- Describe offline performance as Diagnostics-compatible, never as leaderboard reputation.
- After authorization, compare frozen unstaked AdamW, spectral and candidate-blend models prospectively. A leaderboard-quality claim requires resolved live rounds; the current leaderboard's one-year reputation cannot be recreated retrospectively.
