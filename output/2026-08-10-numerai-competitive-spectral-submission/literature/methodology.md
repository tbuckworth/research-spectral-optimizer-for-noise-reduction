# Methodology literature: leakage-resistant optimizer comparison on Numerai panels

**Scope.** This report executes Group 2 of the search plan. It concentrates on maintained official Numerai examples and primary literature that bears directly on temporal validation, nested model selection, optimizer benchmarking, and frequency-domain optimization. Sources were checked on 2026-08-10. Platform details should be re-pinned when the experiment runs because the data version and examples change.

> **Correction by the orchestrator:** the spectral literature portion below searched FFT/frequency-domain gradient methods. That is not the optimizer under test, which maintains top eigenvectors of a streaming p-by-p gradient covariance through rank-one updates. The FFT/SMI mechanism discussion is excluded from the project synthesis and experimental rationale. The Numerai, temporal-validation and fair-optimizer-comparison sections remain applicable.

## Bottom line

The defensible design is an era-level, chronological, nested evaluation with a target-horizon purge, several inner walk-forward validation windows, and one latest contiguous final holdout that is never used for architecture choice, early stopping, hyperparameter search, spectral-filter design, or ablation choice. AdamW and the spectral method need separate, predeclared search spaces but equal trial/compute budgets. All rows from an era stay together, and uncertainty is calculated over eras or era blocks, not over the much larger number of stock rows.

Numerai's maintained materials support this design more strongly than its introductory notebooks alone. The current benchmark-model documentation uses expanding walk-forward training, 156-era prediction blocks, and purges of 8 eras for 20-day targets and 16 for 60-day targets [@numerai_models_2026]. By contrast, the current `hello_numerai`, feature-neutralization, and target-ensemble notebooks remove only the first four validation eras after the training boundary for a 20-day target [@numerai_examples_2026]. The four-era notebook rule is a useful minimum overlap embargo; the benchmark construction is the stronger current precedent for official walk-forward comparisons. The experiment should therefore pin the target and adopt the current official benchmark purge (8/16) unless a source-code audit establishes a different target-specific information interval.

The spectral case is plausible enough to test but not established. Spectral-bias papers concern frequencies of the **learned function with respect to input coordinates**, whereas the closest direct optimizer paper applies an FFT to reshaped **parameter-gradient arrays** [@rahaman_2019_spectral_bias; @xu_2020_frequency_principle; @huang_2025_smi]. Those are different objects. No located primary work demonstrates that FFT filtering of MLP parameter gradients improves generalization in era-indexed financial panels. The direct SMI evidence is a two-seed, single-dataset, 10.7M-parameter character-level transformer study; its validation-loss changes versus AdamW are about -0.27% to +0.14%, and the authors explicitly identify limited data/architecture coverage, heuristic frequency selection, unresolved convergence bias, and unresolved generalization theory [@huang_2025_smi]. This makes the Numerai experiment a genuine domain-transfer test, not a replication of an established financial method.

## Official maintained Numerai examples

### What is currently maintained

The official `numerai/example-scripts` repository identifies four Tournament notebooks: Hello Numerai, Feature Neutralization, Target Ensemble, and Model Upload. The repository was inspected at commit `2447005b3f2af1fd45b883f2e63d1ebb80f75981` (2026-07-16); all four notebooks pin data version `v5.3` [@numerai_examples_2026]. The official docs link the tutorials as the recommended entry point and separately document benchmark construction [@numerai_models_2026]. `numerai-tools` is the maintained scoring/submission package and states that its scoring functions are used in Numerai's scoring system [@numerai_tools_2026].

The notebooks establish a reproducible baseline sequence:

1. download `features.json`, `train.parquet`, and `validation.parquet` with `NumerAPI`;
2. select a documented feature set, fit an LGBM model, and predict validation rows only;
3. preserve eras and compute CORR/MMC per era with `numerai-tools`;
4. inspect cumulative and summary era metrics;
5. optionally rank predictions within era, ensemble targets, or neutralize predictions;
6. serialize a `predict(live_features, live_benchmark_models)` function for Model Upload.

These are operational tutorials, not a model-selection protocol. They fit once on the complete training partition, inspect the complete validation partition, and invite experiments with feature sets, targets, ensemble weights, and neutralization proportions. Repeatedly following those invitations on the same validation eras turns that partition into development data. The `example_model` notebook says that joining train and validation may improve the final model but makes Diagnostics misleading; the scoring docs similarly warn against validation overfitting [@numerai_examples_2026; @numerai_scoring_2026]. Thus an experiment making comparative claims must add nested temporal selection around the tutorial pipeline.

### Era and overlap semantics

Numerai describes each row as one stock at one date and each era as one weekly cross-section. Stock identity cannot be followed across eras, while the 20-day and 60-day forward targets overlap across adjacent weekly eras [@numerai_data_2026]. Consequences:

- The effective evaluation unit is the era, not a row. Random row splits would place the same market state on both sides and are invalid.
- Every preprocessing fit that learns across rows or eras—imputation, scaling, feature selection, neutralization coefficients if learned historically, early stopping, and HPO—must be fitted using only the training side of a fold.
- A purge belongs between the latest training label interval and earliest validation era. An ordinary chronological split without this gap can still leak overlapping forward returns.
- Within-era rank transforms used only on predictions from that same era are compatible with live inference. A transform whose parameters use future eras is not.

The benchmark docs provide the most concrete official template: train through era 148, purge eras 149–156, predict 157–312; then train through 304, purge 305–312, predict 313–468, and continue [@numerai_models_2026]. This is expanding-window walk-forward validation. It should be treated as a benchmark-generation precedent, not as proof that 156 eras or an expanding window is universally optimal.

## Primary evidence on temporal and nested selection

### What transfers cleanly

Cawley and Talbot show that the model-selection criterion itself can be overfit and that this effect can be as large as differences between algorithms. An unbiased but high-variance selection estimate is not sufficient; performance evaluation must be outside the selection loop [@cawley_talbot_2010]. Varma and Simon demonstrate the same issue experimentally and show that nested cross-validation greatly reduces error-estimation bias after model selection [@varma_simon_2006]. These results are not finance-specific, but the logic applies directly to choosing between AdamW and spectral variants after HPO.

For temporal data, generic nested K-fold is not enough. Bergmeir, Hyndman, and Koo show that ordinary K-fold can be valid for a restricted autoregressive setting when errors are uncorrelated [@bergmeir_hyndman_koo_2018]. That assumption is not a safe default for nonstationary financial panels with overlapping labels. Cerqueira, Torgo, and Mozetič find that cross-validation is competitive for stationary series but that order-preserving out-of-sample methods estimate performance better under nonstationarity; they recommend repeated holdouts/test periods in realistic nonstationary settings [@cerqueira_torgo_mozetic_2020]. The appropriate transfer is multiple chronological inner windows, not shuffled folds.

Gu, Kelly, and Xiu provide the closest high-quality panel-finance analogue. They split a large stock-month panel into disjoint, temporally ordered training, validation, and test periods; tune only on validation; call the test sample the only truly out-of-sample evaluation; and roll training and validation forward during the long test period [@gu_kelly_xiu_2020]. Their rows are firms within months, analogous to stocks within Numerai eras. Their exact horizons are not Numerai defaults, but their separation of estimation, tuning, and testing is directly applicable.

Financial backtesting research adds two cautions. Bailey et al. quantify how selecting the best historical strategy can produce backtest overfitting and propose combinatorially symmetric cross-validation to estimate its probability [@bailey_etal_2017]. Arnott, Harvey, and Markowitz emphasize a research protocol that records trials and protects against repeated specification search in data-poor, adaptive markets [@arnott_harvey_markowitz_2019]. Neither paper validates the exact “combinatorial purged cross-validation” implementation often circulated in libraries for a Numerai-style panel. Purging is justified here directly by Numerai's overlapping targets; combinatorial rearrangement is not a license to train on future eras when the deployment claim is forward prediction.

### What does not transfer without qualification

- Results proving ordinary K-fold validity for stationary autoregressions do not establish validity for obfuscated cross-sectional panels under regime change.
- CPCV/CSCV estimates the instability or overfitting of historical strategy selection, but folds that train on eras later than their test eras answer an interpolation/exchangeability question rather than the intended live-deployment question.
- A single latest holdout is temporally realistic but can be regime-specific. Multiple inner walk-forward windows reduce this variance; the untouched final holdout remains necessary to estimate the selected pipeline.
- A final holdout ceases to be a holdout after its results influence any redesign. Subsequent changes require a new forward period or must be reported as exploratory.

## Fair optimizer comparison

Optimizer rankings are unusually sensitive to tuning protocol. Choi et al. show that changing optimizer search spaces can reverse empirical conclusions [@choi_etal_2020]. Schmidt, Schneider, and Hennig find strong task dependence and no optimizer that consistently dominates across more than 50,000 standardized runs [@schmidt_schneider_hennig_2021]. Consequently, “AdamW tuned first, then add a spectral wrapper” is not fair if the spectral method gets extra adaptive choices or if AdamW's chosen configuration is frozen while the spectral method sees later validation evidence.

A fair budget has three layers:

- **Data budget:** identical features, target, era folds, training examples, and permissible preprocessing.
- **Optimization budget per trial:** identical maximum updates/examples and comparable wall-clock or accelerator budget. Report both because FFT processing adds overhead.
- **Selection budget per method:** the same number of completed configurations and seeds, or the same total accelerator time under a predeclared scheduler. The spectral method's retention threshold, EMA, blend schedule, and FFT-axis/reshape choices count against this budget.

Search spaces need not have identical dimensions; each must expose the parameters required to make that method competitive. Predeclare distributions and a budget-aware sampler. Compare methods as complete tuning procedures, not just their best hand-picked runs. Keep trial logs, including failed runs. Use the same seed set across paired AdamW/spectral evaluations where possible.

## Spectral optimization and generalization literature

Rahaman et al. provide empirical and theoretical evidence that ReLU networks learn low-frequency functions before high-frequency functions, with dependence on data-manifold geometry [@rahaman_2019_spectral_bias]. Xu et al. report a similar frequency principle and connect it to implicit low-frequency bias [@xu_2020_frequency_principle]. Zhang et al. derive an explicit frequency-principle norm and a generalization bound for sufficiently wide two-layer ReLU networks in a limited low-dimensional setting [@zhang_etal_2019_fp_norm]. These works support a mechanism hypothesis—optimization can favor smoother learned functions—but do not imply that high-index coefficients of a reshaped parameter gradient are noise.

SMI is the closest direct antecedent found. It takes a 2-D FFT of each reshaped parameter gradient, tracks an EMA of magnitude, keeps coefficients above a quantile threshold, inverse-transforms, and blends the result with the original gradient before a base optimizer step [@huang_2025_smi]. Important qualifications are:

- The mask is magnitude-selective, not intrinsically low-pass. “High frequency” and “small magnitude” are not synonyms.
- FFT frequency depends on tensor axes and element ordering. Hidden units in an MLP can be permuted with compensating permutations in adjacent layers without changing the represented function, yet the gradient spectrum of the stored arrays can change. Any claimed spectral effect therefore needs a neuron-permutation or reshape-axis invariance test.
- The spectral-bias literature transforms the learned function over meaningful input coordinates. SMI transforms parameter arrays. The link between these spectra is a hypothesis, not a theorem.
- In Numerai, the row dimension is a stock cross-section and the era dimension is time, but neither is an axis of a weight tensor. Filtering weight-gradient frequencies is not temporal smoothing of era outcomes.
- The primary SMI study uses two seeds and one Shakespeare task; it does not establish financial generalization. Its strongest reported change is inference throughput, a result especially difficult to attribute causally when architecture and nominal operation count are unchanged.

The right ablations are therefore stronger than “spectral versus AdamW”:

1. AdamW with identical schedule and regularization;
2. SMI/spectral method at matched training steps and at matched wall-clock budget;
3. a no-filter wrapper controlling for implementation overhead;
4. an energy/sparsity-matched random frequency mask;
5. a time-domain magnitude mask or clipping control;
6. alternative tensor reshape/FFT axes and a function-preserving hidden-unit permutation;
7. filter-strength zero and several predeclared strengths;
8. identical early-stopping and seed rules.

A spectral method should be credited only if it improves the untouched era-level holdout under the predeclared selection procedure, survives these controls, and does not merely shift CORR against exposure, drawdown, or contribution metrics.

## Recommended leakage-resistant protocol

### 1. Freeze the contract

Pin repository commits, package versions, dataset version, target, feature set, benchmark-prediction files, metric implementation, random seeds, hardware, and the ordered era list. Verify that benchmark predictions cover exactly the scored rows and data version. Define the primary metric and tie-breakers before HPO.

### 2. Reserve the outer final holdout

Take the latest contiguous block of resolved eras of adequate length as `H_final`. Place the target-specific purge immediately before it. No prediction, score, diagnostic, plot, feature statistic, or early-stopping decision from `H_final` is available until all methods, spaces, budgets, ablations, and selection rules are frozen. If the project has already inspected the official validation partition repeatedly, that partition is development data; use a later untouched era block or forward live results for confirmatory evidence.

### 3. Build inner walk-forward folds from pre-holdout eras

Use at least three temporally ordered validation blocks spanning different regimes. For inner fold (k): train only on eras before the fold, remove the 8-era (20D) or 16-era (60D) official benchmark purge, and validate on the next contiguous block. Expanding training windows match the official benchmark; a fixed-length rolling window can be a preregistered sensitivity analysis. Never split stocks within an era.

### 4. Run nested HPO

Each candidate configuration is trained and scored on exactly the same inner folds. Aggregate per-era scores first within each fold and then across folds; do not weight eras by row count. Select with a preregistered objective, for example mean per-era CORR subject to exposure/drawdown constraints, rather than opportunistically choosing whichever metric favors a method. Early stopping is inner-fold training logic and cannot see the final holdout.

Give AdamW and spectral procedures equal search budget. Spectral-only hyperparameters consume that budget. If architecture, dropout, batch size, or target choice are also tuned, tune them inside the same nesting and count all trials. A cleaner causal optimizer test freezes the architecture and non-optimizer pipeline before comparing optimizer procedures.

### 5. Lock and evaluate once

Choose one configuration per method using inner results only. Refit on all permitted pre-holdout eras, preserving a purge before `H_final`; determine training duration from inner folds rather than final-holdout early stopping. Score each frozen model once on `H_final` and join official benchmarks on identical rows.

Use paired per-era differences between spectral and AdamW. Report the full era series, mean/median, dispersion, worst block, drawdown, and relevant official metrics. Estimate uncertainty with a moving-block bootstrap or HAC-style procedure over eras; an ordinary row bootstrap would create spurious precision. Include multiple training seeds and distinguish seed variance from temporal-regime variance.

### 6. Interpret claims narrowly

Passing this protocol supports “better on the frozen historical holdout under this search budget.” It does not establish live leaderboard competitiveness. Once the holdout is opened, any redesign is exploratory. The next confirmatory test is a genuinely forward, unstaked live submission observed until targets resolve.

## Evidence gaps

- No primary paper located validates one canonical nested/purged HPO recipe specifically for anonymized, era-indexed cross-sectional panels with Numerai's overlapping targets.
- No located study tests FFT parameter-gradient filtering on Numerai or a comparable cross-sectional return panel.
- The direct spectral-optimizer literature is sparse; the closest paper has two seeds, one language dataset, heuristic filtering, and no established optimizer-to-generalization theory.
- Official introductory notebooks and official benchmark docs use different 20D gaps (four-era validation embargo versus eight-era benchmark purge). The experiment should use the more conservative current benchmark rule and record the discrepancy.
- Historical validation cannot establish current live reputation or payout performance; only resolved forward submissions can do that.

## Sources and search coverage

Queries covered the official Numerai docs, `numerai/example-scripts`, `numerai-tools`, staff-linked forum material, arXiv, PMLR, JMLR, publisher pages, Crossref, and primary finance journals. Searches targeted temporal/walk-forward validation, purging/embargo, nested HPO, panel stock-return prediction, optimizer benchmarking, spectral bias, and frequency-domain gradient filtering. The bibliography fragment contains only sources used above.
