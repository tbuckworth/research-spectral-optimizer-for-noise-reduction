# Search Plan

## Topic Summary

Identify the current, authoritative Numerai Tournament evaluation and submission contract, then find reproducible modelling and temporal-validation practices needed to compare a carefully tuned AdamW MLP with a spectral-optimization variant under equal search and evaluation budgets. The search must separate Diagnostics-compatible offline evidence from performance that can only be established by forward, unstaked live submissions.

## Key Concepts

- **Numerai Tournament**: The flagship Numerai product; exclude Numerai Signals and Numerai Crypto unless a source is explicitly about a shared platform mechanism.
- **Current data contract**: Versioned tournament dataset, feature/target columns, era identifiers, feature metadata, and train/validation/live partitions used by the official pipeline.
- **Official scoring contract**: Numerai's current correlation, MMC, FNC, drawdown, Sharpe-like and payout/risk definitions, including any neutralization or exposure controls in official code.
- **Diagnostics-compatible offline evaluation**: Validation-era predictions scored through the same documented definitions and benchmark comparisons as Numerai Diagnostics; useful, but not evidence of live leaderboard reputation.
- **Forward live evaluation**: Unstaked, time-forward submissions measured after the target resolves; the only direct evidence for live leaderboard/reputation performance.
- **Temporal model selection**: Chronological train/validation splits, purging/embargo where appropriate, fixed final holdout, and safeguards against repeated holdout selection in panel financial ML.
- **AdamW baseline**: A 7.49M-parameter MLP tuned before spectral optimization is exposed to the final comparison; optimizer and training hyperparameters must have an equal, predeclared search budget.
- **Spectral optimization**: The candidate intervention, including spectral filtering or frequency-domain parameter/update manipulation intended to reduce overfitting; distinguish it from spectral regularization, spectral normalization, and Fourier features.
- **Benchmark predictions**: Official Numerai example/benchmark model predictions used as an offline comparator, not as proof that a new model will rank similarly live.

## Search Tasks

### Group 1: Official Numerai sources — docs, API, source, data, and leaderboard definitions

1. **Query**: `Numerai Tournament documentation dataset v5 validation live eras feature metadata targets`
   - Source targets: `docs.numer.ai`, `docs.numer.ai/tournament`, official Numerai GitHub repositories
   - Date range: 2024-01-01 to 2026-08-10; also retain the currently linked canonical documentation regardless of publication date
   - Max results: 15
   - Sort order: canonical/current documentation first, then recency
   - Rationale: Establish the current dataset schema and official meaning of train, validation, live, eras, targets, and metadata before designing splits or loaders.

2. **Query**: `site:docs.numer.ai Numerai API download dataset submission upload model id tournament`
   - Source targets: official docs and official API/client repositories only
   - Date range: current documentation plus releases since 2023-01-01
   - Max results: 15
   - Sort order: relevance
   - Rationale: Locate the current supported API workflow and the boundary between preparing an artifact and actually uploading/submitting it.

3. **Query**: `site:github.com/numerai Numerai metrics correlation mmc fnc sharpe max drawdown payout source code`
   - Source targets: `github.com/numerai`, linked official repositories, package release notes
   - Date range: 2023-01-01 to 2026-08-10, with git history for the active implementation
   - Max results: 25
   - Sort order: most recently updated, then exact symbol/function match
   - Rationale: Pin scoring and payout claims to implementation, including sign conventions, aggregation, neutralization/exposure behaviour, and version changes that prose docs may omit.

4. **Query**: `site:docs.numer.ai OR site:numer.ai Numerai leaderboard payout target diagnostics benchmark model predictions`
   - Source targets: official product/docs pages, official announcements, official benchmark prediction artifacts
   - Date range: 2023-01-01 to 2026-08-10
   - Max results: 20
   - Sort order: canonical definition first, then recency
   - Rationale: Define what the relevant leaderboard, payout target, Diagnostics display, and official benchmark predictions actually measure, and record which claims require forward live results.

5. **Query**: `site:github.com/numerai OR site:docs.numer.ai example predictions benchmark model parquet validation Numerai Tournament`
   - Source targets: official example repositories, dataset releases, model-prediction downloads, metadata endpoints
   - Date range: current active dataset/release, with prior versions only for compatibility interpretation
   - Max results: 20
   - Sort order: relevance to current dataset version
   - Rationale: Find exact benchmark-model artifacts, their generation assumptions, and version matching requirements for an apples-to-apples offline comparison.

### Group 2: Official Numerai examples/tutorials and academic financial-ML sources

#### Official Numerai examples and tutorials

1. **Query**: `site:docs.numer.ai OR site:github.com/numerai Numerai Tournament example model validation diagnostics Python`
   - Date range: 2023-01-01 to 2026-08-10; include maintained older tutorials if linked from current docs
   - Max results: 20
   - Sort order: most recently updated
   - Rationale: Recover supported baseline architectures, preprocessing, prediction formatting, diagnostics calls, and evaluation sequencing from official maintained examples.

2. **Query**: `site:forum.numer.ai official example model benchmark predictions feature neutralization validation`
   - Date range: 2023-01-01 to 2026-08-10
   - Max results: 15
   - Sort order: relevance, prioritizing posts by Numerai staff or links to official code
   - Rationale: Find explanatory material around official examples and identify version-specific caveats not yet reflected in tutorials.

#### arXiv API Queries

1. **Query**: `all:"financial machine learning" AND (all:"temporal validation" OR all:"walk-forward validation" OR all:"time series cross validation")`
   - Categories: `q-fin.ST, stat.ML, cs.LG, econ.EM`
   - Date range: 2023-01-01 to 2026-08-10; expand backward for highly cited seminal methodology
   - Max results: 35
   - Sort order: relevance, then submitted date
   - Rationale: Identify defensible time-ordered validation procedures for financial prediction, especially claims about leakage and selection bias.

2. **Query**: `all:("purged cross-validation" OR "combinatorial purged cross-validation" OR embargo) AND all:(finance OR financial OR trading)`
   - Categories: `q-fin.ST, stat.ML, cs.LG`
   - Date range: 2015-01-01 to 2026-08-10
   - Max results: 30
   - Sort order: relevance
   - Rationale: Target techniques intended to mitigate overlap, label leakage, and multiple-testing bias in financial temporal evaluation.

3. **Query**: `all:("hyperparameter optimization" OR "model selection") AND all:("time series" OR panel) AND all:(finance OR asset returns)`
   - Categories: `q-fin.ST, stat.ML, cs.LG, econ.EM`
   - Date range: 2020-01-01 to 2026-08-10
   - Max results: 35
   - Sort order: relevance, then submitted date
   - Rationale: Establish how to allocate a fixed HPO budget and maintain a truly untouched final holdout when comparing AdamW and spectral methods.

4. **Query**: `all:("spectral optimization" OR "frequency-domain optimization" OR "spectral filtering") AND all:(neural network OR deep learning OR optimizer)`
   - Categories: `cs.LG, stat.ML, cs.AI`
   - Date range: 2018-01-01 to 2026-08-10
   - Max results: 40
   - Sort order: relevance
   - Rationale: Map the intervention's closest technical antecedents and isolate methods that change optimization dynamics rather than merely using Fourier features or spectral normalization.

5. **Query**: `all:("spectral bias" OR "frequency principle") AND all:(generalization OR overfitting) AND all:(neural network OR MLP)`
   - Categories: `cs.LG, stat.ML, cs.AI`
   - Date range: 2017-01-01 to 2026-08-10
   - Max results: 35
   - Sort order: relevance
   - Rationale: Seek mechanisms and counterexamples for the hypothesis that spectral filtering suppresses overfitting in the MLP setting.

#### Semantic Scholar API Queries

1. **Query**: `financial machine learning temporal validation walk-forward cross-validation`
   - Fields: `title,abstract,year,authors,citationCount,url,externalIds,venue`
   - Date range: 2015-01-01 to 2026-08-10
   - Max results: 40
   - Sort order: citationCount for seminal work, then recency for updates
   - Rationale: Triangulate arXiv results with peer-reviewed and influential work on time-aware validation.

2. **Query**: `purged cross validation combinatorial purged cross validation financial machine learning`
   - Fields: `title,abstract,year,authors,citationCount,url,externalIds,venue`
   - Date range: 2015-01-01 to 2026-08-10
   - Max results: 25
   - Sort order: relevance
   - Rationale: Retrieve the primary methodology and independent examinations of its assumptions and limits.

3. **Query**: `hyperparameter optimization time series forecasting model selection nested validation`
   - Fields: `title,abstract,year,authors,citationCount,url,externalIds,venue`
   - Date range: 2018-01-01 to 2026-08-10
   - Max results: 40
   - Sort order: relevance, then citationCount
   - Rationale: Find evaluation-budget and nesting guidance applicable when one method is tuned first and another is compared later.

4. **Query**: `spectral optimization neural networks frequency domain optimizer generalization`
   - Fields: `title,abstract,year,authors,citationCount,url,externalIds,venue`
   - Date range: 2018-01-01 to 2026-08-10
   - Max results: 40
   - Sort order: relevance
   - Rationale: Gather primary papers, replications, and negative results relevant to the spectral candidate.

5. **Query**: `cross-sectional stock return prediction neural network panel data validation`
   - Fields: `title,abstract,year,authors,citationCount,url,externalIds,venue`
   - Date range: 2018-01-01 to 2026-08-10
   - Max results: 35
   - Sort order: relevance
   - Rationale: Capture differences between ordinary univariate time-series validation and Numerai's era-indexed, cross-sectional panel prediction task.

### Group 3: Numerai forum and high-quality community evidence

1. **Query**: `site:forum.numer.ai Tournament validation live performance leaderboard diagnostics correlation MMC FNC`
   - Date range: 2022-01-01 to 2026-08-10
   - Max results: 30
   - Sort order: relevance, then recency
   - Rationale: Find experienced accounts of how offline Diagnostics relate—and fail to relate—to later live performance and leaderboard standing.

2. **Query**: `site:forum.numer.ai validation era split holdout overfitting hyperparameter optimization tournament`
   - Date range: 2020-01-01 to 2026-08-10
   - Max results: 30
   - Sort order: relevance
   - Rationale: Collect concrete community practices and disagreements about temporal splits, holdout reuse, and validation overfitting.

3. **Query**: `site:forum.numer.ai AdamW MLP neural network Tournament model training`
   - Date range: 2021-01-01 to 2026-08-10
   - Max results: 25
   - Sort order: relevance, prioritizing posts with code, result tables, or post-hoc updates
   - Rationale: Locate field-tested baseline choices and pitfalls specific to tournament data without treating unreplicated reports as facts.

4. **Query**: `site:forum.numer.ai feature neutralization exposure control benchmark predictions meta model tournament`
   - Date range: 2020-01-01 to 2026-08-10
   - Max results: 30
   - Sort order: relevance
   - Rationale: Identify community explanations and empirical evidence concerning neutralization, exposure, benchmarks, and metric trade-offs.

5. **Query**: `site:forum.numer.ai live submission unstaked model reputation payout target current tournament`
   - Date range: 2023-01-01 to 2026-08-10
   - Max results: 20
   - Sort order: recency
   - Rationale: Surface practical, current guidance on creating a live model and interpreting live metrics while respecting the no-stake/no-upload authorization boundary.

## Evidence Handling Rules

- Prioritize official docs, APIs, released data metadata, and active source code for platform facts. Record version, URL, retrieval date, and commit/release identifier where available.
- Treat example models and benchmark predictions as reproducible comparison artifacts, not as an upper bound or a claim of live competitiveness.
- Treat forum posts as contextual and hypothesis-generating evidence. Prefer posts with code, dated validation/live results, subsequent corrections, or staff confirmation; record author role and date.
- For academic work, distinguish methods demonstrated on genuinely time-forward financial panels from generic random-split or synthetic experiments. Search citation chains for both supporting and critical/negative evidence.

## Expected Coverage

This plan should establish the exact current Numerai contract, a reproducible offline protocol, and credible implementation options for the two-method comparison. Remaining gaps are unavoidable without running the experiment: the current dataset's realized target outcomes, sensitivity to the chosen temporal holdout, and any claim of competitiveness or reputation on the live leaderboard, which requires time-forward submissions after explicit authorization.
