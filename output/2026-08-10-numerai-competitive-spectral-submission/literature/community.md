# Numerai community evidence: offline/live comparability, validation, MLPs, neutralization, and benchmarks

**Scope.** This report executes Group 3 of the search plan. It covers Numerai Tournament forum material, prioritising dated posts with code, numerical results, forward updates, or comments by Numerai staff. It does not use Signals or Crypto evidence. Sources were retrieved on 2026-08-10.

## Evidence labels

- **Staff-supported platform evidence**: a dated statement by a Numerai staff account on the Numerai forum. This is useful for platform intent or a disclosed internal result, but current contracts should still be checked against current official docs and code.
- **Supported community evidence**: a forum report with code, a stated protocol, quantitative results, a paired comparison, or a later live update. It remains observational unless independently replicated.
- **Anecdote**: an individual account, opinion, screenshot, or incompletely specified experiment. It is hypothesis-generating, not a reliable effect estimate.

## Searches executed

The five planned searches were run with the `forum.numer.ai` restriction: (1) validation/Diagnostics versus live CORR, MMC, FNC and leaderboard results; (2) era splits, holdouts, HPO and validation overfit; (3) AdamW, MLP and neural-network practice; (4) feature neutralization, exposure, benchmark and meta-model predictions; and (5) unstaked live submissions, reputation and payout targets. Follow-up searches used the forum's Discourse search endpoint for `AdamW`, `MLP`, `neural network`, `PyTorch`, `validation overfitting`, `holdout`, and `live validation`. The exact `AdamW` search returned zero forum posts. The broader searches produced dozens of candidates; 19 sources were retained for evidential value.

## Main conclusions

1. **Offline Diagnostics are useful model-development measurements, but they are not a demonstrated proxy for live competitiveness.** The best quantitative community evidence is mixed and heavily confounded. One 2021 analysis found modest rank associations between several offline diagnostics and a family of the author's live models, while a 2020 comparison found that validation matched live for one model and failed badly for another. A 2026 walk-forward ensemble with respectable benchmark-relative offline scores opened with poor live scores, although those live scores were explicitly interim. These reports support forward submission as the decisive test; they do not establish a stable mapping from Diagnostics to live rank.
2. **Repeated use of the fixed validation period is recognised as a major overfit channel.** The strongest forum guidance is a 2024 staff recommendation to generate walk-forward out-of-sample predictions, use an era embargo, and limit iterations. Training on the labelled validation set makes its Diagnostics non-generalisation evidence. Using a different model for feature selection does not fix selection on the same validation outcomes.
3. **Community MLP practice supports a serious MLP baseline, but not AdamW specifically.** Reproducible posts use shallow MLPs, dropout, batch normalisation, large or era-contained batches, time-series CV, early stopping, multi-target learning, Optuna/KerasTuner, and ensembling. The forum contains no located AdamW-specific Tournament evidence. Therefore AdamW should be treated as an implementation choice to test fairly, not a community-established competitive default.
4. **Neutralization has a real, model-dependent trade-off, not a monotone benefit.** Offline examples show sharply reduced feature exposure and sometimes higher era Sharpe at a cost in mean CORR. A 13-round live comparison reported progressively worse CORR and MMC with stronger neutralization. A 2024 bootstrap experiment characterised the effect as increased bias and reduced variance, with its apparent optimum explicitly model-specific.
5. **Benchmarks are comparators and ensemble ingredients, not ceilings.** Staff posts describe 20,000-tree benchmark models and publish their predictions to accelerate research and support benchmark-relative experiments. Version, target, transformation, and neutralization choices matter. Good validation performance relative to a benchmark remains offline evidence; live leaderboard reputation is calculated from resolved live rounds.

## Detailed findings

### 1. Offline Diagnostics versus live performance

#### Quantitative but confounded rank comparison

In [Does Good Model Diagnostics Correlate with Tournament Performance?](https://forum.numer.ai/t/does-good-model-diagnostics-correlate-with-tournament-performance/1454), **nasdaqjockey** (2021-01-12, with updates through 2021-02-06) compared diagnostic ranks with live-performance ranks across a related set of models. Reported rank correlations included validation mean 0.232, feature-neutral mean 0.182, CORR+MMC Sharpe 0.257, MMC mean 0.204, correlation with example predictions -0.054, and a custom cross-era consistency score 0.457. A second participant supplied code to download submission histories and reported qualitatively similar associations.

- **Label: Supported community evidence, low-to-moderate strength.** There are numerical tables and code, but the sample is a selected family of models from one participant, model construction and staking are not independent, multiple metrics were inspected, and no later holdout replication is reported.
- **Implication:** mean/consistency/MMC-oriented diagnostics are reasonable screening statistics, but the post does not justify converting an offline score into an expected live rank or claiming competitiveness.

#### Direct examples of match and mismatch

In [Stories of Validation](https://forum.numer.ai/t/stories-of-validation/70), **bor1** (2020-03-25) compared 51 held-out eras with 18 weeks of live results for two long-running models. The live and validation distributions aligned closely for one model (`badtimes`) but approximately half of the live eras for another (`Thirteen`) were worse than anything observed in its validation set. The author also notes that additional random holdout runs performed worse than the first selected run.

- **Label: Supported anecdote.** The duration, number of holdout eras, and counterexample are reported, but only two idiosyncratic models are studied and the holdout-selection scheme is bespoke.
- **Implication:** comparability can be model-dependent. A successful validation/live match for one model does not calibrate another model, and selection among holdout runs can itself favour a lucky split.

#### Recent benchmark-relative walk-forward result with an unfavourable first live update

In [Vibesciencing my way through v5.2 data (Faith II)](https://forum.numer.ai/t/vibesciencing-my-way-through-v5-2-data-faith-ii/8214), **degerhan** reported on 2026-02-04 a three-step, leakage-avoiding walk-forward procedure using 480 training and 160 validation eras. The ensemble obtained offline Numerai CORR 0.0171 and correlation contribution of 0.0073 versus the `ender_20` benchmark; it combined small-feature CatBoost models, benchmark-residual models, an MLP ranking model, sparse ensemble weights, and mild feature neutralization. Five ensembles were then deployed from round 1191; the first scores were poor.

- **Label: Anecdote with a useful forward update.** The protocol and scores are concrete, but code and complete fold results are not in the post, only one author is involved, and the first live scores were unresolved/interim.
- **Implication:** even a comparatively careful walk-forward result does not license a live-performance claim. It also provides a contemporary example of an MLP being used for diversity within an ensemble rather than as the sole CORR anchor.

#### Live reputation is intrinsically forward-looking

In [1 Year Reputation](https://forum.numer.ai/t/1-year-reputation/6195), staff member **ark** (2023-03-03) described reputation/rank as model scores intended to measure long-run performance and rank the leaderboard, and changed them to an unweighted average of resolved live rounds in the preceding year, with missing scores filled by zero. This is a historical platform definition and must be checked against current official documentation before implementation.

- **Label: Staff-supported platform evidence (historical).** 
- **Implication:** a Diagnostics-compatible offline evaluation cannot itself establish live reputation. Only time-forward submissions that later resolve enter this kind of leaderboard evidence.

### 2. Validation overfit and temporal model selection

#### Best concrete forum protocol: rolling walk-forward with embargo

In [ShatteredX's Improved & Compact Feature Set](https://forum.numer.ai/t/shatteredxs-improved-compact-feature-set-225-features-for-v4-3-midnight-data/6982), **master_key** (Numerai staff, 2024-01-22/23) warned that choosing features because they work over most of the fixed validation period guarantees an attractive validation result, even if a different model is used for selection. The proposed check was to reselect features using data only through era X, train through X, predict eras X+5 to X+50 (a five-era embargo), repeat forward, and compare the resulting out-of-sample predictions with an all-feature model. The post explicitly advised iterating only a couple of times to avoid overfitting the walk-forward result itself.

- **Label: Staff-supported methodological guidance.** It is a concrete, falsifiable protocol, though the post does not present a completed experiment proving that five eras is universally optimal.
- **Implication:** use chronological folds and fit every adaptive component—feature selection, HPO, neutralization strength and ensemble weights—inside each fold. Preserve a final untouched chronological block after all such choices.

#### Training on validation destroys the interpretation of Diagnostics

In [Overfitting to Validation Data](https://forum.numer.ai/t/overfitting-to-validation-data/3442), **sirbradflies** reported a 20-round paired experiment with four models trained only on train and four corresponding models retrained on train+validation. The aggregate comparison found no critical reason to train on validation, and the author reverted to validation-only use; the four model types included CatBoost, sklearn MLP, ridge, and Keras. Separately, [Do really model diagnostics makes sense?](https://forum.numer.ai/t/do-really-model-diagnostics-makes-sense/3842) records the straightforward but important point that Diagnostics on a validation period used for training are optimistic and no longer measure generalisation.

- **Label: Supported community evidence for the paired 20-round comparison; methodological fact for in-sample Diagnostics.** The live comparison is small and aggregates heterogeneous models, so it is not proof that adding recent labelled data never helps.
- **Implication:** do not add the final validation/holdout eras to training before the AdamW-versus-spectral decision. If a final production model is later refit on all labelled data, its old Diagnostics must not be presented as out-of-sample evidence.

#### Attractive validation-only neural-network results are not forward evidence

In [NN architecture for &gt;0.03 CORR on validation set](https://forum.numer.ai/t/nn-architecture-for-0-03-corr-on-validation-set/3145), **nyuton** (2021-05-01 onward) reported very high validation CORR, used early stopping against that same validation set, acknowledged possible large overfit, and initially had no forward results. Other participants explicitly asked whether the result would survive production.

- **Label: Anecdote / cautionary example.** Screenshots and implementation discussion exist, but no clean untouched test or later live result is supplied.
- **Implication:** a high score obtained after architecture, features, epochs, or seeds have repeatedly been chosen against one validation period is selected performance, not an unbiased estimate.

### 3. MLP and optimizer practice

The forum search found no post containing `AdamW`. The nearest reproducible Tournament evidence is:

- [Numerai Tournament Example code using Pytorch NN and Optuna](https://forum.numer.ai/t/numerai-tournament-example-code-using-pytorch-nn-and-optuna/4639), **meaten12121**, 2021-12-17 (updated 2022-04-25): linked code for time-series CV, Optuna HPO, era-boosted training and era-contained batches. Discussion recommends either whole-era batches or minibatches containing only one era. **Label: Supported community practice (code-linked), not a competitive benchmark.**
- [MLP hyperparameter tuning starter](https://forum.numer.ai/t/mlp-hyperparameter-tuning-starter/1496), **katsu1110**, 2021-01-20: a KerasTuner starter; the author explicitly notes NN performance is hyperparameter-sensitive. **Label: Reproducible tutorial, not live evidence.**
- [L2 regularization in MLPs and noisy domains](https://forum.numer.ai/t/l2-regularization-in-mlps-and-noisy-domains/3848), **neosbrother**, 2021-07-26: reports collapse to nearly constant predictions under stronger L2, especially in deeper networks, with larger batches around 4,000 working better for that setup. **Label: Anecdote with concrete failure symptoms; not replicated.**
- [AutoEncoder and multitask MLP on new dataset](https://forum.numer.ai/t/autoencoder-and-multitask-mlp-on-new-dataset-from-kaggle-jane-street/4338), **jrai**, 2021-10-15: community implementations use a denoising autoencoder, multi-target MLP, noise, dropout, batch normalisation and careful fold-local supervised representation learning. Replies include both poor replications and warnings that future data in an autoencoder can leak into historical evaluation. **Label: Code-rich but mixed community evidence.**
- The 2026 `Faith II` report above uses an MLP ranking model as a diversity/orthogonality component, while boosted trees supply CORR anchors. **Label: Contemporary anecdote.**

**Bottom line for the planned baseline:** an MLP is credible and field-tested, but the community record does not identify AdamW as superior to Adam, SGD or another optimizer. AdamW, learning rate, weight decay, batch construction, depth, width, dropout, normalisation and early stopping should therefore receive a predeclared search space and budget. The spectral method should receive the same budget and access to the same chronological folds. An optimizer-specific claim requires this experiment, not citation to community convention.

### 4. Neutralization and exposure control

#### Offline effect on exposure, CORR and era Sharpe

In [Model Diagnostics: Feature Exposure](https://forum.numer.ai/t/model-diagnostics-feature-exposure/899), **jrb** (2020-09-03) applied full feature neutralization to example-model validation predictions. Reported feature exposure fell from 0.0850 to 0.0061 and max feature exposure from 0.2955 to 0.0153; mean validation CORR fell from 0.0291 to 0.0255 while validation Sharpe rose from 0.9608 to 1.2436. Code and partial/top-exposure examples were supplied.

- **Label: Supported community evidence, single-model offline experiment.** It directly demonstrates the mechanical trade-off for one historical dataset/model, not a universal live benefit.

#### Bias–variance experiment

In [Feature Neutralization Increases Bias and Reduces Variance](https://forum.numer.ai/t/feature-neutralization-increases-bias-and-reduces-variance/7486), **by256** (2024-06-04) repeatedly fit models to bootstrapped era samples and evaluated on validation while varying neutralization proportion. Bias rose and variance fell with stronger neutralization; approximately 0.5 balanced them for the tested model. The post links a notebook and notes numerical concerns with explicitly forming an inverse in the common projection formula.

- **Label: Supported community evidence, offline and model-specific.** The method is clearer than an impressionistic leaderboard report, but the selected validation period and model delimit external validity.

#### Small live dose-response experiment

In [Liz Experiment Review Q1 2021](https://forum.numer.ai/t/liz-experiment-review-q1-2021-generating-features-and-applying-feature-neutralization/3011), **liz** (2021-04-22) held the underlying model fixed and submitted five neutralization levels (0%, 25%, 50%, 75%, 100%) over rounds 244–256. Across these 13 rounds, CORR and MMC generally worsened as neutralization increased. The author corrected an earlier concern about whether neutralization had been applied properly.

- **Label: Supported community evidence, small live paired experiment.** The same rounds and base model improve comparability, but 13 rounds are too few for a general optimum and the scoring/data regime is historical.

#### Optimising FNC directly and metric caveats

In [Optimizing for FNC and TB scores](https://forum.numer.ai/t/optimizing-for-fnc-and-tb-scores/5132), staff member **mdo** (2022-03-22) supplied PyTorch code for a shallow MLP with dropout, differentiable feature neutralization, CORR/FNC/TB objectives and exposure penalties. The post says TB500 was more stable to optimise than TB200, which overfit more easily in their work. In a later clarification, mdo stated that predictions were not feature-neutralized before then-current True Contribution calculations, only Gaussian-rank transformed.

- **Label: Staff-supported implementation evidence (historical scoring regime).** It demonstrates feasibility, not that direct FNC optimisation improves current live performance. Current metric code must supersede the 2022 clarification.

**Practical conclusion:** neutralization proportion and feature set are model-selection parameters. Evaluate raw and neutralized predictions on exactly the same out-of-sample eras; report CORR, MMC/benchmark-relative contribution, FNC/exposure, era volatility, and drawdown together. Do not select a neutralization level on the final holdout.

### 5. Benchmarks and competitive model practice

In [Benchmark Models](https://forum.numer.ai/t/benchmark-models/6754), staff member **master_key** (2023-10-28) explained that Numerai publishes predictions from models trained for new datasets and targets so users can compare modelling choices, allocate stakes, and build ensembles. The thread confirms that the benchmark models at that time used 20,000 trees and describes per-era Gaussianisation, standardisation, weighted ensembling, re-Gaussianisation and optional neutralization. [Super Massive Data: Sunshine](https://forum.numer.ai/t/super-massive-data-sunshine/5977), also by master_key (2022-12-27), described an internal example that ensembled six target models and used partial feature neutralization; Numerai called it one of its best internal models at the time and released delayed meta-model predictions for contribution research.

- **Label: Staff-supported benchmark provenance, historical versions.** These posts explain intended use and model construction but do not make any benchmark a fixed upper bound.
- **Implication:** compare the AdamW and spectral models against version-matched benchmark predictions on identical rows and eras. Benchmark-relative residual/contribution modelling is a credible competitive practice, but tuning ensemble weights or residual targets against the final holdout would contaminate the comparison.

The 2024 [V5 Atlas Data Release](https://forum.numer.ai/t/v5-atlas-data-release/7576) illustrates why version matching is essential: staff reported that retraining the same tutorial model on v5 raised validation mean CORR from 0.0245 to 0.0293 and Sharpe from 1.1379 to 1.335, while old v4.3 models suffered a steep internal performance decline when applied to the changed v5 universe. These are disclosed internal validation results, not live proof, but they show that apparent model quality can move materially with the data contract alone.

- **Label: Staff-supported internal offline evidence.**
- **Implication:** freeze dataset version, target, benchmark artifact and scoring implementation for the comparison. Do not compare scores across dataset regimes as though only the optimizer changed.

## Evidence-to-design recommendations

For a defensible AdamW-versus-spectral experiment:

1. Use era-ordered walk-forward folds with a documented embargo. Fit preprocessing, feature selection, neutralization choice, HPO and ensemble weights only on each fold's past.
2. Give both methods the same predeclared number of trials, seeds, epochs/early-stopping rules and compute allowance. Keep architecture and data constant unless architecture is explicitly part of the search budget.
3. Lock a final chronological holdout before inspecting either method. Report every predeclared metric and the distribution across eras/seeds, not only the winning aggregate.
4. Include a version-matched official benchmark as a comparator. Treat benchmark-relative contribution as an offline diagnostic, not a forecast of leaderboard rank.
5. Evaluate neutralization as a paired post-processing ablation. Expect lower exposure and potentially lower variance, but allow CORR/MMC to fall.
6. After the offline decision, use parallel unstaked live submissions for both frozen methods, if and only if upload authorisation is later given. Wait for sufficient resolved rounds before making a competitive or reputation claim.

## Gaps and negative findings

- No Numerai Forum post mentioning **AdamW** was found. There is no forum basis for calling it the established competitive optimizer.
- No controlled community study was found that compares AdamW with another optimizer on the current Tournament dataset under equal HPO budgets.
- No controlled community study was found that maps a given Diagnostics improvement to expected current live leaderboard rank.
- MLP reports are mostly from 2021–2022; the most recent retained MLP account (2026) uses it as one ensemble component and has only an unfavourable interim live update.
- Neutralization studies are model- and regime-specific. The strongest live dose-response report has only 13 historical rounds.
- Forum statements about scoring, payout, reputation, datasets and benchmark files age quickly. Current official documentation and source code must govern implementation.
