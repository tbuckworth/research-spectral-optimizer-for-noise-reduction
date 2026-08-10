# Steinhardt Decomposition

## Objective and boundary

The objective is to produce, without using the sealed official validation for development, (1) a genuinely tuned AdamW MLP, (2) an equal-budget test of the existing streaming gradient-covariance eigenspace filter, (3) a frozen offline candidate evaluated once against the version-matched Numerai benchmark, and (4) a submission-ready but not uploaded or staked artifact. The scientific question and the competitive question are separate: spectral can fail while AdamW still yields a useful artifact, and strong historical Diagnostics-compatible results do not establish live leaderboard reputation.

The development universe is the 574 v5.3 training eras. The resolved official-validation eras remain sealed until the optimizer configurations, target tracks, post-processing recipes, seeds, metric code, bootstrap code, and decision rules are frozen. No validation prediction, aggregate, plot, or intermediate diagnostic may be inspected before that gate. The 20D and 60D tracks use 8-era and 16-era purges respectively and are never silently pooled.

All probabilities below are evidence-calibrated point estimates for passing the stated quick test, not posterior probabilities of a leaderboard rank. `T` is elapsed turnaround for a minimum discriminating test with one desktop RTX 3090 or one free MATS L40 allocation; queue delay is excluded. Lambda is `-ln(P_success)/T`. Rows are sorted by lambda, but dependency gates determine executable order.

## Challenge revisions (governing)

The independent challenge pass found major but repairable design defects. The following rules supersede any conflicting wording below:

1. The primary endpoint is standalone exact CORR on `target_cyrusd_20`, spectral minus AdamW. Other targets, BMC, neutralization and blends are secondary.
2. AdamW and spectral are compared as complete procedures. Each gets 40 completed configurations under the same multi-fidelity schedule and the same shared architecture/training search envelope. Spectral also samples filter parameters inside—not in addition to—those 40 configurations.
3. Runtime is irrelevant to scientific fairness. Match configurations, examples/updates, folds and seeds. Rerun infrastructure failures; retain algorithmic/numerical failures.
4. Spectral smoke/pilot runs test integrity only and cannot stop its matched HPO for lack of efficacy.
5. Selection performance is estimated with nested outer walk-forward folds inside the 574 training eras. Conditional bootstrap intervals over HPO winners are not treated as selection-adjusted inference.
6. The old-recipe +0.001 gate and 95%-of-benchmark gate are removed. A defensibly tuned AdamW remains a valid control regardless of improvement over the old incomparable recipe.
7. Historical promotion yields an “offline submission candidate,” not a competitive model. Competitive claims require resolved forward rounds.

## Lambda table

| # | Component | P_success | Evidence calibration | T (h) | lambda | Explicit quick-test threshold | Dependencies | Status |
|---:|---|---:|---|---:|---:|---|---|---|
| 1 | Seal, provenance, and contract guard | 0.75 | Files and hashes already exist, but the 8.9 GB Parquet contract, target aliases, and prior provenance failure make integration non-trivial | 1 | 0.288 | PASS only if all five SHA-256 hashes match, era/row counts and target aliases match `data-manifest.md`, validation access is blocked/audited, and no final-validation-derived artifact exists | None | PENDING |
| 2 | Exact metric and row-alignment parity | 0.60 | Official scoring source is pinned; tied ranks and benchmark alignment are known prior failure channels | 2 | 0.255 | PASS if local CORR/BMC tests match pinned `numerai_tools` to `abs(error) <= 1e-12` on synthetic tied data and sampled real train rows, and shuffled/missing/duplicate IDs are rejected | #1 | PENDING |
| 3 | Memory/numerical feasibility for the MLP and filter | 0.50 | A 7.49M-parameter MLP ran previously, but v5.3 is larger and the streaming eigenspace implementation adds substantial state | 3 | 0.231 | PASS if a 2-era train/eval smoke run completes without OOM, finite loss/gradients, and peak GPU memory `<=22 GB` on RTX 3090 or `<=44 GB` on L40. Record throughput but do not gate on it | #1 | PENDING |
| 4 | AdamW learns non-collapsed train-only signal | 0.45 | MLPs are field-tested on Numerai, but no current AdamW-specific competitive evidence was found and weak-signal collapse is documented in community reports | 5 | 0.160 | On one early train-only walk-forward fold, PASS if three seeds all have prediction SD `>=0.02`, no NaN/Inf, and median era CORR exceeds both zero and the original-study AdamW recipe by `>=0.001` | #2, #3 | PENDING |
| 5 | Leakage-resistant walk-forward harness | 0.55 | Nested temporal selection is well supported, while target overlap, fold-local preprocessing, and repeated selection are substantial implementation risks | 4 | 0.149 | PASS if three predeclared expanding train-only folds reproduce exactly, preserve whole eras, enforce 8/16-era purges, fit every adaptive transform on fold-train only, and fail deliberate future-era leakage tests | #1, #2 | PENDING |
| 6 | Competitive AdamW HPO on train eras | 0.45 | Similar MLP HPO exists, but optimizer rankings reverse with search protocol and no evidence establishes AdamW as competitive on v5.3 | 6 | 0.133 | Multi-fidelity pilot PASS if at least 8/10 trials complete, successive-halving ranks have Spearman `rho >= 0.5` between low/high fidelity, and the promoted recipe improves robust mean walk-forward CORR by `>=0.001` over the original recipe with no worse median BMC by more than `0.001` | #4, #5 | PENDING |
| 7 | Mechanism-accurate spectral implementation and controls | 0.40 | The optimizer exists and has prior successes, but repository provenance distinguishes two materially different mechanisms and current Numerai integration is untested | 8 | 0.115 | PASS if commit/config are pinned; strength-zero matches AdamW parameter updates to relative error `<=1e-6` for 100 steps; retained basis is orthonormal to `1e-5`; restart is deterministic; and rank-matched random-subspace control runs | #3, #4 | PENDING |
| 8 | Spectral filter shows enough train-fold signal to justify full budget | 0.35 | Prior optimizer evidence is mixed and explicitly predicts failure when useful signal is weak or distributed; no direct Numerai evidence exists | 10 | 0.105 | On paired fold/batch/seed streams, PASS if at least 2 of 3 predeclared spectral settings have mean paired CORR delta `>0`, the best has delta `>=0.0005`, and neither BMC nor max drawdown regresses by more than `0.001` or 10% respectively; random subspace must not match it within `0.0002` | #5, #7 | PENDING |
| 9 | Equal-evidence spectral-vs-AdamW selection is robust | 0.50 | Fair-search methodology is strong, but no direct result predicts which procedure wins | 12 | 0.058 | PASS if completed configurations, folds, seeds, training examples and update budgets match within `<=10%`, all failures are retained, and the selected spectral procedure has positive paired mean CORR with a train-fold moving-block-bootstrap 80% interval lower bound `>0` and no material BMC/drawdown regression. Wall-clock cost is recorded but not matched | #6, #8 | PENDING |
| 10 | Frozen post-processing/blend yields a submission-worthy route | 0.55 | Benchmarks are intended as ensemble ingredients and positive contribution can matter, but neutralization effects are model-specific and may reduce CORR/MMC | 14 | 0.043 | Using train-only OOF predictions, PASS if a predeclared raw, neutralized, or Ender blend either reaches `>=95%` of matched benchmark mean CORR or has mean BMC `>=0.002` with 80% block-bootstrap lower bound `>0`, while Sharpe is positive and max drawdown is no worse than benchmark by `>20%` | #6, #9 | PENDING |
| 11 | Artifact fits Numerai Model Upload runtime | 0.70 | Official interface is explicit; CPU-only 4 GB/10-minute limits make a large MLP or bundled preprocessing a real packaging risk | 12 | 0.030 | In a clean Python 3.10 CPU container capped at 4 GB, `predict(live_features, live_benchmark_models)` returns valid, finite, uniquely indexed predictions in `[0,1]` in `<8 min`, peak RSS `<3.5 GB`, with deterministic checksum and no network access | #6, #7; can start before #10 | PENDING |
| 12 | Conventional live prediction path is schema-safe | 0.80 | Standard batch inference is easier than Model Upload, but live IDs/features/benchmark joins can still drift | 8 | 0.028 | PASS if a dry-run against a current-format, target-free fixture emits exactly one finite `[0,1]` prediction per ID, rejects feature/version mismatch, records hashes/config, and performs no API upload call | #1, #10 | PENDING |
| 13 | One-time official-validation reveal yields an offline candidate | 0.65 | Official benchmarks are strong (Ender-20 CORR about 0.037--0.040; Ender-60 about 0.053), while recent community walk-forward success has not reliably transferred live | 16 | 0.027 | After freeze, call spectral improved only if paired mean CORR delta vs AdamW is `>0` and the 95% era-block-bootstrap lower bound is `>0`, with BMC regression `<0.001` and drawdown regression `<10%`. Call a candidate submission-worthy only if standalone mean CORR is `>=95%` of matched Ender or mean BMC is `>=0.002` with 95% lower bound `>0` and the frozen blend improves benchmark CORR with 95% lower bound `>0` | #9, #10 and signed freeze manifest | PENDING, SEALED |
| 14 | Submission-ready release bundle | 0.75 | Reproducibility requirements are conventional, but model size, environment locking, dual inference paths, and no-upload safety broaden the bundle | 16 | 0.018 | PASS if a clean replay reproduces the frozen validation table to `1e-10`, artifact checksums match, model-upload and conventional paths agree within `1e-7` on a fixture, README names the scoring targets and claim boundary, and an upload requires an explicit separate command/credential step | #11, #12, #13 | PENDING |

No component is below the 0.05 showstopper threshold. The lowest substantive probabilities are the empirical claims that AdamW is competitive and spectral adds signal; neither is known impossible, and both can fail while preserving a useful negative result.

## Component details and stop rules

### 1–3: Integrity and feasibility gates

These establish that later scores mean what they claim and that the chosen model family is runnable. The seal should be mechanical: development commands accept only a train-data root; validation access is enabled only by a separate reveal command that requires a signed freeze manifest. Logs should record every path opened. The metric suite must include tie-heavy predictions, constant predictions, shuffled IDs, missing rows, duplicate IDs, era mismatch, and benchmark-version mismatch.

For feasibility, start with the `small` feature set, one modest MLP, memory-mapped/streamed era blocks, and the exact current filter implementation. If component 3 fails, first reduce rank/update frequency and activation footprint while preserving the mechanism. Runtime overhead is measured but is not a failure condition: the user explicitly does not care if spectral training is much slower. Do not buy elastic compute without separate approval.

### 4–6: Establish AdamW before testing the headline method

Component 4 is a falsification test for the basic premise that this MLP family can extract train-era signal. Its old-recipe improvement threshold is removed; pass requires finite, non-collapsed predictions and positive train-only outer-fold signal after basic debugging.

Component 6 should use a predeclared multi-fidelity scheduler on train eras only. A reasonable full search envelope is 30--40 AdamW configurations: 20% of updates on all three folds, 40--50% promoted to medium fidelity, and 4--6 finalists at full fidelity with three paired seeds. Search learning rate, weight decay, batch construction, schedule/warm-up, clipping, early stopping, width/depth, dropout, and normalization within one fixed MLP family. Aggregate eras equally, require every promoted configuration to complete all folds, and retain failed trials. If low-fidelity ranks are unstable (`rho < 0.5`), spend the remaining budget on fewer full-fidelity configurations rather than trusting successive halving.

AdamW baseline success means a defensible selected baseline, not necessarily benchmark parity. If it fails to beat the original recipe by 0.001 robust mean CORR, stop the causal spectral comparison and report that the prerequisite competitive control was not established. A weaker model may still be packaged only as an engineering demonstration, not a competitive candidate.

### 7–9: Test spectral mechanism, then the complete procedure

Pin the optimizer repository commit and document that this is the streaming parameter-space gradient-covariance eigenspace method, not the older per-sample Gram method and not FFT filtering. The mandatory controls are tuned AdamW, filter-strength zero, and a rank-matched random subspace. A shuffled-history or reset-frequency control is desirable if it fits the equal budget.

Component 8 is an integrity and mechanism-diagnostics pilot only. It cannot stop spectral for a null/negative efficacy estimate. Passing numerical, equivalence and control checks unlocks a 40-configuration spectral search symmetric with AdamW. Spectral-only rank, EMA/forgetting, projection/blend strength, warm-up and update cadence are sampled inside that budget; shared architecture/training parameters use the same distributions as AdamW. Slower spectral trials may use more accelerator time to complete the same updates and examples.

The 80% interval in component 9 is a promotion rule on development folds, not evidence for the final claim. The final 95% interval is computed only after the seal is opened. If spectral fails component 9, freeze AdamW as the optimizer candidate and retain the spectral result as a negative domain-transfer finding.

### 10: Select post-processing without contaminating the holdout

Construct raw, predeclared neutralization levels, and frozen blends with the matching `v53_lgbm_ender20` or `v53_lgbm_ender60` train benchmark using OOF predictions only. Target tracks are reported independently:

- primary 20D development tracks: `target_cyrusd_20`, `target_ender_20`, and a predeclared target ensemble;
- separate 60D Diagnostics/CORJ60 track: `target_ender_60` with a 16-era purge.

The unavailable live `target_cyrus_20` is never imputed or relabelled. If no standalone or blend route passes component 10, the result can still answer the optimizer question but is not promoted as submission-worthy.

### 11–12: Build both inference paths before the reveal

Packaging can proceed with placeholder/frozen train-only checkpoints. The Model Upload path must obey the documented Python 3.10--3.13, one-CPU, 4 GB, ten-minute contract. If a direct MLP cannot fit, acceptable predeclared remedies are reduced precision, a smaller frozen model, or conventional weekly batch inference. Distillation into a materially different model is a new adaptive component and cannot be chosen after seeing official validation.

The conventional path should download nothing in tests and must make upload impossible by default. It should produce a local prediction CSV plus manifest only. Credentialed submission, model creation, upload, staking, or API mutation is explicitly outside this decomposition without later authorization.

### 13: Single reveal and decision tree

Before validation access, write an immutable freeze manifest containing data/package/optimizer commits, complete trial ledger, selected AdamW and spectral configs, seeds, target tracks, benchmark columns, blend and neutralization coefficients, bootstrap block rule, all thresholds above, and the exact analysis command. Then score the resolved official-validation eras once.

Decision outcomes are exhaustive:

1. **Spectral improvement and submission-worthy spectral/blend:** package that route and AdamW comparator.
2. **Spectral tie/failure but submission-worthy AdamW/blend:** package AdamW; report the spectral null/negative result.
3. **Neither route submission-worthy:** produce reproducible research artifacts, but label the live artifact noncompetitive and do not recommend upload.
4. **Integrity or schema failure during reveal:** invalidate the reveal, repair only the mechanical error under an audit trail, and do not tune any model or threshold using the observed scores.

Historical results must always print the exact released target beside each number. They may be called v5.3 Diagnostics-compatible, not live CORR20v2/MMC reputation. The latter remains unknowable until authorized forward submissions resolve.

### 14: Submission-ready artifact

The bundle contains the frozen checkpoint(s), preprocessing and feature list, target/benchmark metadata, inference code, model-upload callable where feasible, conventional prediction command, environment lock, checksums, smoke fixtures, raw per-era official-validation predictions/metrics, trial ledger, and a short model card. A no-op/dry-run is the default. Upload and stake operations are absent or separately gated.

## Dependency graph

```text
#1 contract/seal
 ├──>#2 metric parity ──>#5 walk-forward ──>#6 AdamW HPO ──┐
 ├──>#3 feasibility ──>#4 AdamW sanity ────────┘             │
 │                     └──>#7 spectral integrity ──>#8 pilot ──>#9 equal-budget comparison
 │                                                                    │
 │                                      #6 ───────────────────────────┤
 │                                                                    v
 │                                                              #10 blend/freeze
 │                                      #6/#7 ──>#11 upload runtime    │
 └──────────────────────────────────────────────>#12 live schema path  │
                                                                       v
                                                        signed freeze -> #13 reveal
                                                        #11/#12/#13 ──>#14 bundle
```

The graph has two terminal scientific branches: spectral failure routes to the tuned AdamW candidate; lack of offline competitiveness routes to a negative result rather than forced live deployment.

## Parallelisation and compute plan

Use the desktop RTX 3090 for deterministic smoke tests, metric tests, profiling, final packaging, and one-off confirmation. Use free MATS `compute` L40 Slurm jobs for independent HPO trials; never run GPU work in an SSH shell. Paid `elastic-*` partitions are neither required nor authorized.

| Phase | RTX 3090 estimate | Free MATS estimate | Storage/RAM expectation |
|---|---:|---:|---|
| Integrity, metrics, folds (#1–5) | 4–8 wall h, 1–3 GPU h | Not required | Read 8.9 GB source Parquet; allow 50–150 GB for shards, OOF predictions, and caches |
| AdamW pilot and HPO (#6) | 120–220 GPU h serial | 30–60 L40 GPU h if 4–8 independent jobs are available; roughly 8–20 h elapsed excluding queue | 20–60 GB checkpoints/logs; stream features rather than materializing all 3,555 columns |
| Spectral integrity, pilot, HPO (#7–9) | 180–400 GPU h serial, dominated by filter overhead | 50–120 L40 GPU h; roughly 12–36 h elapsed if parallel capacity is available | Additional 30–100 GB; rank/update cadence must stay within component-3 memory limits |
| Blends, bootstrap, sealed evaluation (#10, #13) | 8–20 wall h, 2–8 GPU h | CPU jobs or one L40; 4–12 h elapsed | Raw predictions dominate; preserve all era-level arrays and bootstrap seeds |
| Packaging and replay (#11–14) | 12–24 wall h, mostly CPU | Not required | Final upload artifact must itself remain below runtime memory limits |

These are planning ranges, not measurements. Component 3 replaces them with observed throughput before HPO. Match selection evidence—completed configurations, folds, seeds, updates and examples—not accelerator-hours. Spectral may consume substantially more free accelerator time to obtain that matched evidence. Paid compute remains separately gated.

Parallel work that does not threaten the seal:

- after #1, metric tests (#2) and throughput harness preparation (#3) can run concurrently;
- after #5, independent AdamW trial/fold/seed jobs can run concurrently on MATS;
- after #7, spectral settings, random-subspace controls, and paired seeds can run concurrently, subject to equal accounting;
- upload-runtime packaging (#11) can start from train-only checkpoints while #8–10 run;
- conventional schema tests (#12) can run once the frozen post-processing interface is known;
- #13 is serial and one-time; #14 follows it.

## Overall probability and interpretation

Multiplying all point estimates would give about `0.00026`, but that number is not a credible project probability: the events are strongly correlated, several rows are engineering gates, and spectral success is not required for an AdamW artifact. More useful branch estimates are:

- probability of establishing a defensible tuned AdamW control: approximately **0.35–0.50**;
- conditional probability that spectral passes the final improvement gate given a valid AdamW control: approximately **0.25–0.40**;
- probability of an offline submission-worthy AdamW or spectral/blend candidate: approximately **0.20–0.35**;
- conditional probability of a submission-ready artifact given an offline candidate: approximately **0.70–0.85**.

The probability of a genuinely competitive live reputation is not estimated from historical scores because the literature supplies no reliable offline-to-live mapping and the payout target is unreleased. It remains a prospective empirical question after explicit upload authorization and enough rounds resolve.

## Showstoppers and hard stops

There are no prior-probability showstoppers below 0.05. The following observed failures are hard stops for the associated branch:

- seal violation or unexplained data/hash mismatch: stop all confirmatory claims;
- metric disagreement above `1e-12` or silent row misalignment: stop scoring;
- persistent infeasibility after bounded rank/update-frequency reduction: stop the large spectral model;
- AdamW sanity/HPO failure: stop claims about spectral improving a competitive baseline;
- spectral pilot or train-fold promotion failure: stop spectral HPO and proceed with AdamW only;
- no train-OOF submission-worthy route: do not characterize the artifact as competitive;
- official-validation failure: do not retune on it; report the result and await genuinely forward evidence;
- runtime/schema failure: retain a conventional local prediction path, but do not call the model-upload artifact ready.

## Evidence basis

Probability estimates and thresholds are anchored to the run's official contract audit, methodology report, community report, benchmark snapshot, novelty assessment, and success criteria. Strong evidence supports the data/scoring contract and nested temporal protocol. Moderate evidence supports MLP feasibility and benchmark blending. Direct evidence for competitive AdamW on current v5.3 and for this streaming covariance-eigenspace filter on financial panels is absent; those probabilities are therefore deliberately below 0.5. No FFT or parameter-frequency result is used as evidence for the filter under test.
