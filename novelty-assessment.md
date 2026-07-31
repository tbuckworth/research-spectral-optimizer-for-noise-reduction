# Novelty Assessment

## Proposed Research

Empirically test whether the existing Spectral Optimizer — a wrapper around Adam/AdamW that eigendecomposes the per-sample gradient similarity/covariance matrix and keeps only gradient directions many samples agree on, using a Marchenko-Pastur-style threshold — improves out-of-sample predictive performance versus tuned Adam/AdamW on large, noisy (low-SNR) financial time-series prediction, evaluated across an MLP and at least one recurrent/sequence architecture under matched tuning budgets.

## Verdict: NOVEL

No discovered work combines (a) per-sample gradient covariance/similarity eigendecomposition with RMT/Marchenko-Pastur thresholding as the optimizer update rule, with (b) evaluation on low-SNR financial prediction. Both halves of the combination were checked independently across three search channels (arXiv + Semantic Scholar, lab blogs, community forums), and the nearest neighbors differ in method, domain, or both. The verdict comes with two calibration caveats, detailed under Differentiation Analysis: novelty rests on absence of evidence, and the surrounding niche is active enough (2024–2026) that positioning against the nearest neighbor must be explicit.

## Closest Existing Work

### 1. Gradient Agreement Filtering (Chaubard, Eddy & Kochenderfer, 2024)
- **Overlap**: Related (closest of all findings)
- **What they did**: In distributed data-parallel image-classification training, filtered/reweighted the macrobatch update based on cross-microbatch gradient agreement (orthogonality/negative-correlation checks), finding that filtering disagreeing gradients reduces memorization and improves generalization.
- **How it relates**: Same underlying problem (noise-fitting gradient directions hurt generalization; suppress them via cross-sample agreement) and the same intervention level (the optimizer update). This is the paper the proposed work must be positioned against.
- **Key difference**: GAF uses simple pairwise orthogonality/correlation checks at microbatch granularity; the Spectral Optimizer eigendecomposes the full B×B per-sample similarity matrix and applies an RMT (Marchenko-Pastur) threshold — a principled spectral criterion rather than a heuristic pairwise one. GAF targets distributed image classification; the proposed work targets low-SNR financial regression, a regime GAF never touched and where the mechanism's transfer is genuinely uncertain.
- **Source**: https://arxiv.org/abs/2412.18052

### 2. Coherent Gradients (Chatterjee & Zielinski, 2020)
- **Overlap**: Related (theory, not method)
- **What they did**: Proposed the hypothesis that SGD generalizes because gradient directions common across many examples are reinforced more than idiosyncratic ones; supported with qualitative/simulated evidence.
- **How it relates**: This is the theoretical ancestor of the Spectral Optimizer's mechanism — the proposed work is a quantitative operationalization of the hypothesis via RMT.
- **Key difference**: Coherent Gradients is an explanatory hypothesis with no optimizer built on it and no RMT machinery; the academic search found no follow-up operationalizing it via eigenvalue/spectral methods. The proposed work builds and tests the operationalization in a new domain.
- **Source**: https://arxiv.org/abs/2002.10657

### 3. Financial Applications of Random Matrix Theory (Bouchaud & Potters, 2009; and the Laloux/Pafka/Kondor lineage)
- **Overlap**: Related (same mathematical tool, different object)
- **What they did**: Established ~20 years of Marchenko-Pastur eigenvalue cleaning of financial *return* correlation matrices — clipping the noise bulk improves portfolio risk estimates.
- **How it relates**: Supplies the exact denoising logic the Spectral Optimizer uses, and even the same domain (finance) — but applied to the returns-covariance matrix as a post-hoc estimation step, never to the per-sample *gradient* covariance during model training.
- **Key difference**: The transposition from returns covariance to training-gradient covariance appears unpublished. This literature is a framing/analogy asset, not competing prior work.
- **Source**: https://arxiv.org/abs/0910.1205

### 4. Orthogonal Gradient Constraints Shape Noisy-Label Memorization Dynamics (Mai, 2026)
- **Overlap**: Related
- **What they did**: OrthoGrad — removes the gradient component parallel to the current weight vector — reduces corrupted-label fitting on MNIST/CNNs in small-data regimes.
- **How it relates**: Same genre (geometric optimizer-level surgery against noise memorization) and evaluated on the same style of benchmark as the Spectral Optimizer's own prior MNIST validation. Confirms the niche is active as of 2026 — no one published this exact mechanism within the last 12 months, but adjacent mechanisms are appearing.
- **Key difference**: A per-parameter geometric projection, not cross-sample consensus; no covariance eigendecomposition, no RMT threshold, no financial data.
- **Source**: https://arxiv.org/abs/2607.16231

### 5. GradSentry: Gradient Spectral Entropy for Backdoor Sample Filtering (Zhao et al., 2026)
- **Overlap**: Related
- **What they did**: Used spectral entropy of per-sample gradients to detect and filter poisoned samples in LLM fine-tuning.
- **How it relates**: Independently validates the core premise that spectral properties of per-sample gradient populations carry exploitable signal.
- **Key difference**: Data filtering for backdoor detection, not a training update rule; entropy statistic, not covariance eigenspectrum thresholding; security domain, not financial regression.
- **Source**: https://arxiv.org/abs/2605.26574

### 6. Per-sample clipping (Nobile & Grohs, 2026) and DP-PMLF (Xu et al., 2025)
- **Overlap**: Tangential-to-Related
- **What they did**: Per-sample gradient clipping with heavy-tailed-noise convergence theory; per-sample momentum + low-pass filtering for DP-SGD.
- **How it relates**: Further evidence that per-sample gradient statistics are a live optimizer-design surface in 2025–2026.
- **Key differences**: Clipping/low-pass mechanisms, not consensus/eigenspectrum filtering; neither touches financial data.
- **Sources**: https://arxiv.org/abs/2605.02701, https://arxiv.org/abs/2511.08841

### 7. Benchmarking Optimizers for MLPs in Tabular Deep Learning (Gorishniy et al., 2026)
- **Overlap**: Tangential (methodology, not mechanism)
- **What they did**: Benchmarked 15 optimizers on 17 tabular datasets under matched protocols; Muon consistently beats AdamW.
- **How it relates**: Not competing prior work, but it sets the evidentiary bar: an optimizer claim on tabular/MLP data made without a matched-tuning-budget protocol (and arguably without Muon as a baseline) will not be credible.
- **Key difference**: No noise-filtering mechanism, no financial low-SNR regime, no per-sample gradients.
- **Source**: https://arxiv.org/abs/2604.15297

### 8. Hypothesis: Gradient Descent Prefers General Circuits (Pope, 2022)
- **Overlap**: Tangential
- **What they did**: Alignment Forum post informally restating the Coherent Gradients mechanism (general circuits get gradient reinforcement from many samples).
- **How it relates**: Shows the intuition was independently rediscovered in the community; no one there proposed the MP-spectral mechanism or a financial application.
- **Key difference**: Informal reasoning only; no method, no experiments.
- **Source**: https://www.alignmentforum.org/posts/JFibrXBewkSDmixuo/hypothesis-gradient-descent-prefers-general-circuits

## Differentiation Analysis

The proposed work is novel because no existing work combines per-sample gradient covariance eigendecomposition with Marchenko-Pastur thresholding as a training update rule, and no work in the gradient-agreement family (GAF, OrthoGrad, PS-Clip-SGD, GradSentry, DP-PMLF) has been evaluated on low-SNR financial prediction. The two literatures the idea sits between each stop short of the combination: the quant-finance RMT literature cleans *returns* covariance post-hoc, never training gradients; the per-sample-gradient optimizer literature uses pairwise/geometric/clipping heuristics, never a full covariance eigenspectrum with an RMT threshold. The domain application is also non-trivial in the calibration sense: financial noise is feature/target noise in a heavy-tailed low-SNR regression regime — not the synthetic label flips of the prior project's MNIST validation — and both the prior project's own weak-signal failure mode and Feldman's long-tail memorization theory predict the mechanism could plausibly *hurt* here. The outcome is genuinely uncertain in both directions, which is exactly what makes the transfer question informative either way.

Two caveats on the verdict:

1. **The novelty claim rests on absence of evidence.** Three independent search channels found nothing, which is supportive but not conclusive; the closest-mechanism paper (GAF) appeared only in late 2024, and the niche is producing adjacent papers at a rate of several per year. A same-idea preprint appearing during the run is a live possibility; the write-up should date the search and position against GAF explicitly rather than claiming a vacuum.
2. **The contribution is the transfer verdict, not the optimizer.** The Spectral Optimizer itself is unpublished prior work by the same author; the publishable contribution here is the rigorously-evaluated answer to "does gradient-consensus spectral filtering help in the low-SNR financial regime," including a well-documented negative result. Framing it as "a new optimizer beats Adam" would both overlap the GAF genre more and be harder to defend against the tuned-AdamW/Muon bar.

## Recommendations

- **Proceed to Step 4 (success criteria).** The novelty gap is confirmed as well as absence-of-evidence allows.
- **Position explicitly against GAF** in all framing: the differentiators are (a) MP-spectral thresholding of the full similarity matrix vs. pairwise orthogonality checks, and (b) the financial low-SNR regression domain. If budget allows, a GAF-style simple-agreement ablation arm would directly demonstrate that the spectral machinery adds value over the cheaper heuristic — this is the single strongest way to convert "related work exists" into a sharp contribution.
- **Meet the benchmark bar**: matched hyperparameter-tuning budgets (DeepOBS standard), multiple seeds, tuned AdamW as the primary baseline, Muon as a secondary baseline if the 5-experiment budget permits.
- **Adopt the Numerai practitioner protocol** (era-purged temporal CV, per-era correlation, FNC where feasible) so the evaluation is credible in the target domain — the absence of academic Numerai literature means the write-up itself contributes protocol documentation.
- **Test the Feldman counter-hypothesis directly** where cheap: report performance on rare-event/tail eras separately, since consensus filtering suppressing rare genuine signal is the sharpest predicted failure mode and finding it would itself be a contribution.
- **Frame both outcomes as deliverables**: a positive result establishes the transfer; a negative result, mechanistically connected to the weak-signal/long-tail failure mode, delineates the boundary of when coherence-amplifying optimizers help. Neither is a fallback.
