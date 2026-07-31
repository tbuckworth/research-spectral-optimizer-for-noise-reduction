# Search Results: Academic Sources (arXiv + Semantic Scholar)

## Search Queries Executed

### arXiv API
1. `per-sample gradient consensus filtering optimizer noise robustness` — 30 results (mostly irrelevant: dominated by unrelated "noise"/"consensus" physics and distributed-consensus control papers; a few relevant hits surfaced)
2. `coherent gradients deep learning memorization noisy labels` — 20 results (several directly relevant memorization/noisy-label hits)
3. `Marchenko-Pastur random matrix theory gradient covariance neural network training` — 25 results (mostly pure-math RMT papers; found the classic finance RMT reviews but no DL-training-gradient application)
4. `random matrix theory denoising covariance financial portfolio returns` — 20 results (strong hits: classic quant-finance RMT literature)
5. `adaptive gradient methods low signal-to-noise ratio regression generalization` — 20 results (mostly irrelevant/off-topic; one relevant optimizer-robustness hit)
6. `deep learning tabular financial time series prediction state space models transformers` — 25 results (several relevant hits on tabular DL and financial time-series benchmarks)
7. `Numerai tournament machine learning benchmark obfuscated financial features` — 15 results (no direct hits on Numerai; confirms the plan's predicted gap)
8. `gradient noise stochastic gradient descent implicit regularization heavy-tailed` — 15 results (several relevant hits on heavy-tailed gradient noise theory)

### Semantic Scholar API
1. `per-sample gradient agreement filtering optimizer robust training` — 20 results (found the single most directly relevant paper: Gradient Agreement Filtering)
2. `Marchenko-Pastur eigenvalue denoising covariance matrix estimation finance` — 15 results (one relevant finance-covariance-denoising hit; rest off-topic signal-processing/imaging)
3. `noisy label robust deep learning loss correction survey` — 20 results (multiple survey papers, mostly on label-correction rather than gradient-level methods)
4. `optimizer benchmark comparison deep learning fair evaluation hyperparameter tuning` — 15 results (found DeepOBS, the standard optimizer-benchmarking suite)
5. `machine learning stock return prediction low signal to noise ratio deep learning` — 20 results (several relevant financial-ML and denoising papers)

## Key Findings

### Gradient Agreement Filtering (GAF)
- **Source**: https://arxiv.org/abs/2412.18052
- **Authors**: Francois Chaubard, Duncan Eddy, Mykel J. Kochenderfer
- **Date**: 2024-12
- **Summary**: Introduces Gradient Agreement Filtering, which computes per-microbatch gradients in distributed data-parallel training and filters/reweights the macrobatch update based on cross-microbatch gradient agreement (orthogonality/negative correlation), rather than simple averaging. The authors find late-training gradients across microbatches are often orthogonal or negatively correlated, and that this correlates with memorization; filtering disagreeing gradients reduces memorization and improves generalization.
- **Key Claims**:
  - Naive gradient averaging across microbatches implicitly includes "noise-fitting" directions that hurt generalization.
  - A simple, computationally cheap agreement-based filter on the *microbatch* gradient population improves test performance over standard averaging.
- **Relevance**: HIGH — this is the closest known prior mechanism to the Spectral Optimizer's per-sample/per-microbatch gradient-consensus filtering, though GAF operates on inter-microbatch agreement (not full covariance/Marchenko-Pastur spectral thresholding) and targets distributed training rather than noisy-target financial regression. Directly informs the novelty argument (differentiates: MP-spectral thresholding of a covariance matrix vs. simple orthogonality-based filtering).
- **Cite Key**: chaubard2024gradientagreementfiltering

### Coherent Gradients: An Approach to Understanding Generalization in Gradient Descent-based Optimization
- **Source**: https://arxiv.org/abs/2002.10657
- **Authors**: Satrajit Chatterjee, Piotr Zielinski
- **Date**: 2020-02
- **Summary**: Proposes the "Coherent Gradients" hypothesis: because SGD sums per-example gradients, directions that are common (coherent) across many examples are reinforced more than idiosyncratic, example-specific directions, which explains why real (structured) data generalizes better than random labels despite equal capacity to fit both.
- **Key Claims**:
  - Explains implicit regularization of SGD as an emergent property of per-example gradient summation/agreement, not an explicit penalty.
  - Predicts that suppressing example-specific ("incoherent") gradient components should suppress memorization of noise while preserving genuine signal.
- **Relevance**: HIGH — this is the theoretical ancestor explicitly named in the search plan; the Spectral Optimizer's mechanism (spectral/covariance thresholding of per-sample gradient similarity) is a natural quantitative operationalization of the Coherent Gradients hypothesis via RMT rather than the qualitative/simulated mechanism in the original paper.
- **Cite Key**: chatterjee2020coherentgradients

### Orthogonal Gradient Constraints Shape Noisy-Label Memorization Dynamics
- **Source**: https://arxiv.org/abs/2607.16231
- **Authors**: Richard Mai
- **Date**: 2026-06
- **Summary**: Studies OrthoGrad, an optimizer-level intervention that removes the component of each weight gradient parallel to the current weight vector, evaluated on noisy-label image classification (MNIST, CNNs). Finds it reduces corrupted-label fitting and improves test accuracy in small-data regimes, particularly for CNNs.
- **Key Claims**:
  - A purely geometric constraint on the optimizer update (not a loss/architecture/data change) can suppress noisy-label memorization.
  - Effect size and applicability vary by architecture (clearest gains for CNNs vs. other architectures).
- **Relevance**: HIGH — very recent (2026) and directly analogous methodologically: an optimizer-level geometric/spectral intervention targeting noise memorization, evaluated on a noisy-label benchmark similar to the Spectral Optimizer's own MNIST label-noise validation. Important comparison point and evidence that "geometric optimizer surgery" is an active, small-scale research niche.
- **Cite Key**: mai2026orthogonalgradient

### Does Learning Require Memorization? A Short Tale about a Long Tail
- **Source**: https://arxiv.org/abs/1906.05271
- **Authors**: Vitaly Feldman
- **Date**: 2019-06 (STOC 2020)
- **Summary**: Provides a theoretical model showing that memorizing labels of rare/atypical training examples is necessary for near-optimal generalization on long-tailed data distributions, complicating any simple "memorization is always bad" narrative.
- **Key Claims**:
  - Memorization of long-tail examples can be *required* for good generalization, not merely a symptom of overfitting.
  - Any noise-suppression method (including gradient-agreement/spectral filtering) risks discarding legitimate long-tail signal along with noise.
- **Relevance**: MEDIUM-HIGH — important counter-consideration/caveat for the Spectral Optimizer project: filtering "disagreeing" per-sample gradients could suppress useful long-tail learning as well as noise, which is especially salient for financial data with rare regime-shift events.
- **Cite Key**: feldman2019longtailmemorization

### Early-Learning Regularization Prevents Memorization of Noisy Labels
- **Source**: https://arxiv.org/abs/2007.00151
- **Authors**: Sheng Liu, Jonathan Niles-Weed, Narges Razavian, Carlos Fernandez-Granda
- **Date**: 2020-06 (NeurIPS 2020)
- **Summary**: Shows that DNNs trained with noisy labels go through an "early learning" phase (fitting clean patterns) before memorizing noisy labels, proves this is fundamental even in linear models, and proposes a regularization term exploiting early-learning predictions to prevent the later memorization phase.
- **Key Claims**:
  - Early-learning/memorization is a two-phase dynamic that a loss-level regularizer can exploit.
  - Theoretical grounding of the phenomenon in simple (linear) models.
- **Relevance**: MEDIUM — establishes the phase-transition framing of memorization that's relevant context for any noise-suppression mechanism, but operates at the loss level rather than the optimizer/gradient level.
- **Cite Key**: liu2020earlylearningregularization

### Robust and Fast Training via Per-Sample Clipping
- **Source**: https://arxiv.org/abs/2605.02701
- **Authors**: Davide Nobile, Philipp Grohs
- **Date**: 2026-05
- **Summary**: Proposes per-sample clipped SGD (PS-Clip-SGD), a robust gradient estimator based on per-example gradient clipping, with theoretical convergence guarantees under heavy-tailed gradient noise and matching high-probability bounds.
- **Key Claims**:
  - Per-sample clipping (not just macro-gradient clipping) achieves optimal convergence under heavy-tailed noise.
  - Provides both in-expectation and high-probability convergence theory.
- **Relevance**: MEDIUM — adjacent per-sample-gradient-statistics optimizer method (clipping rather than covariance/agreement filtering), useful as a comparison baseline/related-work anchor and evidence per-sample gradient interventions are an active research area in 2026.
- **Cite Key**: nobile2026persampleclipping

### GradSentry: Gradient Spectral Entropy for Backdoor Sample Filtering in LLM Fine-Tuning
- **Source**: https://arxiv.org/abs/2605.26574
- **Authors**: Haodong Zhao, Tianyi Xu, Tianhang Zhao, Zhuosheng Zhang, Gongshen Liu
- **Date**: 2026-05
- **Summary**: Uses the spectral entropy of per-sample gradients to detect and filter poisoned/backdoor samples in LLM fine-tuning data, finding poisoned samples produce gradients with distinctively higher spectral entropy than clean samples.
- **Key Claims**:
  - Spectral properties of per-sample gradients (not just magnitude/direction) carry a detectable signal distinguishing "bad" from "clean" samples.
  - Method operates without needing a clean reference set, using spectral entropy as an unsupervised signal.
- **Relevance**: MEDIUM — different problem (backdoor detection vs. general noise robustness) but methodologically close: it independently validates that spectral analysis of per-sample gradient populations is a viable practical tool, lending indirect support to the Spectral Optimizer's premise that gradient spectra carry exploitable structure.
- **Cite Key**: zhao2026gradsentry

### Enhancing DPSGD via Per-Sample Momentum and Low-Pass Filtering
- **Source**: https://arxiv.org/abs/2511.08841
- **Authors**: Xincheng Xu, Thilina Ranbaduge, Qing Wang, Thierry Rakotoarivelo, David Smith
- **Date**: 2025-11
- **Summary**: Proposes DP-PMLF, combining per-sample momentum with low-pass filtering to jointly reduce DP-SGD's added noise and clipping bias, addressing a tradeoff that existing methods handle only partially.
- **Key Claims**: Per-sample momentum/filtering can simultaneously target two distinct sources of gradient corruption (privacy noise and clipping bias).
- **Relevance**: MEDIUM — a different application domain (differential privacy) but reinforces that per-sample gradient filtering combined with momentum/low-pass techniques is an established optimizer-design pattern, structurally analogous to combining spectral filtering with Adam/AdamW.
- **Cite Key**: xu2025dppmlf

### Financial Applications of Random Matrix Theory: a short review
- **Source**: https://arxiv.org/abs/0910.1205
- **Authors**: Jean-Philippe Bouchaud, Marc Potters
- **Date**: 2009-10
- **Summary**: Canonical review of applying random matrix theory (Marchenko-Pastur eigenvalue bulk, eigenvalue clipping) to clean empirical correlation/covariance matrices of financial returns, separating "signal" eigenvalues (portfolio structure) from "noise" eigenvalues consistent with the MP distribution.
- **Key Claims**: Large fractions of the eigenvalue spectrum of empirical return-correlation matrices are statistically indistinguishable from pure noise (MP prediction); portfolio risk estimates improve substantially when noise eigenvalues are shrunk/removed.
- **Relevance**: HIGH — establishes the classical quant-finance analogue (return covariance denoising) that the Spectral Optimizer explicitly generalizes to gradient covariance; essential background/contrast citation.
- **Cite Key**: bouchaud2009rmtfinancereview

### Financial Applications of Random Matrix Theory: Old Laces and New Pieces
- **Source**: https://arxiv.org/abs/physics/0507111
- **Authors**: Marc Potters, Jean-Philippe Bouchaud, Laurent Laloux
- **Date**: 2005-07
- **Summary**: Earlier/companion review covering the same RMT-covariance-cleaning territory (Laloux/Bouchaud-Potters line of work) with additional worked examples and portfolio-optimization implications.
- **Relevance**: MEDIUM-HIGH — companion/earlier version of the review above; useful for citing the original Laloux et al. lineage.
- **Cite Key**: potters2005rmtoldlaces

### Exponential Weighting and Random-Matrix-Theory-Based Filtering of Financial Covariance Matrices for Portfolio Optimization
- **Source**: https://arxiv.org/abs/cond-mat/0402573
- **Authors**: Szilard Pafka, Marc Potters, Imre Kondor
- **Date**: 2004-02
- **Summary**: Combines exponential weighting of historical returns with RMT-based eigenvalue filtering, showing the two techniques are complementary for constructing more accurate, lower-risk covariance estimates for portfolio optimization.
- **Relevance**: MEDIUM — a practical variant of the RMT-cleaning method combined with time-weighting; useful for the "classical RMT covariance filtering" literature base but not gradient-related.
- **Cite Key**: pafka2004rmtexponentialweighting

### A novel approach to denoising correlation matrices with applications to global portfolio management with a large number of assets
- **Source**: https://www.semanticscholar.org/paper/3cb918f4577cd603840d43e1a7ead43d6551b49a
- **Authors**: E. Lakshtanov, Marat Molyboga
- **Date**: 2023
- **Summary**: Proposes a modern correlation-matrix denoising method for large-scale (high-dimensional) global portfolios, extending the RMT-cleaning tradition to more realistic institutional-scale settings.
- **Relevance**: MEDIUM — recent, practically-oriented extension of the classical RMT covariance-cleaning literature; shows the technique is still actively developed for large asset universes (relevant scale analogy to large financial feature sets like Numerai's).
- **Cite Key**: lakshtanov2023denoisingcorrelation

### Benchmarking Optimizers for MLPs in Tabular Deep Learning
- **Source**: https://arxiv.org/abs/2604.15297
- **Authors**: Yury Gorishniy, Ivan Rubachev, Dmitrii Feoktistov, Artem Babenko
- **Date**: 2026-04
- **Summary**: Systematically benchmarks 15 optimizers (including AdamW, Muon, and others) on 17 tabular datasets for MLP-based tabular deep learning under a shared protocol, finding Muon consistently outperforms AdamW, the current default.
- **Key Claims**: The choice of optimizer for tabular DL has been under-examined relative to architecture choice; systematic, matched-protocol benchmarking reveals meaningful, consistent optimizer differences.
- **Relevance**: HIGH — directly relevant methodological anchor for benchmarking a new optimizer (the Spectral Optimizer) against AdamW on MLP/tabular-style data; also flags Muon as a strong, very recent baseline that should probably be included as a comparison optimizer alongside AdamW.
- **Cite Key**: gorishniy2026benchmarkingoptimizers

### Revisiting Deep Learning Models for Tabular Data
- **Source**: https://arxiv.org/abs/2106.11959
- **Authors**: Yury Gorishniy, Ivan Rubachev, Valentin Khrulkov, Artem Babenko
- **Date**: 2021-06 (NeurIPS 2021)
- **Summary**: Systematic, fair comparison of deep learning architectures for tabular data (MLP variants, ResNet, FT-Transformer) against gradient-boosted trees, establishing strong, simple baselines and a shared evaluation protocol.
- **Relevance**: MEDIUM-HIGH — establishes that MLPs remain a competitive and standard architecture choice for tabular data (supporting the project's MLP baseline choice) and that GBTs remain the dominant baseline to beat, consistent with the search plan's expectations.
- **Cite Key**: gorishniy2021revisitingtabular

### DeepOBS: A Deep Learning Optimizer Benchmark Suite
- **Source**: https://arxiv.org/abs/1903.05499
- **Authors**: Frank Schneider, Lukas Balles, Philipp Hennig
- **Date**: 2019-03
- **Summary**: Argues there is no agreed-upon protocol for evaluating deep learning optimizers, and introduces DeepOBS, a benchmark suite standardizing test problems, tuning budgets, and reporting to enable fair, reproducible optimizer comparisons.
- **Key Claims**: Fair optimizer comparison requires matched tuning budgets and standardized problem/reporting protocols; ad hoc comparisons in the optimizer literature are often confounded by unequal tuning effort.
- **Relevance**: HIGH — directly relevant to the project's own evaluation design: is a canonical methodological reference for avoiding tuning-budget confounds when comparing the Spectral Optimizer to AdamW baselines.
- **Cite Key**: schneider2019deepobs

### Fast Optimizer Benchmark
- **Source**: https://arxiv.org/abs/2406.18701
- **Authors**: Simon Blauth, Tobias Bürger, Zacharias Häringer (Semantic Scholar record; verify additional authors from source)
- **Date**: 2024-06
- **Summary**: A newer, lightweight/fast benchmark suite for optimizer comparison, presumably addressing the compute cost of full protocols like DeepOBS.
- **Relevance**: MEDIUM — potential lower-cost benchmarking methodology reference, useful given this project's likely constrained compute budget; abstract not independently verified beyond the Semantic Scholar record, treat with appropriate caution.
- **Cite Key**: blauth2024fastoptimizerbenchmark

### Sign-Based Optimizers Are Effective Under Heavy-Tailed Noise
- **Source**: https://arxiv.org/abs/2602.07425
- **Authors**: Dingzhi Yu, Hongyi Tao, Yuanyu Wan (et al.)
- **Date**: 2026-02
- **Summary**: Analyzes why sign-based optimizers (e.g., signSGD, and by extension Adam's normalization behavior) are robust under heavy-tailed gradient noise, providing theoretical convergence guarantees in this regime.
- **Relevance**: MEDIUM-HIGH — directly relevant theoretical framing for why standard adaptive optimizers (Adam/AdamW) already have some built-in heavy-tailed-noise robustness, which the Spectral Optimizer needs to beat, not just match.
- **Cite Key**: yu2026signheavytailed

### On the Overlooked Structure of Stochastic Gradients
- **Source**: https://arxiv.org/abs/2212.02083
- **Authors**: Zeke Xie, Qian-Yuan Tang, Mingming Sun, Ping Li
- **Date**: 2022-12
- **Summary**: Empirically and theoretically studies structural (non-Gaussian, heavy-tailed, and correlated) properties of stochastic gradient noise across training, challenging simplified Gaussian-noise assumptions common in optimization theory.
- **Relevance**: MEDIUM — provides evidence that gradient noise has exploitable non-trivial structure (heavy tails, correlation) beyond simple magnitude, supporting the plausibility of a covariance/spectral-based filtering approach as opposed to plain gradient clipping.
- **Cite Key**: xie2022overlookedstructure

### Algorithmic Stability of Stochastic Gradient Descent with Momentum under Heavy-Tailed Noise
- **Source**: https://arxiv.org/abs/2502.00885
- **Authors**: Thanh Dang, Melih Barsbey, A K M Rokonuzzaman Sonet (et al.)
- **Date**: 2025-02
- **Summary**: Provides generalization/stability bounds for SGD-with-momentum specifically under heavy-tailed gradient noise, connecting noise structure to generalization theory.
- **Relevance**: MEDIUM — theoretical anchor connecting gradient noise structure directly to generalization bounds, useful in framing why noise-structure-aware optimizers (like the Spectral Optimizer) could plausibly improve generalization, not just training stability.
- **Cite Key**: dang2025heavytailedstability

### A Robust Adaptive Stochastic Gradient Method for Deep Learning
- **Source**: https://arxiv.org/abs/1703.00788
- **Authors**: Caglar Gulcehre, Jose Sotelo, Marcin Moczulski, Yoshua Bengio
- **Date**: 2017-03
- **Summary**: Proposes a robust adaptive gradient method designed to be less sensitive to gradient outliers/noise than standard Adam-style methods, an earlier precedent for optimizer-level robustness to noisy gradients.
- **Relevance**: LOW-MEDIUM — older, general-purpose robust optimizer; useful as historical context for "robust adaptive optimizer" lineage but not RMT/spectral-specific.
- **Cite Key**: gulcehre2017robustadaptive

### Review — A Survey of Learning from Noisy Labels
- **Source**: https://www.semanticscholar.org/paper/a68fafb588d7ffe28ead8cd0286d4537cec07fbe
- **Authors**: Xuefeng Liang, Xingyu Liu, Longshan Yao
- **Date**: 2022
- **Summary**: Survey covering the broad landscape of noisy-label learning methods (loss correction, sample selection, robust architectures, regularization).
- **Relevance**: MEDIUM — provides broad context for where the Spectral Optimizer's MNIST label-noise validation sits relative to the wider noise-robust-training literature; most cited methods operate at the loss/data level rather than the optimizer/gradient-covariance level, reinforcing that the project's mechanism (RMT-spectral gradient filtering) occupies a comparatively under-explored niche within this space.
- **Cite Key**: liang2022noisylabelsurvey

### A survey on learning from data with label noise via deep neural networks
- **Source**: https://www.semanticscholar.org/paper/5091d308e7a42d569ca0e279f27005c022302188
- **Authors**: Baoye Song, Shihao Zhao, Luyao Dang (et al.)
- **Date**: 2025
- **Summary**: More recent (2025) comprehensive survey of DNN noisy-label learning methods.
- **Relevance**: MEDIUM — complements the above survey with a more current view of the field as of 2025.
- **Cite Key**: song2025labelnoisesurvey

## Themes Identified

- **Per-sample/per-microbatch gradient filtering is an active but small niche (2024-2026)**: Gradient Agreement Filtering (chaubard2024gradientagreementfiltering), Orthogonal Gradient Constraints (mai2026orthogonalgradient), Per-Sample Clipping (nobile2026persampleclipping), GradSentry (zhao2026gradsentry), and DP-PMLF (xu2025dppmlf) all show that 2024-2026 has seen a burst of interest in operating on per-example/per-microbatch gradient statistics at the optimizer level, but none combine this with full Marchenko-Pastur-style spectral/eigenvalue thresholding of a gradient covariance/similarity matrix — this appears to be the Spectral Optimizer's genuine point of novelty.
- **Coherent Gradients as the theoretical ancestor**: Chatterjee & Zielinski's 2020 hypothesis (chatterjee2020coherentgradients) is the clearest conceptual precedent explaining *why* gradient-agreement filtering should suppress noise memorization; GAF (chaubard2024gradientagreementfiltering) is the closest empirical operationalization found, but it uses simple orthogonality/correlation checks rather than RMT eigenvalue thresholds.
- **Classical RMT covariance-cleaning is a mature, ~20-year-old quant-finance literature**: Bouchaud/Potters/Laloux/Pafka/Kondor (bouchaud2009rmtfinancereview, potters2005rmtoldlaces, pafka2004rmtexponentialweighting) established Marchenko-Pastur eigenvalue cleaning for return-covariance matrices well before deep learning adopted similar ideas; more recent work (lakshtanov2023denoisingcorrelation) shows it is still being extended for large-scale portfolios — a strong analogy base for the Spectral Optimizer's core mechanism, applied to gradients instead of returns.
- **Memorization-vs-generalization tension complicates any blanket noise-suppression claim**: Feldman's long-tail theory (feldman2019longtailmemorization) shows memorization can be *necessary* for good generalization on heavy-tailed data — a caveat directly relevant to financial time series with rare regime-shift events, where "noise filtering" could inadvertently suppress rare but genuine signal.
- **Optimizer benchmarking methodology is a live, well-developed concern**: DeepOBS (schneider2019deepobs), Fast Optimizer Benchmark (blauth2024fastoptimizerbenchmark), and the very recent tabular-MLP optimizer benchmark (gorishniy2026benchmarkingoptimizers, which found Muon beats AdamW) together give a strong methodological template — and flag Muon as a baseline the Spectral Optimizer arguably needs to be compared against, not just AdamW.
- **Heavy-tailed gradient noise theory is mature and directly relevant**: Multiple 2025-2026 papers (yu2026signheavytailed, dang2025heavytailedstability, xie2022overlookedstructure) establish that gradient noise has non-Gaussian, structured, heavy-tailed properties exploitable by robust optimizers, and that sign-based/adaptive methods (i.e., Adam's own normalization) already provide some heavy-tailed robustness — meaning the Spectral Optimizer's comparison baseline (AdamW) is not naive to noise, raising the bar for a convincing improvement.

## Gaps

- **No academic literature on the Numerai tournament dataset specifically** was found in either arXiv or Semantic Scholar searches (Query 7 and general financial-ML queries). This confirms the search plan's prediction: Numerai's obfuscated-feature, era-based evaluation protocol appears to be documented almost exclusively in practitioner/community sources (forum.numer.ai), not peer-reviewed venues.
- **No paper was found combining per-sample gradient RMT/Marchenko-Pastur spectral filtering with financial time-series prediction** — confirming the plan's expectation that this exact combination is a genuinely novel niche, not a rediscovery of prior published work.
- **No direct academic follow-up citing Chatterjee & Zielinski's Coherent Gradients (2020) that operationalizes it via RMT/eigenvalue spectral methods** was found; the closest related work (GAF, OrthoGrad) uses simpler orthogonality/clipping criteria rather than full covariance-spectrum thresholding.
- Semantic Scholar's citation-count field was frequently low/zero for the most relevant (2024-2026) hits, since these are very recent papers; citation-based relevance ranking was not reliable for this topic and direct abstract review was needed instead.
- The Semantic Scholar API returned repeated 429 (rate limit) errors during this session, requiring retries with exponential backoff; all five planned queries eventually succeeded, but this indicates the shared API key/IP may be near a rate ceiling — future searches in this run should budget extra time for Semantic Scholar calls.
