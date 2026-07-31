# Literature Synthesis

**Topic**: Spectral Optimizer (per-sample gradient consensus / covariance filtering) for noise reduction on financial time-series data
**Sources**: 28 curated references across academic (22), lab blogs (2), and community (4) channels. Full details in `search-001-academic.md`, `search-002-blogs.md`, `search-003-community.md`.

## Key Findings by Theme

### 1. Per-sample gradient filtering is an active but small niche (2024–2026) — and nobody does RMT spectral thresholding

The closest empirical precedent is **Gradient Agreement Filtering** (Chaubard et al. 2024, `chaubard2024gradientagreementfiltering`): filters/reweights macrobatch updates by cross-microbatch gradient agreement, finding late-training gradients are often orthogonal or negatively correlated across microbatches and that filtering disagreement reduces memorization. However, GAF uses simple orthogonality/correlation checks, not eigendecomposition of a gradient similarity matrix with a Marchenko-Pastur threshold, and targets distributed image-classification training, not noisy financial regression.

A burst of adjacent 2025–2026 work confirms per-sample gradient statistics are a live optimizer-design surface: OrthoGrad geometric constraints against noisy-label memorization (`mai2026orthogonalgradient`), per-sample clipped SGD with heavy-tailed-noise theory (`nobile2026persampleclipping`), gradient spectral entropy for backdoor-sample filtering (`zhao2026gradsentry`), and per-sample momentum + low-pass filtering for DP-SGD (`xu2025dppmlf`). GradSentry in particular independently validates that spectral properties of per-sample gradient populations carry exploitable signal. **None combines full covariance/similarity-matrix eigendecomposition with an RMT threshold as the update rule** — that remains the Spectral Optimizer's apparent point of novelty.

### 2. Coherent Gradients is the theoretical ancestor; the community independently rediscovered it

Chatterjee & Zielinski 2020 (`chatterjee2020coherentgradients`) argue SGD generalizes because directions common across many per-example gradients are reinforced more than idiosyncratic ones — the Spectral Optimizer is a quantitative operationalization of this via RMT. Quintin Pope's 2022 Alignment Forum post (`pope2022general_circuits`) independently articulates the same mechanism informally ("general circuits" get gradient reinforcement from many samples). No academic follow-up operationalizing Coherent Gradients via eigenvalue/RMT methods was found.

### 3. RMT covariance cleaning is mature in quant finance — but applied to returns, never to training gradients

The Bouchaud/Potters/Laloux lineage (`bouchaud2009rmtfinancereview`, `potters2005rmtoldlaces`, `pafka2004rmtexponentialweighting`, and recently `lakshtanov2023denoisingcorrelation`) established Marchenko-Pastur eigenvalue cleaning of financial return-correlation matrices ~20 years ago: large fractions of empirical eigenspectra are indistinguishable from noise, and clipping them improves portfolio risk estimates. This is a strong analogy base and framing asset for the project — the Spectral Optimizer transposes the same denoising logic from the returns-covariance domain to the per-sample gradient-covariance domain during training. The transposition itself appears unpublished.

### 4. The baseline is not naive: Adam-family optimizers already have noise robustness, and Muon is the new bar

Heavy-tailed gradient-noise theory (`yu2026signheavytailed`, `dang2025heavytailedstability`, `xie2022overlookedstructure`) shows gradient noise is structured and heavy-tailed, and that sign-based/adaptive normalization (i.e., what Adam already does) confers meaningful robustness in exactly this regime. Separately, a 2026 benchmark of 15 optimizers on tabular MLPs (`gorishniy2026benchmarkingoptimizers`) finds **Muon consistently beats AdamW** on tabular data. Implications: (a) beating tuned AdamW on noisy data is a genuinely high bar, not a strawman comparison; (b) Muon is a candidate additional baseline if budget permits.

### 5. Fair optimizer comparison methodology is well-codified

DeepOBS (`schneider2019deepobs`) and Fast Optimizer Benchmark (`blauth2024fastoptimizerbenchmark`) establish the standard: matched hyperparameter-tuning budgets, multiple seeds, standardized reporting. Ad hoc optimizer comparisons confounded by unequal tuning effort are a known failure mode of this literature — the experiment design must give AdamW the same tuning budget as the Spectral Optimizer.

### 6. Evaluation protocol for Numerai-style financial data comes from practitioner sources, not academia

No academic Numerai literature exists (confirmed gap across arXiv and Semantic Scholar). The practitioner-standard protocol comes from the Numerai forum: **era-based/purged cross-validation** (temporal group structure; `numerai_forum2021_eras`), **per-era correlation as the metric and loss target** (plain MSE misaligns with the evaluation metric), and **Feature Neutral Correlation** (`numerai_forum2021_fnc`) to distinguish genuine generalizable signal from narrow feature overfitting — high raw correlation with low FNC predicts drawdowns. One structural note: Numerai asset IDs reset each era, so true longitudinal per-asset recurrent modeling across eras is not possible on that dataset; recurrent/sequence architectures would need within-era structure or a different dataset (e.g., OHLCV) for a sequence-modeling arm.

### 7. Architecture landscape for noisy tabular/financial prediction

MLPs remain competitive and standard for tabular data, with GBTs the dominant classical baseline (`gorishniy2021revisitingtabular`). For the recurrent/sequence arm, lab work on covariance-informed optimization for RNNs exists (K-FAC for RNNs, `martens2018kfacrnn`) but for curvature preconditioning, not noise rejection. OpenAI's gradient-noise-scale framing (`mccandlish2018gradientnoisescale`) treats gradient noise as a scalar SNR statistic for batch-size selection — a useful contrast to full-spectrum filtering.

## Consensus vs. Disagreements

- **Consensus**: cross-sample gradient agreement correlates with generalization; suppressing incoherent components suppresses memorization (Coherent Gradients, GAF, OrthoGrad, community posts all align).
- **Key tension — the long-tail caveat**: Feldman 2019 (`feldman2019longtailmemorization`) proves memorization of rare examples can be *necessary* for generalization on long-tailed distributions. Financial data is heavy-tailed with rare regime-shift events; a consensus filter could suppress rare-but-genuine signal along with noise. This is the strongest theoretical argument that the Spectral Optimizer could *hurt* in the financial regime, consistent with the prior project's own finding that the filter hurts when signal is weak and not gradient-dominant (sparse parity). The outcome is genuinely uncertain in both directions.
- **Minor tension**: the noisy-label literature operates mostly at the loss/data level (surveys `liang2022noisylabelsurvey`, `song2025labelnoisesurvey`, ELR `liu2020earlylearningregularization`); optimizer-level gradient-covariance filtering is a comparatively unexplored corner of that space, which cuts both ways — novelty, but also little prior evidence it is the right level of intervention.

## Most Relevant Sources (top 6)

1. `chaubard2024gradientagreementfiltering` — closest mechanism; novelty differentiator is MP-spectral thresholding vs. simple agreement.
2. `chatterjee2020coherentgradients` — theoretical foundation for why consensus filtering should work.
3. `bouchaud2009rmtfinancereview` — canonical RMT covariance-cleaning in finance; the analogy the project transposes.
4. `gorishniy2026benchmarkingoptimizers` — methodological template for tabular optimizer benchmarking; flags Muon.
5. `numerai_forum2021_eras` + `numerai_forum2021_fnc` — the evaluation protocol (era-purged CV, per-era correlation, FNC) the experiments must adopt.
6. `feldman2019longtailmemorization` — the sharpest counter-hypothesis; motivates checking performance in rare-event/tail eras specifically.

## Gaps in Coverage

- No academic work on Numerai; practitioner forum knowledge only partially surveyed (deeper implementation-level threads exist on forum.numer.ai).
- No published work combining per-sample gradient RMT filtering with financial prediction — the novelty claim rests on absence of evidence across three independent search channels, which is supportive but not conclusive.
- Recurrent/state-space architectures on Numerai are structurally awkward (era-reset asset IDs); the sequence-architecture arm may need an OHLCV-style dataset or within-era sequence framing — a design decision for Steps 4–5.
- Lab blogs contributed only contrast/framing material (gradient noise scale, K-FAC); no direct engagement with the mechanism.

## Implications for Next Steps

- **Novelty (Step 3)**: strong prima facie case — the mechanism is novel in combination, with GAF as the nearest neighbor to differentiate against.
- **Success criteria (Step 4)**: must include matched tuning budgets (DeepOBS standard), multiple seeds, era-purged temporal validation, per-era correlation and ideally FNC as metrics, and tuned AdamW (not default hyperparameters) as the baseline.
- **Design (Step 5)**: the highest-information experiments target the transfer question — does consensus filtering help or hurt in a low-SNR regime where the Feldman long-tail caveat and the prior project's weak-signal failure mode both predict it could hurt.
