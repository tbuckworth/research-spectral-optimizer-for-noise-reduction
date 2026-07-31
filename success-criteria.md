# Success Criteria

**Framing note.** This project asks a directional transfer question — does per-sample gradient MP-spectral filtering (the existing Spectral Optimizer) improve out-of-sample performance vs tuned Adam/AdamW on low-SNR financial time-series prediction? The deliverable is a verdict: *helps / doesn't help / hurts*, across an MLP and at least one sequence architecture. "SOTA" here therefore has two distinct meanings, and both matter:

1. **Task SOTA** (what score is good on noisy financial prediction) — sets realistic metric ranges so results are interpretable.
2. **Comparison SOTA** (what optimizer baseline must be matched/beaten, under what protocol) — sets the credibility bar for the verdict.

The contribution is the verdict plus the protocol, not a leaderboard entry. A well-supported negative result is a full deliverable, not a fallback (per the novelty assessment).

## State of the Art

### Current Best Methods

| Method | Source | Key Metric | Value | Year |
|--------|--------|------------|-------|------|
| Muon (tabular MLPs) | `gorishniy2026benchmarkingoptimizers` | rank vs 14 optimizers, 17 tabular datasets | consistently beats AdamW | 2026 |
| Tuned AdamW (tabular MLPs) | `gorishniy2026benchmarkingoptimizers`, `gorishniy2021revisitingtabular` | de-facto default; strong when tuned | reference baseline | 2021–2026 |
| GBTs (LightGBM-style) | `gorishniy2021revisitingtabular`, Numerai community | dominant classical Numerai baseline | mean per-era corr ≈ 0.02–0.03 (validation) | ongoing |
| Numerai practitioner NN models | `numerai_forum2021_eras`, `numerai_forum2021_fnc` | mean per-era corr (validation eras) | ≈ 0.015–0.03; corr-Sharpe ≈ 0.5–1.0; FNC ≈ 0.005–0.015 | 2021– |
| Gradient Agreement Filtering | `chaubard2024gradientagreementfiltering` | CIFAR-100(N) accuracy vs SGD | +up to ~18pp under label noise; never on regression/finance | 2024 |
| Spectral Optimizer (prior project) | `/home/titus/pyg/optimizers` README | MNIST @ 90% label noise, accuracy | ~80% vs ~37% (Adam); hurts on sparse parity (weak signal) | 2025 (unpublished) |

*Numerai numbers are practitioner-reported ranges, not peer-reviewed results; there is no academic Numerai literature (confirmed gap, Step 2). Treat them as sanity ranges, not targets to beat.*

### SOTA Summary

**Task side.** On Numerai-style obfuscated financial data, absolute performance is tiny by ML standards: a *good* model achieves mean per-era Spearman correlation around 0.02–0.03 on held-out eras, with per-era corr standard deviation of the same order or larger (hence corr-Sharpe ≈ 0.5–1.0 being respectable). Feature Neutral Correlation is smaller still. Signal detection, not accuracy maximization, is the game. Any per-era corr in the 0.01–0.04 band on properly purged validation eras confirms the pipeline is working; numbers far above that band indicate leakage, not brilliance — this is the key sanity check for the whole study.

**Comparison side.** The optimizer-comparison bar is set methodologically, not numerically. Adam-family optimizers already carry heavy-tailed-noise robustness (`yu2026signheavytailed`, `xie2022overlookedstructure`), so tuned AdamW is a genuinely strong opponent, and the 2026 tabular benchmark shows Muon beats AdamW on MLP/tabular tasks — meaning "beats default-hyperparameter Adam" would be a strawman claim. The field's accepted standard (DeepOBS, `schneider2019deepobs`; `blauth2024fastoptimizerbenchmark`) is: matched hyperparameter-tuning budgets, multiple seeds, and standardized reporting. Gorishniy et al. 2026 is the closest thing to a current protocol reference for this exact model class (MLPs on tabular data).

**Where the specific question stands.** Nobody has run a per-sample gradient covariance/RMT filter on financial prediction (novelty verdict: NOVEL). The nearest mechanism (GAF) improved generalization under image-classification label noise; the prior project's own results and Feldman's long-tail theory (`feldman2019longtailmemorization`) predict the mechanism could hurt in the weak-signal financial regime. SOTA for the *question* is therefore: no evidence either way. Any competently-executed answer moves the field's knowledge from zero.

## Benchmarks

### Standard Evaluation

| Benchmark/Dataset | Metrics | Typical Range | Notes |
|-------------------|---------|---------------|-------|
| Numerai public dataset (v4.x/v5, free download) | mean per-era Spearman corr; per-era corr Sharpe; FNC; max era drawdown | corr 0.01–0.04; Sharpe 0.5–1.0; FNC 0.005–0.015 | Canonical large/noisy/obfuscated financial tabular task. Era-purged temporal CV mandatory (embargo gap between train/val eras). Asset IDs reset per era → no longitudinal recurrent modeling; sequence arm needs within-era framing or OHLCV. |
| Public OHLCV next-period return prediction (e.g., Yahoo-style daily equities/crypto) | out-of-sample IC/rank-corr, MSE vs naive baselines | IC ≈ 0.0–0.05; must beat zero-predictor / historical-mean | Fallback + sequence-architecture arm. No standardized splits exist — the protocol (strict temporal split, purge/embargo) must be documented explicitly. |
| Tail/rare-event eras (subset of the above) | same metrics restricted to worst-decile / high-volatility eras | wider spread, often negative | Direct test of the Feldman counter-hypothesis; cheap (re-slicing existing predictions, no extra GPU time). |
| DeepOBS-style protocol (not a dataset) | matched tuning budget, ≥3 seeds, report mean±sd across seeds and eras | n/a | Credibility requirement for any optimizer claim (`schneider2019deepobs`, `gorishniy2026benchmarkingoptimizers`). |

**Known benchmark issues**: Numerai target is obfuscated and era-binned (regression on ~5-bin targets); plain MSE misaligns with the corr metric (`numerai_forum2021_eras`) — training loss choice must be held fixed across optimizer arms so the comparison isolates the optimizer. Raw corr without FNC can reward feature overfitting. Subsampling eras/features to fit the <30-min job budget is acceptable if identical across arms.

### Required Baselines

For the verdict to be credible, the study must compare against:

1. **Tuned AdamW** (same tuning budget as the Spectral Optimizer, same search space size, ≥3 seeds) — the primary comparison; the motivating question is literally "vs tuned Adam/AdamW". Default-hyperparameter Adam alone is a known strawman.
2. **Spectral Optimizer's own base optimizer, unfiltered** — since the filter wraps AdamW, the cleanest ablation is identical base config with the filter on vs off. This isolates the mechanism from hyperparameter luck and is the single most load-bearing comparison.
3. **GAF-style simple-agreement filter** (optional but high-value, from the novelty assessment) — same batch, pairwise agreement check instead of MP eigendecomposition. Distinguishes "any consensus filtering helps/hurts" from "the spectral machinery specifically matters". If the 5-experiment budget forces a cut, cut this before the first two.
4. **Muon** (secondary, budget permitting) — the 2026 tabular bar; absence is defensible in a workshop paper if noted as a limitation, since the claim is about the filter's delta, not overall optimizer supremacy.
5. **Zero-predictor / era-mean sanity baseline** (free) — confirms the models find any signal at all; essential context for interpreting a null.

Not required within budget: GBT baselines (LightGBM) — useful context but orthogonal to the optimizer question; note as future work.

## Publishability Criteria

### Target Venues

- **Primary**: **TMLR** — explicitly values well-executed empirical studies and negative/null results with rigorous protocols; no novelty-of-mechanism requirement (the optimizer is the author's prior unpublished work; the contribution is the transfer verdict). Fits the "verdict either way" framing best.
- **Secondary**: **ACM ICAIF** (AI in Finance) — the domain match is exact (optimizer-level study on noisy financial prediction with practitioner-grade protocol); the absence of academic Numerai literature makes the protocol documentation itself a contribution there.
- **Workshop**: **NeurIPS OPT (Optimization for Machine Learning)** or an ICML workshop on ML for finance — appropriate for the scoped 5-experiment version; optimizer-comparison studies with matched-budget protocols are core OPT content.

### Evidence Thresholds

**Minimum viable (workshop paper)**:
- One dataset (Numerai), one architecture (MLP), Spectral Optimizer vs tuned AdamW *and* the filter-on/filter-off ablation, ≥3 seeds each, era-purged temporal validation with embargo, per-era corr reported per-era (not just the mean).
- Matched tuning budget documented (same number of trials, same search space scope for both arms).
- A clear directional verdict with uncertainty: paired per-era comparison (filter vs no-filter on the same eras/seeds) with a bootstrap or paired-test confidence interval on the mean per-era corr difference.
- Evidence the filter was actually active (spectral diagnostics: number of eigendirections kept vs the MP bulk, fraction of gradient norm passed) — without this, a null is uninterpretable as "mechanism doesn't help" vs "mechanism never engaged".

**Solid contribution (main conference / TMLR)**:
- All of the above, plus the second architecture arm (within-era sequence model on Numerai or LSTM/GRU-class model on OHLCV), giving a consistency statement across architectures.
- FNC (or a documented feature-neutralization proxy) alongside raw corr, so the verdict distinguishes real signal from feature overfitting.
- Tail-era breakdown testing the Feldman long-tail counter-hypothesis (does filtering specifically hurt on rare-regime eras?).
- A mechanistic account connecting the outcome to the prior project's coherence-amplifier characterization: e.g., measured gradient-coherence spectra on financial data vs MNIST-label-noise, showing *why* the regime does or doesn't suit the filter.
- The GAF-style ablation arm, converting "related work exists" into "the spectral machinery does/doesn't add value over the cheap heuristic".

**Strong contribution (best paper contender)**:
- A predictive boundary characterization: a measurable statistic of the data/gradient regime (e.g., signal eigenvalue separation from the MP bulk) that predicts, ahead of time, whether spectral filtering helps — validated across both datasets and architectures. This would upgrade the paper from a verdict to a usable decision rule for when coherence-amplifying optimizers apply. (Likely beyond the 5-experiment budget; treat as stretch/future work.)

### What Counts as an Informative Outcome

Positive, negative, and null results are treated as equally valuable. An outcome is informative and publishable if:

- The comparison was fair by field standards: matched tuning budgets, ≥3 seeds, identical data pipeline/loss/architecture across arms, era-purged temporal splits — so the result cannot be attributed to unequal tuning effort (the known failure mode of optimizer papers).
- The filter's engagement was verified via spectral diagnostics, so "no effect" means "mechanism engaged but didn't change generalization", not "implementation no-op".
- The effect estimate carries calibrated uncertainty: a paired per-era CI on the corr difference. **Helps** = CI excludes zero in the positive direction and the point estimate is practically meaningful (≥ ~0.005 mean per-era corr, i.e., ~20–25% relative on a 0.02-corr task, or a clear corr-Sharpe improvement). **Hurts** = same, negative. **Doesn't help (null)** = CI is narrow enough to exclude improvements ≥ ~0.005 mean corr — an equivalence-style statement, not just p > 0.05.
- Absolute performance lands in the sane range for the task (per-era corr ≈ 0.01–0.04 for the working arms, near zero for the sanity baseline), ruling out leakage or a broken pipeline as the explanation.
- The write-up dates the literature search and positions against GAF explicitly (novelty caveat 1), and frames the contribution as the transfer verdict, not a new optimizer (caveat 2).

## Minimum Viable Contribution

A single-dataset (Numerai), single-architecture (MLP) study — tuned AdamW vs Spectral-filter-on-AdamW under matched tuning, ≥3 seeds, era-purged validation, per-era paired analysis with spectral engagement diagnostics — delivering one of:

- **Helps**: the first evidence that Coherent-Gradients-style spectral filtering transfers to low-SNR financial regression;
- **Hurts / null**: the first documented boundary case for the mechanism in a real (not synthetic-label-noise) low-SNR regime, mechanistically connected to the weak-signal/long-tail failure mode.

Either verdict, so supported, clears the workshop bar. This fits comfortably in ≤3 of the 5 experiment slots (tuning sweep, main seeded comparison, diagnostics can share jobs), each well under 30 GPU-minutes for MLP-scale models with vmap per-sample gradients at moderate batch sizes on one L40, leaving ≥2 slots for the sequence arm and/or the GAF ablation.

## Risks to Success

- **Uninterpretable null (filter never engages)**: On financial data the gradient eigenspectrum may sit entirely inside the MP bulk (no coherent signal to keep) or entirely outside (nothing filtered). *Mitigation*: log spectral diagnostics from the first run; if the filter is degenerate, that is itself a reportable mechanistic finding ("no gradient-coherent signal separable from noise at batch size B") — report it as such, and try the soft/variance threshold modes before concluding.
- **Effect smaller than the noise floor of the design**: With ~small per-era corr and high era variance, 3 seeds x limited eras may give a CI too wide to support any of the three verdicts. *Mitigation*: paired per-era design (same eras, same seeds across arms) cancels most era variance; use as many validation eras as the data allows (era count, not model size, drives power and is nearly free); if still underpowered, report the CI honestly and scope the claim to what it supports.
- **Tuning-budget confound**: The Spectral Optimizer has extra hyperparameters (threshold mode, batch size interacts with the B×B matrix), risking either unfair advantage or unfair handicap. *Mitigation*: equal trial counts per arm, pre-registered search spaces in the experiment plan, and the filter-on/off ablation at identical base hyperparameters as the primary evidence.
- **Sequence-arm fragility**: torch.func vmap per-sample gradients through recurrent models is the least-tested code path (Step 1 assumption 2), and Numerai can't host a longitudinal recurrent arm at all. *Mitigation*: budget the sequence arm as one experiment with a fallback (within-era attention/sequence framing on Numerai if OHLCV+LSTM+vmap fails); if it fails on affordability/feasibility, the MLP-only verdict still clears the minimum bar with the architecture-consistency claim explicitly dropped.
- **Compute-profile overrun**: 5 experiments, <30 min each is tight for tuning sweeps. *Mitigation*: subsample eras/features identically across arms; run tuning as one array-style experiment; Muon and GBT baselines are the designated cuts (recorded as future work), never the seeds or the matched budget.
- **Scooping**: the niche produced several adjacent papers per year in 2024–2026. *Mitigation*: date the search (2026-07-31) in the write-up; the financial-domain evaluation would remain unoccupied even if the mechanism appears elsewhere.
