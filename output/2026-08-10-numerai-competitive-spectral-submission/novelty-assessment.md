# Novelty Assessment

## Proposed Research

Build an honestly tuned AdamW MLP for the current Numerai Tournament, then test whether projecting each gradient onto the leading eigenspace of a streaming parameter-space gradient-covariance estimate improves paired, leakage-resistant v5.3 performance and benchmark-relative contribution. The method tracks persistent directions of the conceptual \(p \times p\) covariance through low-rank/rank-one updates rather than forming that matrix; it is not an FFT, a parameter-frequency mask, or temporal smoothing of financial observations.

## Verdict: PARTIALLY_NOVEL

The application and controlled evaluation appear novel within the searched evidence, but the optimizer is not being introduced here for the first time. It already exists in the associated optimizer project and has been tested on image classification, label noise, grokking, sparse parity, and backdoors. The defensible contribution is therefore a new domain-transfer and comparative evaluation: the first located test of this streaming gradient-covariance eigenspace filter on Numerai's era-indexed equity panel, under equal HPO budgets and current official scoring. It should not be presented as a new spectral optimizer.

## Closest Existing Work

### 1. Spectral Gradient Filter (associated optimizer project, current implementation)

- **Overlap**: Near-identical
- **What it did**: Wrapped a base optimizer and projected gradients onto the top directions of a streaming gradient-covariance estimate. Existing experiments cover noisy-label MNIST/CIFAR-10, modular addition, sparse parity, and planted backdoors, with mixed results: strong resistance to label-noise memorization, task-dependent grokking effects, and failure when useful signal is weak or distributed.
- **How it relates**: This is the same optimizer family and mechanism proposed for the Numerai experiment.
- **Key difference**: The existing work does not test anonymized cross-sectional equity returns, Numerai's weekly eras and overlapping target horizons, official CORR/BMC metrics, benchmark blending, or a tuned AdamW comparison under nested walk-forward selection. The new contribution is application and evaluation, not algorithm invention.
- **Source**: [Optimizer repository README](https://github.com/tbuckworth/optimizers)

### 2. Empirical Asset Pricing via Machine Learning (Gu, Kelly, and Xiu, 2020)

- **Overlap**: Related
- **What it did**: Evaluated machine-learning models on a large firm-by-month return-prediction panel using temporally ordered training, validation, and genuinely out-of-sample test periods with rolling updates.
- **How it relates**: It is the closest high-quality methodological analogue for neural prediction on a cross-sectional financial panel and supports the proposed separation of tuning from final temporal evaluation.
- **Key difference**: It neither studies Numerai nor uses a streaming gradient-covariance eigenspace filter, current Numerai targets, official benchmark predictions, CORR, or BMC.
- **Source**: https://doi.org/10.1093/rfs/hhaa009

### 3. Numerai Tournament Example Code Using PyTorch NN and Optuna (meaten12121, 2021/2022)

- **Overlap**: Related
- **What it did**: Provided code-linked Numerai neural-network practice using time-series cross-validation, Optuna HPO, era-boosted training, and era-contained batches.
- **How it relates**: It establishes that tuned PyTorch neural networks and temporal HPO have already been applied to the Numerai Tournament.
- **Key difference**: The located report does not study AdamW specifically, compare optimizers under equal budgets, or use the proposed covariance-eigenspace filter. It is also not evidence on the current v5.3 contract or live competitiveness.
- **Source**: https://forum.numer.ai/t/numerai-tournament-example-code-using-pytorch-nn-and-optuna/4639

### 4. Vibesciencing My Way Through v5.2 Data (Faith II) (degerhan, 2025; update 2026)

- **Overlap**: Related
- **What it did**: Built a leakage-aware walk-forward Numerai ensemble containing an MLP ranking model, benchmark-residual models, boosted-tree anchors, sparse ensemble weights, and mild feature neutralization; it reported benchmark-relative offline scores and an initially poor forward update.
- **How it relates**: It is the closest recent community example combining an MLP, walk-forward selection, benchmark-relative contribution, and a prospective Numerai submission.
- **Key difference**: The MLP is a diversity component rather than a controlled optimizer experiment; no spectral gradient-covariance filtering or equal-budget AdamW comparison is reported, and it uses v5.2 rather than the current v5.3 contract.
- **Source**: https://forum.numer.ai/t/vibesciencing-my-way-through-v5-2-data-faith-ii/8214

### 5. On Empirical Comparisons of Optimizers for Deep Learning (Choi et al., 2020)

- **Overlap**: Related
- **What it did**: Showed that optimizer rankings can reverse when search spaces and tuning protocols change.
- **How it relates**: It directly motivates treating AdamW and spectral optimization as complete, separately tuned procedures with comparable data and selection budgets.
- **Key difference**: It does not study this filter, finance, or Numerai; it constrains the validity of the proposed comparison rather than anticipating its substantive result.
- **Source**: https://arxiv.org/abs/1910.05446

### 6. Spectral Momentum Integration (Huang, Chen, and Zheng, 2025)

- **Overlap**: Tangential
- **What it did**: Applied a 2-D FFT to reshaped parameter gradients, tracked an EMA of coefficient magnitudes, masked coefficients by a quantile rule, inverse-transformed, and blended the result with the original gradient.
- **How it relates**: It shares the words “spectral,” “gradient,” and “optimizer,” but acts on Fourier coefficients indexed by tensor layout.
- **Key difference**: The proposed method eigendecomposes a streaming covariance operator over parameter-space gradient directions. It has no Fourier transform, frequency axis, reshape-dependent frequency mask, or low-pass interpretation. SMI therefore is not a direct antecedent and its evidence should not be used as support for the proposed mechanism.
- **Source**: https://doi.org/10.3389/frai.2025.1628943

### 7. Spectral-bias and frequency-principle literature (Rahaman et al., 2019; Xu et al., 2020)

- **Overlap**: Tangential
- **What they did**: Studied the tendency of neural networks to learn low-frequency components of functions over meaningful input coordinates before high-frequency components.
- **How it relates**: It provides broad context for frequency-dependent learning dynamics but does not analyze covariance eigendirections of gradients across optimization steps.
- **Key difference**: Function frequency, FFT coefficients of stored parameter arrays, and eigenvectors of a gradient covariance are distinct mathematical objects. These papers neither anticipate the proposed filter nor justify calling its retained directions “low frequency.”
- **Source**: [Rahaman et al.](https://proceedings.mlr.press/v97/rahaman19a.html); [Xu et al.](https://doi.org/10.4208/cicp.OA-2020-0085)

## Differentiation Analysis

No searched source combines all of the following:

1. the flagship Numerai Tournament on the current v5.3 data contract;
2. an MLP trained with a streaming parameter-space gradient-covariance eigenspace filter;
3. an honestly tuned AdamW control with equal, recorded data and selection budgets;
4. expanding walk-forward selection with target-horizon purges and a sealed final validation period;
5. official era-level CORR and BMC evaluation against version-matched Ender benchmark predictions; and
6. a frozen, unstaked forward-submission candidate, with live claims deferred until rounds resolve.

That combination is meaningfully differentiated because the domain poses non-trivial adaptation problems: millions of rows but relatively few era-level statistical units, overlapping 20D/60D labels, nonstationarity, benchmark-relative contribution, strict row/era alignment, and a historical target that is not identical to the current live payout target. A positive or negative result would add evidence about when persistent gradient directions help under weak, nonstationary financial signal.

The differentiation is nevertheless limited in three ways. First, applying an existing optimizer to a new dataset is incremental unless the protocol isolates a domain-relevant finding. Second, the existing optimizer evidence is mixed and predicts a plausible failure mode: weak useful gradients may not occupy the dominant persistent eigenspace, so filtering may suppress signal. Third, the literature search contains no direct paper on this exact covariance filter; absence from the located corpus is not proof that no unpublished code, workshop paper, or differently named subspace optimizer exists.

There is also an internal provenance issue that must be stated clearly. The older `research/findings.md` describes a different per-sample \(B \times B\) Gram-matrix consensus method and calls the streaming \(p \times p\) variant theoretical and unimplemented. The current README describes the streaming parameter-space implementation. The Numerai study must pin the exact implementation/commit and must not import claims or guarantees from the older per-sample method as though they applied to the current filter.

## Recommendations

- Frame the contribution as a controlled Numerai domain-transfer study of an existing covariance-eigenspace filter, not as invention of “spectral optimization.”
- Use mechanism-accurate language: “leading eigenspace of a streaming gradient-covariance operator,” not “frequency filtering,” “low-pass filtering,” or “FFT denoising.”
- Pin and document the exact optimizer implementation, because the repository's older findings concern a materially different \(B \times B\) per-sample method.
- Include controls that identify the value of the learned eigenspace: tuned AdamW, filter-strength zero, a rank-matched random subspace, and—if budget permits—a non-streaming or shuffled-history control. FFT-specific reshape and frequency-mask ablations are not relevant to this method.
- Make the primary scientific claim conditional on the predeclared paired holdout test. If the block-bootstrap interval for spectral-minus-AdamW includes zero, report a tie; if performance declines, report that dominant persistent gradient directions did not improve this weak-signal regime.
- Keep novelty and competitiveness separate. A novel evaluation can yield a noncompetitive model, and strong Diagnostics-compatible results still do not establish live leaderboard reputation without resolved forward rounds.
- Before any publication-level novelty claim, run a targeted follow-up search using mechanism terms such as “online PCA gradient projection,” “gradient covariance subspace optimization,” “streaming eigenspace gradient filtering,” “low-rank gradient covariance optimizer,” and “principal gradient subspace,” because the completed search was partly misdirected toward FFT methods.
