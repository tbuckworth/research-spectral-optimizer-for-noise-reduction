# Search Results: Lab Blog Sources

## Search Queries Executed
1. `gradient noise robust training optimizer site:anthropic.com` — 0 directly relevant Anthropic-authored results (returned mostly off-site arXiv papers and unrelated Anthropic engineering/alignment posts)
2. `adaptive optimizer noisy data training robustness site:openai.com` — 0 directly relevant OpenAI-authored results on this exact query (returned parameter-noise RL post and unrelated API docs)
3. `gradient covariance random matrix theory training dynamics site:deepmind.google` — 1 tangentially relevant result (ICLR 2018 papers roundup page mentioning K-FAC for RNNs)
4. `per-sample gradients training dynamics circuits site:transformer-circuits.pub` — 0 relevant results (no matching content found on transformer-circuits.pub itself)
5. `gradient noise scale large batch training site:openai.com` — 1 highly relevant result (follow-up query, "How AI Training Scales")
6. `Shampoo distributed preconditioning optimizer site:deepmind.google` — 1 tangentially relevant result (follow-up query, DiLoCo/Shampoo ecosystem, mostly off-site)

## Key Findings

### How AI Training Scales (gradient noise scale)
- **Source**: https://openai.com/index/how-ai-training-scales/ (companion paper: arXiv:1812.06162, "An Empirical Model of Large-Batch Training")
- **Authors**: Sam McCandlish, Jared Kaplan, Dario Amodei, OpenAI Dota Team (OpenAI)
- **Date**: 2018-12 (blog), paper same period
- **Summary**: Introduces the "gradient noise scale," a statistic quantifying the signal-to-noise ratio of gradients (ratio of gradient variance across examples to the squared gradient mean), and shows it predicts the critical/maximum useful batch size for a given task (r^2 = 80% across tasks). Noisier gradients (as in complex or low-SNR tasks) permit/benefit from larger batch sizes before returns to parallelism diminish.
- **Key Claims**:
  - Gradient noise scale is estimated from the ratio of per-example gradient variance to the true gradient magnitude — directly related to the per-sample gradient statistics the Spectral Optimizer's covariance/consensus filtering also operates on.
  - Complex/noisy tasks have inherently higher gradient noise scale, which the paper treats as a batch-size design parameter rather than something to filter/threshold at the update level.
  - No mention of spectral/eigenvalue-based filtering of the gradient covariance matrix, per-sample agreement/consensus reweighting, or Marchenko-Pastur-style denoising — the paper characterizes noise, it does not propose removing structured "noise" via RMT thresholding.
- **Relevance**: MEDIUM — establishes the standard OpenAI framing of "gradient noise" as a scalar SNR statistic tied to batch size, which is a different mechanism from the Spectral Optimizer's cross-sample covariance eigenspectrum filtering, but is the closest lab-blog analogue found and useful for framing/contrast in the introduction/related work.
- **Cite Key**: mccandlish2018gradientnoisescale

### Kronecker-Factored Curvature Approximations for Recurrent Neural Networks (via DeepMind ICLR 2018 roundup)
- **Source**: https://deepmind.google/discover/blog/deepmind-papers-at-iclr-2018 (paper itself: Martens, Ba & Johnson, ICLR 2018)
- **Authors**: James Martens, Jimmy Ba, Matthew Johnson
- **Date**: 2018 (ICLR 2018; DeepMind blog roundup same year)
- **Summary**: Extends Kronecker-Factored Approximate Curvature (K-FAC) second-order optimization to RNNs by modeling the covariance of gradients across timesteps with a chain-structured linear Gaussian graphical model, outperforming SGD-with-momentum and Adam baselines on RNN training.
- **Key Claims**:
  - Demonstrates that explicitly modeling gradient covariance structure (across time, in this case, rather than across samples) can meaningfully improve optimizer performance for recurrent architectures — relevant precedent for the recurrent/state-space model arm of the planned experiments.
  - Uses a structured covariance approximation (Kronecker-factored) rather than an RMT/Marchenko-Pastur eigenvalue threshold; the covariance is used for curvature-based preconditioning, not for filtering/rejecting noisy per-sample gradient contributions.
- **Relevance**: LOW-MEDIUM — same broad family (gradient-covariance-informed optimization for RNNs) but a different mechanism and different target (curvature preconditioning vs. noise-sample filtering); useful for related-work framing on covariance-based optimizers for recurrent models.
- **Cite Key**: martens2018kfacrnn

## Themes Identified
- **Gradient noise as a scalar SNR quantity (OpenAI framing)**: OpenAI's gradient-noise-scale work treats gradient noise as an aggregate statistic for batch-size selection, not as a structure to be spectrally filtered — contrasts with the Spectral Optimizer's approach of decomposing the full per-sample gradient covariance/similarity matrix and thresholding its eigenspectrum.
- **Covariance-structured optimizers exist but for curvature, not denoising**: The DeepMind K-FAC-for-RNNs result shows labs have explored gradient covariance structure for optimization, but consistently for second-order curvature preconditioning (Shampoo, K-FAC family) rather than Marchenko-Pastur-style noise rejection or per-sample consensus filtering.
- **Absence of direct engagement**: None of the four major AI lab blogs (Anthropic, OpenAI, DeepMind, Transformer Circuits) appear to have published anything combining per-sample gradient consensus/agreement filtering with random-matrix-theory spectral thresholding — the core mechanism under study here is not represented in lab-blog literature.

## Gaps
- No lab blog post was found describing per-sample gradient agreement/consensus filtering as an update rule (the core mechanism of the Spectral Optimizer).
- No lab blog post was found applying Marchenko-Pastur or other random-matrix-theory eigenvalue thresholding to gradient or Hessian covariance matrices during neural network training.
- No lab blog content was found on financial time-series prediction, Numerai, or noisy-label/noisy-target regression specifically — these topics are outside the typical scope of Anthropic/OpenAI/DeepMind/Transformer-Circuits public blogs, as expected.
- Transformer Circuits Thread yielded no relevant content at all for this topic; its focus (mechanistic interpretability of trained models) does not overlap with optimizer-level gradient covariance filtering.
- This absence is consistent with the search plan's expectation that the exact combination of ideas is unlikely to have prior lab-blog treatment; it supports (but does not on its own confirm) the novelty of the core mechanism.
