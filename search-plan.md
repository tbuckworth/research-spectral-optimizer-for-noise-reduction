# Search Plan

## Topic Summary
We are searching for prior work relevant to testing whether an existing per-sample gradient
consensus/covariance-filtering optimizer wrapper (a "coherence amplifier" built on Adam/AdamW,
using Marchenko-Pastur-style spectral thresholding of the inter-sample gradient similarity or
covariance matrix) improves out-of-sample performance when training MLP and recurrent/state-space
models on large, noisy financial time-series data such as the Numerai tournament dataset.

## Key Concepts
- **Per-sample gradient agreement/consensus filtering**: methods that compute or approximate
  per-example gradients within a batch and filter/reweight updates based on cross-sample
  agreement (e.g., coherent gradients, gradient sign agreement, "Agree-to-Disagree").
- **Random matrix theory (RMT) / Marchenko-Pastur denoising**: eigenvalue-threshold-based
  denoising of empirical covariance/correlation matrices, both classically in quantitative
  finance (portfolio return-covariance cleaning) and more recently applied to gradient or
  Hessian spectra in deep learning.
- **Spectral / covariance-based optimizers**: optimizers that use gradient covariance structure
  (rank-k, streaming, or full eigendecomposition) to precondition or filter updates, distinct
  from standard adaptive optimizers (Adam/AdamW, Shampoo, K-FAC, SOAP).
- **Noisy-label / noise-robust training**: methods for training under label noise or heavy
  target noise; relevant because financial returns are low signal-to-noise regression targets
  (feature/target noise rather than discrete label flips).
- **Low signal-to-noise-ratio (SNR) regression / financial ML**: challenges specific to
  training on data where the predictable signal is a small fraction of variance, and how
  optimizer/regularization choices interact with generalization in this regime.
- **Financial ML evaluation protocols**: era-based / purged / embargoed time-series
  cross-validation, walk-forward validation, and the Numerai tournament's specific
  obfuscated-feature, multi-era dataset design.
- **Architectures for noisy financial tabular/time-series prediction**: gradient-boosted trees
  as the dominant baseline, plus MLPs, LSTM/GRU, temporal convolutional networks, and
  state-space models (S4/Mamba) or transformers applied to financial or other low-SNR
  time-series tasks.
- **Optimizer benchmarking methodology**: how prior work fairly compares optimizers (matched
  compute/tuning budgets, multiple seeds, sensitivity to hyperparameters) to avoid confounds.

## Search Tasks

### Group 1: Academic Sources

#### arXiv API Queries
1. **Query**: `per-sample gradient consensus filtering optimizer noise robustness`
   - Categories: `cs.LG, stat.ML`
   - Date range: `2018-01-01` to `2026-07-31`
   - Max results: `30`
   - Rationale: Direct search for the core mechanism (consensus/agreement filtering of
     per-sample gradients) underlying the Spectral Optimizer.

2. **Query**: `coherent gradients deep learning memorization noisy labels`
   - Categories: `cs.LG, cs.AI`
   - Date range: `2018-01-01` to `2026-07-31`
   - Max results: `20`
   - Rationale: "Coherent Gradients" (Chatterjee & Zielinski) is the closest known prior
     mechanism explaining why gradient-agreement filtering suppresses noise fitting; find
     follow-ups and citing work.

3. **Query**: `Marchenko-Pastur random matrix theory gradient covariance neural network training`
   - Categories: `cs.LG, stat.ML, cond-mat.dis-nn`
   - Date range: `2015-01-01` to `2026-07-31`
   - Max results: `25`
   - Rationale: Find ML-side applications of RMT/Marchenko-Pastur denoising to gradient or
     Hessian spectra during training, as opposed to the classic finance covariance use.

4. **Query**: `random matrix theory denoising covariance financial portfolio returns`
   - Categories: `q-fin.PM, q-fin.ST, stat.ML`
   - Date range: `2010-01-01` to `2026-07-31`
   - Max results: `20`
   - Rationale: Establish the classical quant-finance RMT covariance-cleaning literature
     (Laloux/Bouchaud-style) that the Spectral Optimizer's mechanism is analogous to, applied
     to returns rather than gradients.

5. **Query**: `adaptive gradient methods low signal-to-noise ratio regression generalization`
   - Categories: `cs.LG, stat.ML`
   - Date range: `2019-01-01` to `2026-07-31`
   - Max results: `20`
   - Rationale: Broader query on optimizer behavior/generalization specifically in low-SNR
     regression settings, the regime financial return prediction falls into.

6. **Query**: `deep learning tabular financial time series prediction state space models transformers`
   - Categories: `cs.LG, q-fin.CP`
   - Date range: `2020-01-01` to `2026-07-31`
   - Max results: `25`
   - Rationale: Narrow query to identify standard/SOTA architectures used for noisy financial
     tabular/time-series prediction (needed to select the "recurrent/SOTA" comparison model).

7. **Query**: `Numerai tournament machine learning benchmark obfuscated financial features`
   - Categories: `q-fin.CP, cs.LG`
   - Date range: `2018-01-01` to `2026-07-31`
   - Max results: `15`
   - Rationale: Directly find academic treatments (if any) of the Numerai dataset and its
     era-based evaluation protocol, and papers benchmarking on it.

8. **Query**: `gradient noise stochastic gradient descent implicit regularization heavy-tailed`
   - Categories: `cs.LG, stat.ML`
   - Date range: `2017-01-01` to `2026-07-31`
   - Max results: `15`
   - Rationale: Adjacent field — theory of gradient noise's role in generalization, relevant
     to whether filtering out "noise" in gradients helps or removes beneficial implicit
     regularization.

#### Semantic Scholar API Queries
1. **Query**: `per-sample gradient agreement filtering optimizer robust training`
   - Fields: `title,abstract,year,authors,citationCount,url`
   - Max results: `20`
   - Rationale: Cross-check arXiv results with Semantic Scholar's citation graph for the core
     mechanism, to find highly-cited related work.

2. **Query**: `Marchenko-Pastur eigenvalue denoising covariance matrix estimation finance`
   - Fields: `title,abstract,year,authors,citationCount,url`
   - Max results: `20`
   - Rationale: Locate the seminal and highly-cited RMT covariance-cleaning literature in
     quantitative finance.

3. **Query**: `noisy label robust deep learning loss correction survey`
   - Fields: `title,abstract,year,authors,citationCount,url`
   - Max results: `20`
   - Rationale: Broad survey-level query on noise-robust training methods, to contextualize
     the Spectral Optimizer's MNIST label-noise result within the wider noise-robustness
     literature.

4. **Query**: `optimizer benchmark comparison deep learning fair evaluation hyperparameter tuning`
   - Fields: `title,abstract,year,authors,citationCount,url`
   - Max results: `15`
   - Rationale: Adjacent methodological query — how to run a fair optimizer comparison
     (matched budgets, multiple seeds), relevant to designing this project's own evaluation.

5. **Query**: `machine learning stock return prediction low signal to noise ratio deep learning`
   - Fields: `title,abstract,year,authors,citationCount,url`
   - Max results: `20`
   - Rationale: Find empirical financial-ML papers characterizing the SNR regime and what
     modeling/optimization choices matter for out-of-sample performance.

### Group 2: Lab Blog Sources
1. **Query**: `gradient noise robust training optimizer site:anthropic.com`
   - Rationale: Check whether Anthropic has published on gradient filtering/consensus methods
     or noise-robustness mechanisms relevant to training dynamics interpretability.
2. **Query**: `adaptive optimizer noisy data training robustness site:openai.com`
   - Rationale: OpenAI has published on optimizer scaling and training dynamics; check for
     relevant noise-robustness or gradient-statistics work.
3. **Query**: `gradient covariance random matrix theory training dynamics site:deepmind.google`
   - Rationale: DeepMind has published on optimizer research (e.g., Shampoo, distributed
     shampoo) and training dynamics theory; check for RMT/covariance-based methods.
4. **Query**: `per-sample gradients training dynamics circuits site:transformer-circuits.pub`
   - Rationale: Transformer Circuits publishes on training dynamics and mechanistic aspects of
     gradient-based learning; low-probability but worth checking for gradient-coherence framing.

### Group 3: Community Sources
1. **Query**: `spectral optimizer gradient consensus noise filtering site:lesswrong.com OR site:alignmentforum.org`
   - Rationale: Check whether this or a similar optimizer idea has been discussed/proposed on
     LessWrong/Alignment Forum (the prior project itself may have been posted there).
2. **Query**: `coherent gradients memorization generalization site:lesswrong.com OR site:alignmentforum.org`
   - Rationale: Search for community discussion of the "coherent gradients" mechanism and its
     implications for generalization/memorization, which underpins the Spectral Optimizer's
     theoretical motivation.
3. **Query**: `Numerai machine learning tournament optimizer noise site:lesswrong.com OR site:alignmentforum.org`
   - Rationale: The Numerai tournament has an active community of quant/ML practitioners who
     sometimes cross-post to LessWrong/AF; check for relevant empirical discussion.
4. **Query**: `random matrix theory neural network gradients site:intelligence.org OR site:alignment.org`
   - Rationale: Check MIRI/ARC-adjacent theoretical work for any treatment of RMT applied to
     learning dynamics (low prior probability but cheap to check).
5. **Query**: `Numerai tournament forum optimizer feature noise era`
   - Rationale: Not a lab/alignment source, but the Numerai community forum
     (forum.numer.ai) is the primary source of practitioner knowledge on what works for this
     specific dataset (era-based validation, denoising features, model architectures); include
     as a targeted community query since this is the candidate dataset.

## Expected Coverage
This plan should surface: (1) the academic lineage of per-sample gradient agreement/coherence
methods and their theoretical framing (memorization vs. generalization); (2) the classical and
ML-adjacent RMT/Marchenko-Pastur denoising literature in both finance and deep learning; (3)
noisy-label/noise-robust training as a broader comparison class; (4) standard architectures and
evaluation protocols for noisy financial time-series/tabular prediction, including Numerai
specifics; and (5) any direct prior discussion of this exact optimizer or closely related ideas
in the AI safety community (low expected hits, since this is a general ML optimizer question
rather than a safety topic per se — the AI-safety framing here is secondary/methodological).
Likely gaps: peer-reviewed academic work specifically combining per-sample gradient RMT
filtering with financial time-series prediction is unlikely to exist (this appears to be a novel
combination), so the literature review will mostly establish adjacent building blocks rather than
a directly on-topic prior study; the search should confirm this novelty rather than assume it.
Practitioner-level Numerai forum knowledge (feature neutralization, era-boosting, denoising
tricks) may only be partially indexed by academic/lab/community searches and may need a
dedicated web search pass outside the standard site: filters.
