# Proposed Next Steps: Spectral Optimizer (for noise reduction) on Financial Timeseries Data

**Run ID**: 2026-07-31-spectral-optimizer-for-noise-reduction-on-financial-timeseri
**Outcome**: Rigorous, audited negative result — on the wrong optimizer, against a baseline too weak to have detected a positive one.

## Reflection

This round executed a clean protocol (era-purged blocks, frozen thresholds,
moving-block bootstrap, an independent audit that re-executed the load-bearing
experiment on fresh seeds and a self-built eval shard) and returned a decisive
negative: the filter degraded mean per-era correlation by −0.00527
(CI [−0.00886, −0.00181]) on an MLP, replicated on a GRU.

The blocker was **scientific-adjacent but escapable**, and it has two distinct parts.

**(1) Wrong optimizer.** The brief said "reuse the existing Spectral Optimizer."
The run used `experiments/spectral_optimizer.py` — `SpectralConsensusFilter`,
which eigendecomposes the **B×B inter-sample** gradient similarity matrix. The
actual Spectral Optimizer is the streaming rank-k estimator of the **p×p
gradient covariance**, now at repo root as `spectral_filter.py`
(`SpectralGradientFilter`). This was not a stale-checkout problem: the README
the run read opens by defining the core as the p×p estimator and lists the
per-sample rank-B object as a *variant* whose hypothesis (H4) had **already
failed** in prior work, with the diagnosis "top direction ≈ the bulk-fitting
mean." The run summarized both files correctly in `state.md` and then selected
on filename match, freezing the choice as amendment F11.

Consequently exp-001's headline contribution — the proof that for scalar-output
MSE the engagement eigenspectrum is exactly target-independent — is a true and
well-verified statement **about the wrong object**. It is a boundary result for
the per-sample B×B variant, not for the Spectral Optimizer. It does not bind
`SpectralGradientFilter`, which never row-normalizes per-sample gradients.

**(2) Broken temporal protocol.** The model trained only on eras 0425–0574 and
was evaluated on eras 0971–1225. At the recorded 5-day era spacing that is a
**~5.4-year gap** between the end of training and the start of testing, widening
to ~8.9 years by the end of the test block. The 388-era tuning block sitting
between them was used solely to pick hyperparameters and was **never trained
on**, and only 150 of 574 available training eras were used at all. There was no
refit before the final evaluation. The result is a baseline at +0.0064 mean
per-era corr where Numerai's own example model scores +0.0235 on the same eras —
27% of it. The run's own pre-registered sanity band was [0.0087, 0.0522] and the
realized baseline fell **below** it; this was recorded as "regime drift" and
carried as a limitation rather than treated as a broken baseline.

A −0.005 degradation measured against a +0.0064 edge, using an optimizer that
prior work had already recorded as failing, cannot answer the motivating
question in either direction.

Everything else built this round is sound and reusable: the Numerai download and
shard pipeline, era-purge/embargo arithmetic, per-era correlation and
zero-predictor sanity controls, the moving-block bootstrap, and — the single
best artifact of the run — the norm- and k-matched **random-subspace control**,
which is exactly the control needed to distinguish "spectral selection matters"
from "any low-rank projection does this."

## Ranked Next Steps

### 1. Re-run the actual experiment: p×p `SpectralGradientFilter`, walk-forward protocol, swept rank
- **Do**: Use `SpectralGradientFilter` from `~/pyg/optimizers/spectral_filter.py`.
  Replace the single train/test split with expanding-window walk-forward folds in
  which the test block begins immediately after the training data ends (embargo
  only), refitting on train+valid at the selected hyperparameters before testing.
  Gate on a baseline that reaches a defined fraction of Numerai's example model
  on the same eras. Sweep the number of retained eigendirections rather than
  freezing one operating point, with the filter arm receiving the same number of
  tuning trials as AdamW.
- **Tests**: The original motivating question — does the Spectral Optimizer
  improve out-of-sample performance vs tuned AdamW on large, noisy financial
  data, and is the effect consistent across rank?
- **Needs**: MATS free `compute` partition, ~5 jobs under 30 GPU-min each. The
  p×p filter costs ~2× Adam (not the 21× the B×B variant cost), so matched-budget
  tuning is affordable for the first time. Raw parquets already on local disk.
- **Kills the idea if**: with the baseline gate passed and rank properly swept,
  the best valid-selected filter configuration fails to beat AdamW on test in
  ≥2 of 3 folds. That is a real negative for financial tabular regression and
  the line should stop there.

### 2. Loss classes where the per-sample variant's theorem does not bind
- **Do**: Keep the B×B `SpectralConsensusFilter` but move off scalar-output MSE
  to a multi-output head, ranking loss, or batch-correlation loss.
- **Tests**: Whether exp-001's target-independence result is the whole story for
  the per-sample variant, or an artifact of the loss.
- **Needs**: ~10 GPU-min.
- **Kills the idea if**: engagement becomes target-dependent but performance is
  still indistinguishable from the random-subspace control.
- **Note**: strictly lower priority than #1 — it perfects a variant that prior
  work already recorded as failing.

### 3. Full-scale verdict at B≈4096 on non-subsampled data
- **Do**: The prior run's own W1 future-work item.
- **Needs**: 10–20 L40-hours or A100-class; exceeds the free partition.
- **Kills the idea if**: n/a — only worth doing after #1 shows something.

## Not Worth Pursuing

- Re-confirming the exp-004 negative: already re-executed by the audit on fresh
  seeds and an independent shard; it reproduces, and it is a fact about the
  per-sample variant on a broken split.
- The GRU arm as built: reshaped tabular rows are not a sequence task, and the
  arm inherited untuned MLP hyperparameters. A second architecture is only
  meaningful on genuinely sequential data (OHLCV).
- Tightening the C4 equivalence resolution below ±0.004: measures the wrong
  optimizer more precisely.
