# Follow-Up Summary

**This run**: 2026-08-03-followup-follow-up-spectral-optimizer-redo-pxp-filter-walk-
**Parent run**: 2026-07-31-spectral-optimizer-for-noise-reduction-on-financial-timeseri (issue #2073)
**Parent repo**: https://github.com/tbuckworth/research-spectral-optimizer-for-noise-reduction
**Parent run dir (local, full artifacts incl. `experiments/*/src/` and 6.3 GB `data/`)**:
`/media/titus/big/researcher-output/2026-07-31-spectral-optimizer-for-noise-reduction-on-financial-timeseri/`

## What the parent run did

Asked whether the Spectral Optimizer improves out-of-sample performance vs tuned
AdamW on large, noisy financial data (Numerai v5). Ran a fully pre-registered,
independently audited protocol and returned a decisive, reproducible negative:
filter −0.00527 mean per-era corr (CI [−0.00886, −0.00181]) on an MLP, replicated
on a GRU; MP-eigenselection statistically indistinguishable from a norm/k-matched
random-subspace projection; plus a verified proof that for scalar-output MSE the
B×B variant's engagement eigenspectrum is exactly target-independent.

## Why the answer does not count (the two defects this run corrects)

1. **Wrong optimizer.** The parent used `experiments/spectral_optimizer.py`
   (`SpectralConsensusFilter`, the per-sample **B×B** variant — whose hypothesis
   H4 had already failed in prior work) instead of the actual Spectral Optimizer,
   `SpectralGradientFilter` in `~/pyg/optimizers/spectral_filter.py` (streaming
   rank-k **p×p** gradient-covariance filter, ~2× Adam cost). The parent's
   target-independence theorem is true but binds only the B×B variant (it depends
   on row normalization); it does NOT transfer to `SpectralGradientFilter` and
   must not be re-derived or cited as evidence against it.
2. **Broken temporal protocol.** Trained on eras 0425–0574, tested on 0971–1225
   (~5.4–8.9-year gap, no refit, 150/574 training eras used). Baseline reached
   +0.0064 vs the Numerai example model's +0.0235 on the same eras — below the
   parent's own pre-registered sanity band. A −0.005 effect against a +0.0064
   edge answers nothing.

## What this run does instead (mandatory corrections from the brief)

- **Optimizer**: `SpectralGradientFilter` (verified present at
  `~/pyg/optimizers/spectral_filter.py`, repo pulled 2026-08-03, class at line 52,
  `filter_grad()` at line 303). `normalize="none"` per prior finding H7.
  Free identity check: `weighting="soft", alpha=0, soft_residual=True` must
  reproduce plain AdamW. CPU-fp64 eigh fallback from the start (parent audit
  Finding 5: fp32 cuSOLVER eigh fails stochastically under rank collapse).
- **Protocol**: 3-fold expanding-window walk-forward, embargo E=4 eras, refit on
  train+embargo+valid before each TEST; hard assert
  `min(test_eras) - max(refit_train_eras) == E + 1` (== 5) printed per fold;
  realized boundaries to `protocol.json`. All usable eras, full v5.0 feature set,
  step count re-tuned.
- **Baseline gate (hard)**: tuned AdamW must reach ≥ 0.60× the Numerai example
  model's mean per-era corr on each fold's TEST eras, or the comparison does not
  run on that fold.
- **Arms**: A tuned AdamW; B AdamW+SpectralGradientFilter with rank swept on
  VALID (fixed log grid, adaptive effrank/gap, energy_threshold, soft alpha);
  C norm/k-matched random-subspace control (port from parent `exp-004/src/`);
  D sanity assertions (alpha=0 identity, zero-predictor).
- **Matched tuning budget**: 12 trials per arm, arm B's space includes LR.
- **Inference**: mean per-era numerai_corr primary, ≥3 paired seeds/arm/fold,
  moving-block bootstrap (L recomputed per fold), (B−A), (B−C), (C−A) with 95% CIs.
- **Decision rule**: WINS/HURTS = CI excluding zero same-sign on ≥2 of 3 folds;
  else NULL. Kill criterion: gate passed + real rank sweep + NULL/HURTS on ≥2
  folds ⇒ the line is dead for financial tabular regression; write the negative
  and stop.
- **Scope**: MLP only (no GRU rebuild); no live submission; pre-register the
  protocol, NOT a single filter operating point (rank is tuned on VALID).

## Artifacts reused vs regenerated

| Artifact | Disposition |
|---|---|
| `literature/` (synthesis + 3 searches) | **Copied from parent** — framing unchanged; searches remain valid |
| `novelty-assessment.md` | **Copied from parent** — NOVEL verdict carries; contribution is still the transfer verdict |
| `references.bib`, `citation-registry.md` | **Copied from parent run dir** |
| `success-criteria.md` | **REGENERATED** (Step 4 runs): parent criteria are anchored to the B×B variant + broken split, and Step 10 uses this file as the frozen audit anchor — reusing it would audit this run against the defects it exists to fix. The brief's baseline gate, decision rule, and walk-forward protocol are the new criteria's core. |
| `decomposition.md` | **REGENERATED** (Step 5) — new arms, folds, rank sweep |
| `challenge/` | **REGENERATED** (Step 6) — must target the new design's risks |
| Experiment code | Reuse parent repo `experiments/*/src/` (data prep, sweep harness, C4 random-subspace control, bootstrap, eigh-fallback patch in `audit/rerun-exp-004/src/`); `prior/` here holds only plan/results.md — clone the repo or read the parent run dir for src |
| Data (~6.3 GB) | **Do not re-download** — copy/symlink from parent `data/`: v5.0_train.parquet, v5.0_validation.parquet, v5.0_validation_example_preds.parquet, v5.0_features.json |

## Notes for downstream steps

- Parent `prior/knowledge/` does not exist (parent predates repo KBs); this run
  has no global knowledge base (`knowledge_base: none`) — skip KB actions silently.
- Parent's `next-steps.md` ranked step #1 is exactly this brief; its "Not Worth
  Pursuing" list (re-confirming the exp-004 negative, the GRU-as-built arm,
  tightening C4 equivalence on the wrong optimizer) remains binding.
- Cite the parent's target-independence proof only as a scoped side-note about
  the per-sample B×B variant.
