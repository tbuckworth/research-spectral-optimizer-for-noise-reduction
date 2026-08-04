---
is_followup: true
parent_issue: 2073
prior_repo: https://github.com/tbuckworth/research-spectral-optimizer-for-noise-reduction
prior_run_id: 2026-07-31-spectral-optimizer-for-noise-reduction-on-financial-timeseri
---

## User Feedback

Spectral Optimizer redo: pxp filter, walk-forward split, rank sweep

The parent run answered the motivating question rigorously and got a clean,
audited negative — but it used the wrong optimizer and a broken temporal split,
so the answer does not count. This run redoes it correctly. Reuse the parent's
code wherever possible; almost all of the infrastructure is good.

Two corrections are mandatory and non-negotiable. Everything else in this brief
exists to serve them.


## CORRECTION 1 — Use the p x p Spectral Optimizer, not the B x B variant

The Spectral Optimizer is `SpectralGradientFilter`, defined at the REPO ROOT of
`~/pyg/optimizers` in `spectral_filter.py`. It keeps a streaming rank-k
factorization of the **p x p gradient covariance** (p = number of parameters),
updated by a rank-1 SVD each step, and projects the batch-mean gradient onto its
top-k eigendirections. It never materializes p x p; only a tiny (k+1)x(k+1) eigh
runs on CPU. Cost is about 2x a bare Adam step.

    from spectral_filter import SpectralGradientFilter
    base_opt = torch.optim.AdamW(model.parameters(), lr=..., weight_decay=...)
    filt = SpectralGradientFilter(model, base_opt, rank=200)
    ...
    base_opt.zero_grad()
    loss = loss_fn(model(x), y)
    loss.backward()
    filt.filter_grad()      # filters .grad in place
    base_opt.step()

Before doing anything else: `git -C ~/pyg/optimizers pull`, then assert that
`~/pyg/optimizers/spectral_filter.py` exists and exposes `SpectralGradientFilter`
with a `filter_grad()` method. The local checkout is on branch `main` tracking
`origin/research/literature-review`. If that file is absent, STOP and report —
do not substitute another file.

DO NOT USE `experiments/spectral_optimizer.py` (`SpectralConsensusFilter`). That
is the per-sample **B x B** inter-sample-similarity variant. The parent run used
it by filename match and froze the choice as amendment F11. Two independent
reasons it is the wrong object:

  - The repo README defines the core as the p x p estimator and lists the
    per-sample rank-B object as a *variant* whose hypothesis H4 had ALREADY
    FAILED in prior work ("Never groks — top direction is approximately the
    bulk-fitting mean").
  - The parent run then proved (exp-001, verified fp64 to 8.9e-16, independently
    re-verified in audit) that for scalar-output MSE the B x B variant's
    engagement eigenspectrum is EXACTLY target-independent, because row
    normalization reduces each per-sample gradient to sign(residual) x Jacobian
    direction. That proof is correct and it is fatal — for that variant.

That theorem does NOT bind `SpectralGradientFilter`: it never row-normalizes
per-sample gradients and never forms a B x B matrix. Do not re-derive it, do not
apply it here, and do not treat the parent's negative as evidence about this
optimizer. Cite the parent's exp-001 result only as a scoped side-note about the
per-sample variant.

Relevant knobs of `SpectralGradientFilter.__init__` (read the file for the full
list and the inline commentary, which is extensive and accurate):
  rank              hard cap on kept eigendirections (default 200)
  decay             EMA decay of the covariance estimate (default 0.99)
  warmup            steps observed before filtering starts (default 100)
  weighting         "hard" top-k projection, or "soft" eigenvalue^alpha reweighting
  alpha             soft exponent; alpha=1 means proportional to agreement
  soft_residual     True keeps the out-of-basis component at weight 1
  energy_threshold  keep smallest top-k capturing this fraction of spectral energy
  adaptive          rank rule: "none", "effrank", or "gap"
  normalize         basis: "none" (covariance), "var", "degree"

Use `normalize="none"`: prior finding H7 recorded that raw covariance beats the
var/degree normalizations (92.1% vs 86.6/86.0 on MNIST 50% noise).

Free integration test, use it: `weighting="soft", alpha=0, soft_residual=True`
is by construction a genuine no-filter identity. It must reproduce the plain
AdamW arm bit-for-bit (fp64) or near enough to be indistinguishable in fp32.
Assert this before trusting any comparison.


## CORRECTION 2 — Fix the temporal protocol

The parent run trained on eras 0425-0574 and tested on eras 0971-1225. At the
recorded 5-day era spacing that is a ~5.4-year gap between the last training era
and the first test era, widening to ~8.9 years by the end of the test block. The
388-era tuning block sitting between them was used only for hyperparameter
selection and never trained on. Only 150 of 574 available training eras were used.
There was no refit before the final evaluation. Result: a tuned-AdamW baseline at
+0.0064 mean per-era corr where Numerai's own example model scores +0.0235 on the
same eras. The parent's own pre-registered sanity band was [0.0087, 0.0522] and
the realized baseline fell BELOW it; that was carried as "regime drift" rather
than treated as a broken baseline. A -0.005 effect measured against a +0.0064
edge is not an answer to anything.

Required protocol: expanding-window walk-forward, with a refit before test.

For each fold:
  1. TRAIN   = all usable eras from the start of the data up to T
  2. embargo of E eras
  3. VALID   = the next V eras (hyperparameter selection ONLY)
  4. embargo of E eras
  5. TEST    = the next S eras
  6. REFIT the model from scratch, at the hyperparameters selected on VALID,
     on TRAIN + embargo + VALID (i.e. every usable era up to E eras before
     TEST begins)
  7. Evaluate the refit model on TEST

E = ceil(target_horizon_days / era_spacing_days) = ceil(20/5) = 4, which is what
the parent computed and is correct — keep it.

Run 3 folds. Choose T for each fold so the three TEST blocks are contiguous,
non-overlapping, and collectively cover the most recent portion of the data.
Target roughly V ~= 96 and S ~= 110 eras, but compute exact boundaries from the
realized usable-era list (target coverage >= 0.95) rather than hardcoding, and
write the realized boundaries to `protocol.json`.

THE SINGLE MOST IMPORTANT ASSERTION IN THIS RUN — put it in the code, make it a
hard assert, and print it in the log for every fold:

    assert min(test_eras) - max(refit_train_eras) == E + 1

With E=4 that value is exactly 5. This is what prevents the parent's failure from
recurring. A run that reports results without this assertion having fired is not
answering the question.

Also required:
  - Use ALL usable eras up to each fold's boundary for training. Do not subsample
    eras for the headline runs.
  - Use the full v5.0 feature set, not the 705-feature "medium" subset, unless
    VRAM genuinely forces a reduction — in which case document the constraint and
    report it as a limitation.
  - Train long enough to converge on the larger training set. The parent used
    2000 steps at B=1024 on 384k rows; the full training set is far larger, so
    step count must be re-tuned, not inherited.


## BASELINE GATE (hard gate — do not skip)

Numerai ships example-model predictions for the validation eras:
`data/v5.0_validation_example_preds.parquet`. For each fold, compute the example
model's mean per-era numerai_corr restricted to that fold's TEST eras.

GATE: the tuned-AdamW baseline must reach >= 0.60 x the example model's mean
per-era corr on the same eras.

If the gate fails on a fold, DO NOT proceed to the optimizer comparison on that
fold. Spend remaining budget fixing the baseline — more features, more steps,
larger network, learning-rate schedule, target transform, better regularization —
and re-check. A broken baseline makes the comparison uninterpretable in both
directions, which is precisely how the parent run lost its budget. Report the
gate outcome explicitly either way, with the realized ratio.


## ARMS

  A. Tuned AdamW (control).
  B. AdamW + SpectralGradientFilter (the treatment), rank swept — see below.
  C. Norm- and k-matched RANDOM-SUBSPACE control. Project onto a random
     orthonormal subspace with the same kept-k trajectory k(t) and the same
     update-norm ratio trajectory as the same-seed arm B run. This is the best
     artifact the parent run produced (`exp-004/src/`) and it is what
     distinguishes "spectral selection matters" from "any low-rank projection
     does this". Port it, do not drop it.
  D. Sanity: alpha=0 identity check (above) and a seeded zero-predictor control.
     Not arms — assertions.


## RANK SWEEP (do not freeze a single operating point)

The parent run's fatal process error was selecting one filter configuration from
a diagnostic proxy, never from performance, and then freezing it. Do not repeat
this.

Sweep the number of retained eigendirections on VALID:
  - fixed rank over a log grid, e.g. {8, 32, 128, 512, 2048}, capped at what is
    feasible for p
  - the adaptive rules: adaptive="effrank" and adaptive="gap"
  - energy_threshold in {0.90, 0.99}
  - weighting="soft" with alpha in {0.5, 1, 2}

This is well-motivated by prior findings, not just prudence: H3 recorded that
rank <= 4 destabilizes and ~10 was fastest on sparse parity, and H5 recorded that
adaptive effective-rank was the only variant that groks parity — "adaptive
spectral rank helps iff the task's solution is genuinely low-dimensional."

MATCHED TUNING BUDGET, honestly matched this time. Give arm B the SAME number of
trials as arm A — 12 each is the parent's precedent. Arm B's search space MUST
include the learning rate, because filtering changes the effective step size.
The parent gave the filter arm ~1 trial against AdamW's 12 (because the B x B
variant cost 21x per step), which forced a mandatory downgrade of the claim to
"no evidence of benefit under the affordable tuning budget." The p x p filter is
~2x Adam, so matched-budget tuning is affordable for the first time and there is
no excuse for an asymmetric budget. Report measured seconds/step for both arms.

ALL selection happens on VALID. The TEST block is touched exactly once per fold,
after the refit.


## METRICS AND INFERENCE

  - Primary: mean per-era numerai_corr on TEST.
  - Secondary: mean per-era Spearman; corr-Sharpe (mean/sd across eras).
  - >= 3 seeds per arm per fold, paired: same seed and same data order across
    arms so differences are paired per era.
  - Moving-block bootstrap for CIs, block length L from the lag-1 autocorrelation
    of the per-era corr series (the parent measured lag-1 ACF 0.247 -> L=4 on the
    tuning block, and 0.763 across all validation eras — recompute per fold, do
    not inherit).
  - Report the paired difference (B - A) per fold with 95% CI, and the same for
    (B - C) and (C - A).

Note on the parent's corr-Sharpe: its MLP Sharpe CI was found by the auditor to
be bootstrap-RNG fragile (upper bound straddled zero across RNGs). Report Sharpe
CIs from a fixed, stated RNG seed and check stability across at least two seeds
before making any stability claim.


## DECISION RULE

  WINS   if (B - A) > 0 with 95% CI excluding zero on >= 2 of 3 folds, same sign.
  HURTS  if (B - A) < 0 with 95% CI excluding zero on >= 2 of 3 folds, same sign.
  NULL   otherwise.

Report effect sizes with CIs regardless of category. If B beats A but B is
statistically indistinguishable from C (the random-subspace control), the honest
conclusion is "low-rank projection helps, spectral selection is not the active
ingredient" — say that, do not claim the spectral mechanism.


## KILL CRITERION

If the baseline gate passes, the rank sweep is genuinely executed at matched
budget, and the result is NULL or HURTS on >= 2 of 3 folds: the line is dead for
financial tabular regression. Write it up as a clean negative and STOP. Do not
manufacture a fourth epicycle, do not propose another dataset, and do not
recommend a full-scale rerun to rescue it.


## PROCESS: EXPLORATORY, NOT FROZEN

Pre-register the PROTOCOL — splits, embargo, metric, bootstrap, decision rule,
baseline gate — and freeze those before looking at any TEST data. Do NOT
pre-register a single filter operating point. Rank is a tuned hyperparameter
selected on VALID, not a frozen constant. The parent run's pre-registration
machinery was good and produced a trustworthy result; it just froze a bad choice.
Keep the machinery, move the freeze line.


## REUSE (do not rebuild these)

Parent run directory on local disk:
  /media/titus/big/researcher-output/2026-07-31-spectral-optimizer-for-noise-reduction-on-financial-timeseri/
Parent repo:
  https://github.com/tbuckworth/research-spectral-optimizer-for-noise-reduction

Already downloaded, ~6.3 GB, DO NOT RE-DOWNLOAD — copy or symlink from the parent
run's `data/`:
  v5.0_train.parquet, v5.0_validation.parquet,
  v5.0_validation_example_preds.parquet, v5.0_features.json

Reusable code (parent repo, committed under experiments/*/src/):
  exp-001/src/download_numerai.py     Numerai v5 download
  exp-001/src/data_prep.py            shard building, era-purge/embargo arithmetic
  exp-003/src/build_tuning_shard.py   shard construction per block
  exp-003/src/sweep.py                AdamW hyperparameter sweep harness
  exp-004/src/build_verdict_shard.py  eval shard construction
  exp-004/src/                        C4 random-subspace control, per-era eval,
                                      moving-block bootstrap, analysis
  audit/rerun-exp-004/                independent bootstrap + re-derivation code

Note: the follow-up scaffold copies only `results.md` and `plan.md` into
`prior/experiments/`. Clone the parent repo or read the local run directory to
get `src/`.


## NUMERICAL ROBUSTNESS

The parent's audit found (Finding 5) that `torch.linalg.eigh` fails to converge
stochastically by seed in fp32 on cuSOLVER under rank collapse — the original
runs were 6-of-6-seed lucky-complete. `SpectralGradientFilter` runs a
(k+1)x(k+1) eigh each step. Wrap it in a CPU-fp64 fallback on
`torch._C._LinAlgError` from the start, log every time the fallback fires, and
report the count. A working patch exists at
`audit/rerun-exp-004/src/` in the parent repo.


## SCOPE — KEEP IT SIMPLE

  - MLP is the primary and probably only architecture. A second architecture is
    optional and only if budget remains after folds are complete.
  - DO NOT rebuild the parent's GRU arm. It reshaped tabular rows into fake
    sequences and inherited untuned MLP hyperparameters; the parent's own
    limitations section concedes it is a "reshaped-tabular construct". A sequence
    model is only meaningful on genuinely sequential data (OHLCV), which is out
    of scope here.
  - The garbled "Stadia / RushCursive models" in the original issue was
    speech-to-text noise for "state-of-the-art / recurrent architectures". It
    does not require a recurrent model. Ignore it for this run.
  - No live leaderboard submission. Held-out walk-forward TEST blocks are the
    leaderboard equivalent.


## COMPUTE

MATS Slurm free `compute` partition, 1x L40 per job, jobs under ~30 GPU-min,
max 5 experiments. This fits: the p x p filter is ~2x Adam, AdamW training runs
were ~3 s each in the parent run, and the largest cost is the widened feature set
and longer schedules. If a fold genuinely will not fit, reduce the number of
seeds before reducing the number of folds, and say so.


## WHAT SUCCESS LOOKS LIKE

A verdict on the original question — does the Spectral Optimizer improve
out-of-sample performance versus tuned AdamW on large, noisy financial data —
delivered against a baseline that demonstrably works, with the filter's rank
tuned rather than guessed, on a split where the test period begins the week after
training ends. A well-evidenced negative is as valuable as a positive, but only
if the baseline gate passed and the rank sweep actually happened.
