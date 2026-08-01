# exp-004: Seeded main comparison + mechanism controls (the verdict experiment)

**Component**: the main deliverable — returns the verdict on the motivating
question. Builds on exp-001 (#1 diagnostics + F2/F6 registrations), exp-003
(#8 tuned baseline, frozen F3, F4 GO, budget packing).
**Fail semantics**: NOT a retry loop. No-verdict → honestly-scoped effect
estimate with calibrated uncertainty (pre-authorized, priced outcome).

## AMENDED INTERPRETATION FRAME (binding, from exp-001's result)

exp-001 PROVED that for scalar-output MSE the row-normalized per-sample
gradient similarity matrix satisfies S(y′) = D·S(y)·D (D orthogonal
diagonal), so the filter's eigenselection — and every engagement diagnostic —
is exactly target-independent at fixed parameters. Consequences for this
experiment:
1. The comparison is still the pre-registered verdict on the motivating
   question ("does the Spectral Optimizer improve OOS performance vs tuned
   AdamW on noisy financial data?"), and it remains genuinely uncertain.
2. BUT any verdict must be reported as the effect of **target-blind spectral
   subspace regularization** (Jacobian/factor-structure selection), NOT
   "gradient consensus on signal". The noise-consensus mechanism story is
   ruled out by proof for this setting.
3. The C4 random-subspace norm-matched control is therefore load-bearing:
   it asks whether MP-eigenselection does anything a random subspace at the
   same kept-norm fraction doesn't. Do not cut it (F7 protects it).
4. Engagement diagnostics are still logged during verdict runs — they verify
   the filter is doing sustained selective work (not a no-op), which is what
   a null verdict needs; they can no longer certify target-sensitivity.

## Frozen inputs (reuse verbatim — do NOT re-derive)

- **F1 blocks** (exp-001/out/protocol.json): tuning eras 0579–0966 (388),
  embargo 0967–0970, VERDICT eras 0971–1225 (255). This experiment performs
  the FIRST and ONLY unblinding of the verdict block. Nothing is evaluated on
  it until all arms/configs are frozen; no config may change afterwards.
- **F2** (exp-001/results.md): threshold mode = hard, mp_factor 2.0.
- **F6** (exp-001/results.md): within-era batch composition, B=1024.
- **F3** (exp-003/results.md, FROZEN): verdict threshold = 0.00398 mean
  per-era numerai_corr (spearman variant 0.00391).
- **F5** (exp-003): moving-block bootstrap, block length L=4 eras, paired
  per-era design.
- **Baseline config** (exp-003): t07 = lr 1e-3, wd 1e-3, dropout 0.2,
  2000 steps @ B=1024; 3-seed mean per-era numerai_corr +0.01594 on tuning
  eras. Same MLP as exp-001/exp-003 (197k params, 705 medium features).
- **F12 signatures already on record** (exp-003): wd and dropout best at grid
  boundary with improving trends; spectral arm affords ~1 tuning trial vs
  AdamW's 12. A "hurts" verdict MUST be downgraded to "no evidence of benefit
  under the affordable tuning budget".
- **Data**: train shard (150 eras 0425–0574) already on cluster/ephemeral;
  the VERDICT shard (eras 0971–1225, same 705 features) must be built locally
  (reuse exp-003/src/build_tuning_shard.py machinery pointed at the verdict
  block) and rsynced up. Local full parquet path is in exp-001/out/protocol.json.

## Arms (one GPU job, ~13 min projected; all 2000 steps @ B=1024 within-era)

All arms train on the SAME train shard with the SAME base config (t07) and
the SAME 3 seeds; identical data order per seed across arms:
1. **filter-off** (plain tuned AdamW t07) × 3 seeds — ~4 s each.
2. **filter-on** (SpectralConsensusFilter, hard, mp_factor 2.0, wrapping
   AdamW t07) × 3 seeds — ~63 s each. One pre-registered spectral setting
   only (F12: priors transferred from /home/titus/pyg/optimizers README as
   the search center; record the transferred values).
3. **C4 random-subspace control** × 3 seeds: per step, project the update
   onto a random k-dim sample-subspace with k and kept-norm matched to the
   filter's measured behavior (match the realized k trajectory or its mean;
   record the matching rule) — separates MP-eigenselection from generic
   subspace projection.
4. **GAF-style simple-agreement ablation** × 3 seeds (exp-003 confirmed it
   fits): sign/agreement-based filtering without eigendecomposition —
   separates "any agreement heuristic" from spectral selection.

Diagnostics logged every ~50 steps in all filtered arms: k, kept-norm
fraction, C3 cosines. Evaluation: after training, predict on the VERDICT
shard once per (arm, seed); per-era numerai_corr and spearman vectors saved
to CSV. Nothing else ever touches the verdict shard.

## Verdict computation (local CPU, after the job returns)

- Paired per-era differences (filter-on − filter-off), per seed and pooled;
  moving-block bootstrap (L=4) CIs. Three-way verdict per success-criteria:
  **helps** (CI excludes 0 AND mean improvement ≥ F3=0.00398), **hurts**
  (symmetric, then F12 downgrade applies), **doesn't help** (equivalence: CI
  excludes ±0.00398 improvement). Otherwise: no-verdict → report the effect
  estimate with its CI, honestly scoped.
- **F8 (mandatory)**: tail-era breakdown — verdict-block eras split by
  baseline per-era corr quantiles (and by per-era target dispersion);
  report where the filter helps/hurts. This directly tests the Feldman
  long-tail counter-hypothesis.
- **F10**: corr-Sharpe (mean/std of per-era corr) comparison ONLY via the
  same paired block-bootstrap machinery; otherwise drop it.
- Same machinery applied to C4 and GAF arms vs filter-off (secondary, for
  mechanism attribution): does the spectral arm differ from its norm-matched
  random control?
- Sanity: verdict-block baseline mean corr should be within a plausible
  range of the tuning-block value; report the realized value vs the F9 band
  (regime drift is reportable, not a failure).

## PASS / FAIL

PASS: a three-way verdict is returned per the amended success criteria, with
diagnostics confirming the mechanism stayed engaged (selective filtering
throughout), C4 control results reported, F8 breakdown reported.
FAIL (= pre-authorized write-up path, not retry): no verdict category
reachable (CI too wide), or diagnostics show the filter silently disengaged
(k≈0 or kept-norm≈1 throughout) during verdict runs — the latter reopens
exp-001's mechanistic pivot.

## Compute & workflow

ONE GPU job (--qos=debug, --partition=compute, --gres=gpu:1,
--cpus-per-task=8, --mem=32G, --time well under 2h, logs/ created BEFORE
sbatch), staged per the compute-profile rsync workflow to
mats:/mnt/nw/home/t.buckworth/researcher-runs/<run-id>/exp-004/.
Cluster venv (source ~/venv/bin/activate, torch 2.5.1 pinned — never
install/upgrade torch). Shards via /ephemeral/t.buckworth/researcher-shards/
(re-copy from job script if wiped). Poll squeue with sleeps; pull results
back with rsync; copy slurm log to run.log. Verdict analysis local CPU.

## Deliverables

- results.md: the verdict (or honest no-verdict), paired CIs, F8 tail table,
  C4/GAF attribution results, diagnostics summary, F12 wording applied,
  realized baseline on verdict block, exact configs + seeds, timings.
- run.log (full slurm stdout), per-era CSVs, code + sbatch in this dir.
- No checkpoints; no files >50 MB.
