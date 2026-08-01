# Experiment 003 Results

## Component Tested
Bundle: #8 tuned-AdamW baseline sanity (λ=0.24), #3 statistical power of the
paired per-era design (λ=1.02 — lambda driver), #4 budget-fit arithmetic
(λ=0.89), plus the F4 joint go/no-go gate. Binding amendments applied: F1,
F3, F4, F5, F7, F9, F12.

## Verdict: PASS

All four sub-components PASS. **F4 gate: GO** — exp-004 proceeds as planned.
No leakage signature (the hard fail-fast trigger did not fire). F12
under-exploration signatures ARE present (recorded below; they carry into
exp-004's verdict-downgrade rule but do not fail this experiment).

| Sub-component | Verdict |
|---|---|
| #8 baseline sanity (F9 band, zero-predictor, leakage gate) | PASS |
| #3 statistical power (F5 moving-block bootstrap, verdict-block era count) | PASS |
| #4 budget fit (F7 cut order) | PASS — no cuts needed |
| F4 joint go/no-go gate | GO (0.910 ≥ 0.6) |

## Setup
- GPU part (#8 sweep): Slurm job **6432** on the MATS free `compute`
  partition, 1× NVIDIA L40, torch 2.5.1+cu121 (cluster shared venv).
  Total job compute 65.1 s (~1.1 min). Full stdout in `run.log` (Part 1);
  raw copy kept as `run-6432.log`. Code: `src/sweep.py`, `run.sbatch`.
- Local part (#3/#4/F3/F4): local desktop CPU only, Python 3.12.3,
  numpy 1.26.4 / pandas 3.0.2 / scipy 1.17.1. `src/analyze.py`, 17 s
  wall-clock. Output in `run.log` (Part 2) and `out/analyze.log`;
  machine-readable summary in `out/analysis.json`.
- Data: train shard eras 0425–0574 (384,000 rows) and tuning-block shard
  eras 0579–0966 (993,280 rows, 388 eras), staged on
  `/ephemeral/t.buckworth/researcher-shards/`.
- **Consumed verbatim from exp-001's frozen artifacts** (exp-001/results.md
  was being written concurrently; the frozen values were read from its
  output files, not re-derived):
  - `exp-001/out/protocol.json` (copied to `ref/protocol.json`): F1 blocks —
    tuning eras 0579–0966 (388 eras), embargo 0967–0970 (4 eras,
    F5 rule ceil(20d/5d)), verdict block 0971–1225 (**255 eras**).
  - `exp-001/out/f9_recalibration.json` (copied to `ref/`): F9 sanity band
    [0.0087, 0.0522] mean per-era numerai_corr; leakage gate nc > 0.0696 or
    spearman > 0.06.
  - `exp-001/cluster/out/timing.json`: measured 31.03 ms/step filter-on at
    B=1024 (2000-step run ≈ 62 s ≈ 1.03 min on one L40); 1.19 ms/step
    filter-off.

## What Was Tested

**#8 (GPU):** a 12-trial AdamW hyperparameter sweep (LR ∈ {3e-4, 1e-3, 3e-3}
× wd ∈ {0, 1e-3} × dropout ∈ {0, 0.2}; F12 search center lr=1e-3/wd=0/do=0
transferred from ~/pyg/optimizers) for the plain-AdamW MLP arm
(705-256-64-1, MSE on centered target, 2000 steps @ B=1024), trained on the
train shard and evaluated by mean per-era Spearman (selection metric) and
mean per-era numerai_corr (F9 band metric) on **tuning-block eras only**
(F1 enforced by an assert against protocol.json; the verdict block was never
loaded). Plus a seeded random zero-predictor control and 2 extra re-train
seeds of the best config for the seed-noise estimate.

**#3/F3/F4/#4 (local CPU):** from the returned per-era corr vectors —
lag-1 autocorrelation and block length (F5); the FROZEN F3 threshold;
a moving-block-bootstrap power simulation of the paired 3-seed-per-arm
design at the verdict block's 255-era count under null / ±injected-effect
scenarios (1000 outer MC × 500 inner bootstrap, plus a 2× noise sensitivity
and a non-headline iid comparison); the F4 joint gate; and the #4 packing
arithmetic from measured step times.

## Results

### Raw Output
```
zero-predictor: mean spearman +0.00026, mean numerai_corr -0.00016 over 388 eras
BEST trial (by mean per-era spearman): t07_lr0.001_wd0.001_do0.2 spearman +0.01894 numerai_corr +0.01962
  seed 1: mean spearman +0.01315, mean numerai_corr +0.01286 (2.6s)
  seed 2: mean spearman +0.01478, mean numerai_corr +0.01533 (2.6s)
F9 band [0.0087, 0.0522] (numerai_corr): best +0.01962 -> INSIDE
leakage gate (nc>0.0696 or sp>0.0600): not fired

F5: lag-1 acf (numerai_corr, best seed0) = +0.247; acf[1..5] = ['+0.25', '+0.10', '-0.03', '-0.11', '-0.02']; block length L = 4
F3 FROZEN threshold: numerai_corr 0.00398 (spearman variant 0.00391; selection-value alternative would have been 0.00490)
seed noise (numerai_corr): per-era seed sd 0.01968; 3-seed-mean diff sd 0.01582; diff-series lag1 acf +0.205
power[seed_noise][null]: hw_med 0.00219 any_cat 0.910 helps* 0.052 doesnt 0.838 helps_strict 0.002
power[seed_noise][plus_thr]: hw_med 0.00218 any_cat 0.944 helps* 0.899 doesnt 0.047 helps_strict 0.480
F4 gate: min P(any verdict category) = 0.910 -> GO  (2x-noise sensitivity: 0.096)
#4: exp-004 single job 12.7 min (cap 25) fits=True; exp-005 2-task array 13.7 min/task fits=True; F7 cuts: none
```
(Full raw output: `run.log`.)

### #8 sweep table (388 tuning-block eras, seed 0, 2000 steps @ B=1024)

| Trial | lr | wd | dropout | mean spearman | mean numerai_corr | train s |
|---|---|---|---|---|---|---|
| t00 | 3e-4 | 0 | 0 | +0.01348 | +0.01430 | 2.6 |
| t01 | 3e-4 | 0 | 0.2 | +0.01515 | +0.01660 | 2.6 |
| t02 | 3e-4 | 1e-3 | 0 | +0.01033 | +0.01143 | 2.4 |
| t03 | 3e-4 | 1e-3 | 0.2 | +0.01595 | +0.01640 | 2.6 |
| t04 | 1e-3 | 0 | 0 | +0.01224 | +0.01129 | 2.4 |
| t05 | 1e-3 | 0 | 0.2 | +0.01510 | +0.01448 | 2.6 |
| t06 | 1e-3 | 1e-3 | 0 | +0.01149 | +0.01343 | 2.4 |
| **t07** | **1e-3** | **1e-3** | **0.2** | **+0.01894** | **+0.01962** | 2.6 |
| t08 | 3e-3 | 0 | 0 | +0.00864 | +0.00859 | 2.4 |
| t09 | 3e-3 | 0 | 0.2 | +0.00898 | +0.01030 | 2.6 |
| t10 | 3e-3 | 1e-3 | 0 | +0.01027 | +0.01084 | 2.4 |
| t11 | 3e-3 | 1e-3 | 0.2 | +0.00816 | +0.00812 | 2.6 |

Best config t07 re-trained at 2 extra seeds: seed 1 +0.01315/+0.01286,
seed 2 +0.01478/+0.01533 (spearman/numerai_corr). Note the seed-to-seed
spread: the sweep-selection value (+0.0196 nc) shrinks to a 3-seed mean of
**+0.01594 nc** — winner's-curse shrinkage of ~19%, which is why F3 below
uses the multi-seed mean. F12 tuning-match record: 12 trials × ~3.65 s =
~44 s of tuning compute.

### Metrics
| Metric | Target | Actual | Pass? |
|--------|--------|--------|-------|
| #8 best trial mean per-era numerai_corr | inside F9 band [0.0087, 0.0522] (tuning eras) | +0.01962 (3-seed mean +0.01594 — also inside) | Y |
| #8 zero-predictor \|corr\| | < 0.002 | spearman +0.00026, nc −0.00016 | Y |
| #8 leakage signature | none (nc ≤ 0.0696, sp ≤ 0.0600) | not fired (max trial +0.0196) | Y |
| #8 all-trials-≈0 fail mode | not all < 0.003 | min trial +0.00816 | Y |
| #3 block-bootstrap CI half-width (median, null, 255 eras, 3 seeds) | ≤ ~F3 threshold (0.00398); hard fail > 0.008 | 0.00219 | Y |
| F4 gate: min P(some verdict category reachable) | ≥ 0.6 | 0.910 | Y (GO) |
| #4 exp-004 job (tune + 2 arms × 3 seeds + C4 + GAF + diagnostics) | ≤ 25 min | 12.7 min | Y |
| #4 exp-005 sequence arm (conditional; exp-002 PASSED) | ≤ 25 min/job | 13.7 min per array task (2 tasks) | Y |

### F12 under-exploration signatures (binding record for exp-004)

- **Best config on grid boundary: YES, on 2 of 3 axes.** wd = 0.001 is the
  grid max and the wd slice through the best config is still improving toward
  it (+0.01510 → +0.01894); dropout = 0.2 is the grid max and also still
  improving (+0.01149 → +0.01894). Both axes had only 2 grid points, so
  "boundary" is trivially true — the informative signature is the
  still-improving trend at the boundary: stronger regularization than the
  grid offered might tune the baseline higher.
- **lr is well-behaved**: best at the interior point (1e-3) with a clean
  interior peak in the slice through the best config; 0 of 4 lr slices
  show valley/irregular shapes.
- **Consequence (F12, binding on exp-004):** the tuned baseline may be
  slightly under-tuned. If exp-004 returns a "hurts" verdict, it must be
  downgraded to "no evidence of benefit under the affordable tuning budget".
  Additionally (from #4 below): the matched-compute tuning budget affords
  only ~1 spectral-arm trial, an under-exploration signature for the
  *spectral* arm cutting the same way.

### F5: autocorrelation and block length

Lag-1 autocorrelation of the best trial's per-era numerai_corr series
(seed 0, 388 tuning eras): **+0.247** (spearman variant +0.277; acf lags
1–5: +0.25, +0.10, −0.03, −0.11, −0.02). The seed-difference series has
lag-1 acf +0.205. Block length rule: L = max(embargo-derived overlap = 4
eras [ceil(20d horizon / 5d era spacing), from protocol.json], first lag
with |acf| < 0.1 = 3) → **L = 4 eras**, used for all bootstrap CIs here and
frozen for exp-004's verdict CIs. (For context, exp-001's F9 recalibration
measured lag-1 acf 0.76 for the *example model's* per-era corr — our
lower-capacity model's series is much less persistent, but the positive
autocorrelation confirms iid bootstrap would be anti-conservative; the
moving-block bootstrap is the headline machinery per F5.)

### F3: FROZEN verdict threshold (registered before any arm comparison)

Rule: threshold = min(0.005, 0.25 × realized tuned-baseline mean per-era
corr). **Definition decision** — "realized tuned-baseline" is taken as the
**mean over the 3 available seeds of the best config**, not the
sweep-selection value: the selection value (+0.01962 nc) is inflated by
selection over 12 trials (winner's curse; the two fresh seeds landed at
+0.01286/+0.01533), while the 3-seed mean (+0.01594 nc) is the honest
estimate of the level the 3-seed baseline arm will actually realize in
exp-004. It is also the lower — hence more conservative — choice for the
power/F4 computation: a smaller threshold is harder to resolve, so passing
at it implies passing at the alternative.

- **FROZEN: threshold = 0.00398 (numerai_corr, primary)**; spearman variant
  0.00391. (Alternative under the selection-value reading: 0.00490 —
  recorded, not used.)
- Frozen here, before any filter-on vs filter-off comparison is unblinded;
  binding for exp-004.

### #3 power simulation (F5-compliant moving-block bootstrap)

Design simulated: paired per-era difference of 3-seed-mean arms, **255
verdict-block eras** (era count from protocol.json; F1 — no verdict-block
data touched), block length L=4, 95% percentile CIs, 1000 outer × 500 inner.
Noise model: demeaned pairwise per-era differences between the 3 seeds of
the best config, scaled by 1/√3 to the 3-seed-mean scale (per-era 3-seed-mean
difference sd 0.01582). **Transfer assumption (stated per F1):** the noise
vectors come from tuning-block eras (388) because verdict-block data may not
be touched; the simulation assumes the verdict block's per-era difference
dispersion/autocorrelation resembles the tuning block's. The 2×-noise row
bounds the consequence of that assumption failing.

| Noise | Scenario (true effect) | CI half-width med / p90 | P(any category) | P(helps: CI>0) | P(doesn't-help) | P(helps strict: CI>0 & mean≥thr) |
|---|---|---|---|---|---|---|
| measured | null (0) | 0.00219 / 0.00255 | 0.910 | 0.052 | 0.838 | 0.002 |
| measured | +thr (+0.00398) | 0.00218 / 0.00254 | 0.944 | 0.899 | 0.047 | 0.480 |
| measured | −thr | 0.00219 / 0.00255 | 0.951 | 0.000 (hurts 0.924) | 0.031 | — (hurts strict 0.478) |
| measured | +2·thr | 0.00218 / 0.00253 | 1.000 | 1.000 | 0.000 | 1.000 |
| 2× | null | 0.00435 / 0.00505 | 0.096 | 0.039 | 0.026 | 0.037 |
| 2× | +thr | 0.00434 / 0.00507 | 0.462 | 0.454 | 0.008 | 0.437 |
| 2× | −thr | 0.00437 / 0.00512 | 0.432 | 0.001 (hurts 0.429) | 0.002 | 0.001 |

(iid-bootstrap comparison, non-headline per F5: null half-width 0.00190 —
the block bootstrap is ~15% wider, as expected under positive
autocorrelation. Full numbers in `out/analysis.json`.)

**#3 PASS**: median CI half-width 0.00219 ≤ F3 threshold 0.00398 (and far
below the 0.008 hard-fail bound) at 3 seeds × 255 verdict eras. Detection is
strong for CI-sign verdicts (0.90–0.92 at ±thr) and the null correctly
resolves to "doesn't help" 84% of the time. Honest caveat: the *strict*
helps/hurts verdict (CI excludes 0 AND point estimate ≥ thr) has ~0.48
probability when the true effect sits exactly at the threshold — an
unavoidable property of a point-estimate-vs-threshold rule at the boundary,
not a power defect; effects at 2×thr are detected strictly with p≈1.

### F4 joint go/no-go gate

Gate statistic: min over {null, +thr, −thr} scenarios of P(at least one
verdict category — helps / hurts / doesn't-help — reachable), at measured
noise: min(0.910, 0.944, 0.951) = **0.910 ≥ 0.6 → GO**. exp-004 proceeds as
the categorical-verdict experiment; the pre-authorized re-scope (effect
estimate with calibrated uncertainty) is NOT invoked. Sensitivity: at 2× the
measured seed noise the gate would fail (0.096) — if exp-004's realized
paired-difference dispersion comes in far above the tuning-block estimate,
its own pre-authorized no-verdict path (honest effect estimate) absorbs
this; the gate decision itself uses the measured noise per the
pre-registered rule.

### #4 budget-fit arithmetic (measured times)

Inputs: t_AdamW-run = 3.65 s (median train+eval, measured in this sweep,
2000 steps @ B=1024); t_spectral-run = 62.05 s + 1.05 s eval ≈ 63.1 s
(exp-001 measured 31.03 ms/step filter-on @ B=1024 × 2000 steps);
t_sequence-run ≈ 3.9 min (exp-002's conservative CPU→L40 projection;
exp-002 PASSED so the sequence arm is included per its conditions).
Job cap 25 min; +120 s/job overhead (data load, imports, diagnostics).

F12 matched-compute tuning: AdamW tuning spent 12 × 3.65 ≈ 44 s, which
affords round(44/63.1) = **1 spectral tuning trial** — the spectral arm runs
at the transferred prior-repo search center (F12), with the threshold mode
already fixed by exp-001 per F2. Recorded as a spectral-arm
under-exploration signature (feeds exp-004's F12 downgrade rule).

**exp-004 (one GPU job, 2000 steps @ B=1024 each run):**

| Item | Runs | s/run | Subtotal |
|---|---|---|---|
| Spectral tuning (F12 matched compute) | 1 | 63.1 | 63 s |
| Filter-on (spectral) arm | 3 seeds | 63.1 | 189 s |
| Filter-off (tuned AdamW t07) arm | 3 seeds | 3.65 | 11 s |
| C4 random-subspace norm-matched control (F7-protected) | 3 seeds | 63.1 | 189 s |
| GAF-style simple-agreement ablation | 3 seeds | 63.1 | 189 s |
| Overhead + diagnostics | — | — | 120 s |
| **Total** | | | **761 s ≈ 12.7 min ≤ 25 min → fits** |

Minimum viable (would drop GAF first per F7): 9.5 min. Headroom ≈ 12 min,
so **no F7 cuts are invoked** — the cut order was checked but no pressure
exists. Muon and GBT remain pre-designated future-work cuts (W3), as before.

**exp-005 (conditional sequence arm — conditions met: exp-002 PASSED and
room remains):** 2 arms × 3 seeds at ~3.9 min/run, split as a **2-task array
job** (one arm per task): 3 × 234 s + 120 s = 13.7 min/task ≤ 25 min → fits.
Total ≈ 27.4 min across both tasks. Note 3.9 min/run is exp-002's
deliberately conservative 10× CPU→L40 projection; exp-001's measured MLP
filter-on speedups suggest real runs will be faster.

**#4 PASS**: the full plan — tuning both arms + 2 arms × 3 seeds + C4
protected control + GAF ablation + diagnostics + the conditional sequence
arm — packs into the remaining 2 slots with no cuts.

### Analysis

The tuned-AdamW baseline is sane: +0.0196 (selection) / +0.0159 (3-seed
mean) mean per-era numerai_corr on 388 tuning eras sits comfortably inside
the F9-recalibrated band, the zero-predictor is indistinguishable from
noise, no trial approaches the leakage gate, and every trial clears the
all-≈0 fail mode. The hard fail-fast trigger (leakage) did not fire, so
downstream experiments are interpretable.

The power picture is favorable at the frozen threshold: with the moving-
block bootstrap (L=4) honoring the measured +0.25 lag-1 era autocorrelation,
the paired 3-seed design at 255 verdict eras yields median CI half-widths of
0.0022 — about half the frozen F3 threshold of 0.00398 — so all three
verdict categories are genuinely reachable (F4 gate 0.910). The dominant
uncertainty is the noise-model transfer (tuning-block dispersion standing in
for verdict-block dispersion, and same-config seed differences standing in
for cross-arm paired differences); the 2× sensitivity row shows the gate is
not robust to a doubling of that noise, but that contingency lands in
exp-004's pre-authorized no-verdict write-up path rather than invalidating
the design.

Budget-wise the design is comfortable: the verdict experiment fits in a
single 12.7-min job against a 25-min cap with the F7-protected C4 control
*and* the optional GAF ablation both included, and the conditional sequence
arm fits as a 2-task array. The notable honest finding from the arithmetic
is asymmetric tuning: matched compute gives the spectral arm ~1 tuning trial
against AdamW's 12 — pre-registered here as an under-exploration signature
so exp-004's verdict wording accounts for it.

## Unexpected Observations
- **Winner's curse is material at this noise level**: the sweep-selection
  value (+0.0196) overstates the 3-seed mean (+0.0159) by ~19% after only
  12 trials — vindicating the F3 decision to define the realized baseline
  from multi-seed re-trains, and previewing the seed noise exp-004 must
  average over.
- Per-era seed sd (0.0197) is comparable to the per-era market dispersion
  itself (sd ≈ 0.021–0.023 across eras): at 2000 steps @ B=1024, run-to-run
  variation is as large a noise source as era-to-era variation. The paired
  per-era design plus 3-seed means is what keeps the CI half-width at 0.0022
  despite this.
- Regularization (wd and dropout at their grid maxima, both trends still
  improving) helped monotonically — consistent with the low-SNR regime and
  the reason the F12 signatures fired.
- The filter-off arm is ~17× cheaper than the filter-on arm (3.65 s vs
  63.1 s per run) — the per-sample-gradient + eigh machinery utterly
  dominates exp-004's budget, not the baseline training.
- The sweep's own GPU job cost only 1.1 min of L40 time for 15 training
  runs — the subsampled-shard design leaves large compute headroom.

## Implications

What this tells us: all criteria were met. The baseline is trustworthy and
leakage-free (#8), the paired design has the statistical resolution to
return a real verdict on the verdict block (#3, F4 GO at 0.910), and the
full amended design — including the F7-protected mechanism control and the
conditional sequence arm — fits the remaining budget with no cuts (#4).

Frozen and binding for exp-004/exp-005:
- **F3 threshold: 0.00398 mean per-era numerai_corr** (spearman variant
  0.00391) — registered before any arm comparison is unblinded.
- **F5 block length: 4 eras** for all verdict CIs.
- **Baseline arm config: t07 (lr=1e-3, wd=1e-3, dropout=0.2)**, identical
  base config for the filter-on arm.
- **F12 signatures on record**: baseline best-on-grid-boundary (wd, dropout,
  both still improving) and spectral arm limited to ~1 matched-compute
  tuning trial → a "hurts" verdict in exp-004 must be downgraded to "no
  evidence of benefit under the affordable tuning budget".

Next steps: proceed to exp-004 (single 12.7-min GPU job: 1 spectral tuning
trial + filter-on × 3 seeds + filter-off × 3 seeds + C4 random-subspace
× 3 seeds + GAF ablation × 3 seeds + diagnostics, 2000 steps @ B=1024,
evaluated ONCE on the verdict block per F1), then conditional exp-005
(sequence arm, 2-task array, ~13.7 min/task).
