# Experiment 001 Results

## Component Tested

Gate bundle: #6 regression integration smoke (+C1 unit tests), #9 Numerai
data + era-purged protocol (F1/F5/F9), #7 vmap throughput on L40, and the
**headline #1 spectral-engagement gate** (with C2/C3/C5 controls and the F2/F6
binding decisions).

## Verdict: FAIL (headline #1) — sub-components #6, #9, #7 all PASS

The headline FAIL is on the **C2 conjunct** of the pre-registered PASS
criterion, not on the degeneracy fail-fast clause:

- The filter **does engage mechanically and selectively**: for every mode ×
  batch size × composition, 0 < k < B at 100% of logged steps, k settles at
  3–7 with above-threshold eigenvalues (raw counts 3–11 above the MP-style
  threshold, no clamping artifact), the norm passed is non-degenerate, and
  C3 shows the filtered update is NOT mean-gradient smoothing (cosine
  median 0.25–0.83; >0.95 on ≤15% of steps; never >0.95 "everywhere").
  The HARD FAIL-FAST trigger (all modes ≈0%/≈100% at both B and both
  compositions, or C3 >0.95 everywhere) did **not** fire.
- But the PASS criterion also requires "**C2 permuted-target spectra are
  distinguishable from real-target spectra**", and they are **not** — and we
  can prove they **cannot be**, for this optimizer on this model class:

  **Mechanistic finding.** For a scalar-output model trained with per-sample
  MSE, the per-sample gradient is g_i = 2(f(x_i) − y_i) · ∇_θ f(x_i). After
  the row normalization used by `SpectralConsensusFilter`, the normalized
  gradient is sign(f_i − y_i) · ∇f_i/‖∇f_i‖, so the similarity matrix
  satisfies S(y′) = D S(y) D with D = diag(±1) for ANY targets y′ at the same
  parameter point. D is orthogonal, so the **eigenspectrum — and therefore k,
  the MP-threshold count, kept-energy fraction, and every spectrum-derived
  "engagement" diagnostic — is exactly target-independent at fixed
  parameters.** Targets influence the spectrum only indirectly, through the
  parameter trajectory, which at this horizon/SNR does not measurably
  diverge. Verified three ways below (fp64 unit check to 9e-16; step-0
  cluster spectra identical to float32 roundoff; late-step distributions
  statistically indistinguishable / directionally inconsistent).

Per the plan's fail semantics and the limitation triage, this pivots the run
to a **mechanistic-finding deliverable**: "spectral engagement" as measured by
this optimizer's diagnostics is a property of the Jacobian/feature structure,
not of target signal, on scalar-output MSE regression — so an engagement-
gated helps/hurts comparison on this task cannot attribute engagement to
signal. This is a clean, provable boundary result (contrast: multi-class
cross-entropy, the prior project's MNIST setting, has per-sample gradients
that are NOT scalar multiples of a target-independent vector, so the
diagnostic is target-sensitive there).

## Setup

- Local: Python 3.12, torch 2.11.0+cu128 (CPU used for smoke), numpy/pandas/
  scipy for analysis. Cluster: MATS Slurm job 6430, `--qos=debug`, one NVIDIA
  L40, torch 2.5.1+cu121 (pinned venv).
- Duration: local smoke 7.7 s; data prep ~minutes (6.3 GB of parquet
  downloads); cluster job **2.5 min total** GPU; analysis local CPU.
- Data: Numerai v5.0. Shard used on GPU: 384,000 rows = 150 train eras
  (0425–0574) × ≤2,560 rows, 705 "medium" features. Model: MLP 705-256-64-1
  (197,249 params), per-sample MSE on centered target, AdamW lr 1e-3, wd 0.

## What Was Tested

Locally: the `SpectralConsensusFilter` (F11 identity: exact B×B MP-threshold
filter from `spectral_optimizer.py`, copied — home repo untouched) was wrapped
around AdamW in a fresh regression loop (integration smoke + filter-off
equivalence + streaming-variant smoke + C1 spiked-covariance unit tests), and
the Numerai v5 era-purged protocol (F1/F5/F9) was built from the real
validation data. On the cluster, ONE debug job ran #7 timing (200 steps,
B ∈ {256, 1024}, filter on/off) and the #1 engagement grid: 400 steps ×
{hard, soft, variance} × B ∈ {256, 1024} × {within-era, mixed-era} (F6),
plus C2 permuted-target null runs (hard mode, both B, both compositions),
logging every 10 steps: k, consensus_ratio, top-10 eigenvalues, C3 cosines,
full eigenspectrum snapshots every 100 steps, and C5 era-identity probes
every 50 steps on within-era configs (implemented — not a gap).

**Diagnostic definitions** (from `src/spectral_optimizer.py` /
`src/cluster_job.py`): `k` = eigenvalues of the row-normalized similarity
matrix above threshold 2·(trace/B) ≈ 2.0 (hard/soft; clamped ≥1 in hard) or
the 90%-cumulative-variance count (variance mode). `consensus_ratio` =
‖filtered_grad‖/‖mean_grad‖ — **not bounded by 1**: the filter projects the
uniform weight vector onto the kept eigensubspace of the *normalized*
similarity matrix and applies it to the *unnormalized* G, so removing
anti-aligned (cancelling) components can raise the norm; values >1 mean the
filtered update is larger than the plain mean gradient. The bounded
"fraction of gradient energy kept" was therefore computed separately from
the saved full spectra: sum(eigs > thr)/sum(eigs).

## Results

### Sub-component verdicts

| Sub-component | Criterion | Measured | Verdict |
|---|---|---|---|
| #6 integration | loss decreases; diagnostics emitted; both variants run; filter-off == AdamW; C1 correct | loss 0.0583→0.0425; k∈[7,10], ratio∈[0.13,1.23]; streaming 0.047→0.025; fp64 grad diff 5.6e-17, 50-step param diff 9.9e-15; C1 table below | **PASS** |
| #9 data+protocol | ≥100 usable validation eras; <32 GB; shard readable in GPU job | 643 usable eras; 0.56 GB peak for shard build; job 6430 read the shard from /ephemeral | **PASS** |
| #7 throughput | projected full run < 20 min | 9.9 ms/step (B=256, on), 31.0 ms/step (B=1024, on); projected 2000-step run **0.33 / 1.03 min**; filter-off 1.1/1.2 ms/step | **PASS** (no streaming swap → no F11 scope change) |
| #1 engagement (headline) | selectivity AND C3 ≤ ~0.95 AND **C2 distinguishable** | selectivity: yes; C3: yes; **C2: indistinguishable (provably so)** | **FAIL** |

### C1 spiked-covariance unit tests (calibration of "engaged ≠ meaningful")

| Case | Mode | k (raw above thr) | norm ratio | cos vs planted dir |
|---|---|---|---|---|
| (a) iid noise | hard | 1 (0) | **0.001** | — |
| (a) iid noise | soft | 0 (0) | 0.142 | — |
| (a) iid noise | variance | 219 (0) | **0.950** | — |
| (b) planted spike | hard | 1 (1) | 1.000 | **0.961** |
| (b) planted spike | soft | 1 (1) | 1.000 | 0.961 |
| (b) planted spike | variance | 217 (1) | 1.000 | 0.961 |
| (c) correlated zero-signal confound | hard | 1 (1) | 0.015 | **0.959** |
| (c) correlated zero-signal confound | soft | 1 (1) | 0.151 | 0.096 |
| (c) correlated zero-signal confound | variance | 217 (1) | 0.920 | 0.016 |

Hard mode behaves correctly on (a) and (b); on (c) it **keeps the correlated
zero-signal confound** (cos 0.959 to the confound direction) — the measured
calibration that spectral "engagement" does not certify signal. The #1
headline result is this same phenomenon on real data, at full strength.
Variance mode fails calibration (a): it passes 95% of pure iid noise.

### #1 engagement grid (40 logged steps per config; "late" = steps ≥ 100)

| Config | frac steps 0<k<B | k med (late) | ratio med (late) | ratio∈[0.1,0.9] | ratio>1 | C3 cosF med | cosF>0.95 | cosU>0.95 |
|---|---|---|---|---|---|---|---|---|
| hard B=256 within | 1.00 | 4.0 | 0.67 | 0.62 | 0.30 | 0.59 | 0.00 | 0.00 |
| hard B=256 mixed | 1.00 | 4.0 | 0.77 | 0.68 | 0.28 | 0.36 | 0.03 | 0.00 |
| hard B=1024 within | 1.00 | 4.5 | 1.09 | 0.38 | 0.55 | 0.54 | 0.05 | 0.00 |
| hard B=1024 mixed | 1.00 | 5.0 | 1.72 | 0.33 | 0.62 | 0.54 | 0.07 | 0.00 |
| soft B=256 within | 1.00 | 3.0 | 0.93 | 0.47 | 0.45 | 0.73 | 0.03 | 0.00 |
| soft B=256 mixed | 1.00 | 4.0 | 0.95 | 0.53 | 0.38 | 0.60 | 0.05 | 0.00 |
| soft B=1024 within | 1.00 | 4.0 | 1.33 | 0.40 | 0.60 | 0.44 | 0.05 | 0.00 |
| soft B=1024 mixed | 1.00 | 4.0 | 1.52 | 0.45 | 0.55 | 0.56 | 0.07 | 0.00 |
| variance B=256 within | 1.00 | 3.0 | 0.96 | 0.42 | 0.50 | 0.83 | 0.15 | 0.20 |
| variance B=256 mixed | 1.00 | 4.0 | 1.46 | 0.33 | 0.60 | 0.72 | 0.12 | 0.17 |
| variance B=1024 within | 1.00 | 4.0 | 1.13 | 0.38 | 0.62 | 0.76 | 0.12 | 0.17 |
| variance B=1024 mixed | 1.00 | 3.0 | 1.67 | 0.33 | 0.62 | 0.76 | 0.10 | 0.17 |
| hard B=256 within PERM | 1.00 | 4.5 | 0.57 | 0.68 | 0.25 | 0.60 | 0.00 | 0.00 |
| hard B=256 mixed PERM | 1.00 | 4.0 | 0.67 | 0.70 | 0.30 | 0.32 | 0.00 | 0.00 |
| hard B=1024 within PERM | 1.00 | 5.0 | 1.66 | 0.28 | 0.70 | 0.25 | 0.00 | 0.00 |
| hard B=1024 mixed PERM | 1.00 | 5.0 | 1.09 | 0.30 | 0.57 | 0.55 | 0.15 | 0.00 |

Selectivity conjunct: met (all configs, all modes, sustained). Bounded
kept-energy fraction from full spectra: 0.29→0.81 (B=256) and 0.39→0.89
(B=1024) over training — inside (0,1) throughout, non-degenerate. C3
conjunct: met (cosF median 0.25–0.83; the >0.95 pivot never fires; the
AdamW-update-level cosine exceeds 0.95 on ≤20% of steps and only in variance
mode).

### C2 real vs permuted targets — the failed conjunct (numbers)

Three independent lines of evidence:

1. **fp64 proof check** (`src/verify_target_independence.py`): at fixed
   parameters, max |eig(real) − eig(permuted)| = **8.9e-16**;
   real vs entirely-different targets = 1.3e-15. Spectrum is exactly
   target-independent, as the S(y′) = D S(y) D argument requires.
2. **Step-0 cluster spectra** (same seed → same init parameters, same batch,
   only y permuted): full-spectrum max abs diff 3.1e-5–2.8e-4 on top1 of
   39–154, i.e. float32 roundoff. k identical (11=11, 12=12, 57=57, 57=57).
3. **Late-training distributions** (steps ≥ 100, n=30 logged steps each,
   Mann-Whitney two-sided):

| Pair (hard mode) | k real/perm (median) | top1 real/perm (median) | p(k) | p(top1) | p(ratio) |
|---|---|---|---|---|---|
| B=256 within | 4.0 / 4.5 | 186.1 / 174.6 | 0.012 | 0.033 | 0.26 |
| B=256 mixed | 4.0 / 4.0 | 182.4 / 182.7 | 0.33 | 0.85 | 0.88 |
| B=1024 within | 4.5 / 5.0 | 813.9 / 810.3 | 0.49 | 1.00 | 0.13 |
| B=1024 mixed | 5.0 / 5.0 | 840.2 / 871.5 | 0.81 | **0.0015** | 0.32 |

The two nominally significant p-values point in **opposite directions**
(real top1 higher at B=256 within; **permuted** top1 higher at B=1024 mixed)
— trajectory noise, not signal. Kept-energy fractions at step 300:
real 0.81/0.80/0.86/0.89 vs permuted 0.80/0.81/0.86/0.91. Training loss at
the end is also indistinguishable (e.g. B=256 within: real 0.0522 vs
permuted 0.0482 — real is not even lower), consistent with the task's tiny
SNR at a 400-step horizon. **The filter does identical "work" on pure-noise
targets.** (Permuted controls were run for hard mode only, per plan; the
target-independence proof covers all three modes, since all operate on the
same spectrum.)

### C5 era-identity probe — implemented; result: no evidence of era-specificity

Implemented in `cluster_job.py` (within-era configs, every 50 steps, 8 probes
per config): cosine of the filtered update against the mean gradient of (i)
256 held-out same-era rows vs (ii) 256 rows from another era.

| Config | cos same-era (mean) | cos other-era (mean) | gap | Wilcoxon p |
|---|---|---|---|---|
| hard B=256 within | +0.195 | +0.308 | −0.113 | 0.55 |
| hard B=1024 within | +0.179 | +0.230 | −0.050 | 1.00 |
| soft B=1024 within | −0.262 | +0.155 | −0.417 | 0.078 |
| variance B=1024 within | −0.076 | +0.321 | −0.397 | 0.11 |
| hard B=256 within PERM | +0.194 | −0.096 | +0.290 | 0.64 |
| hard B=1024 within PERM | +0.101 | −0.200 | +0.300 | 0.20 |

No positive same-era gap anywhere on real targets (gaps are negative), the
permuted nulls show gaps of comparable magnitude with the opposite sign, and
nothing is significant at n=8 probes. The probe finds **no evidence that the
kept subspace is era-specific**, but at this power it is inconclusive rather
than exculpatory; carried as a limitation.

### F9 recalibrated bands (binding for exp-003)

From current v5.0 example-model preds over 651 usable validation eras:
mean per-era numerai_corr **0.0348** (sd 0.0211, Sharpe 1.65), Spearman mean
0.0330. **Sanity band [0.00870, 0.05220]** (0.25×–1.5× example mean).
**Leakage gate: numerai_corr > 0.0696 or Spearman > 0.06.** Lag-1
autocorrelation of per-era corr **0.763** (F5-relevant: block bootstrap
mandatory). Tuning-block example mean 0.0418; verdict-block 0.0235.

### F1/F5 block boundaries (binding, must be reused downstream)

- Train: eras ≤ 0574 (GPU shard: 0425–0574).
- Dropped at train boundary (embargo 4 = ceil(20d horizon / 5d spacing)):
  0575–0578.
- **Tuning block: 0579–0966** (388 eras).
- Embargo between blocks: 0967–0970.
- **Final-verdict block: 0971–1225** (255 eras). Total usable: **643 ≥ 100**.

### F2 pre-registered threshold mode (binding; from engagement diagnostics ONLY)

**Registered mode: `hard` (mp_factor = 2.0).**
Rule used (no out-of-sample/validation performance was computed or seen in
this experiment): choose the mode that (i) passes the C1 calibration —
keeps ≈0 gradient norm on pure iid noise (hard: 0.1%; soft: 14%; variance:
95% — variance fails outright) while keeping a planted spike (all pass);
and (ii) on real data shows sustained non-degenerate selectivity with the
lowest incidence of mean-gradient-like steps (cosF > 0.95 on 0–7% of steps
for hard, vs 10–15% for variance). Hard mode uniquely satisfies both.

### F6 batch-composition recommendation

**Within-era** for any downstream comparison. Justification: engagement
diagnostics are statistically equivalent across compositions (late-k medians
4–5 both; overlapping ratio/cosF distributions), so there is no diagnostic
cost; within-era matches the per-era evaluation metric and the era-factor
hypothesis, and uniquely supports C5-style probes. Mixed-era at B=1024
showed the heaviest norm inflation (late ratio median 1.72). (Moot for the
comparison track given the headline FAIL, but binding if a follow-up runs
it.)

### Raw Output

Full slurm stdout of job 6430 in `run.log` (byte-identical to
`cluster/logs/slurm-6430.out`). Key lines:

```
device: cuda, torch 2.5.1+cu121 / gpu: NVIDIA L40
shard: 384000 rows, 705 features, 150 eras (0425..0574); MLP params: 197249
timing_B=256_filteron:  9.9 ms/step; projected 2000-step full run: 0.3 min
timing_B=1024_filteron: 31.0 ms/step; projected 2000-step full run: 1.0 min
mode=hard_B=256_comp=within:          step 300 k 4 ratio 1.120
mode=hard_B=256_comp=within_permuted: step 300 k 4 ratio 0.736
Total job time: 2.5 min
```

### Metrics (headline #1 conjuncts)

| Conjunct | Target | Actual | Pass? |
|---|---|---|---|
| Sustained 0 < k < B, some mode, both B | nontrivial fraction of training | 100% of logged steps, ALL modes/configs | Y |
| Grad-norm passed ~[0.1, 0.9] | nontrivial fraction | ratio in [0.1,0.9] on 28–70% of steps; kept-energy fraction 0.29–0.89 throughout | Y |
| C3 mean-gradient cosine ≤ ~0.95 | not smoothing | median 0.25–0.83; >0.95 on ≤15% of steps | Y |
| **C2 real vs permuted distinguishable** | distinguishable | indistinguishable; proven target-independent at fixed θ (fp64 diff 9e-16) | **N** |

### Analysis

The gate did exactly what it was designed to do. Mechanically, the filter is
healthy and selective on real financial gradients: a handful of eigenvalues
(3–11) stand far above the MP-style threshold (top1/threshold 20→430 over
training), the kept subspace carries 30–90% of gradient energy, and the
filtered update is measurably different from mean-gradient smoothing. If the
PASS criterion had been only "does the filter engage", this would be a PASS.

The C2 control, added at the limitation triage precisely because "engagement
diagnostics validate execution but not interpretation" (A1/A3/M3/P2), showed
the engagement is entirely explained by feature/Jacobian structure: every
spectrum-derived diagnostic is provably invariant to the targets at fixed
parameters for scalar-output MSE, because row normalization reduces each
per-sample gradient to ±(Jacobian direction). The C1(c) confound test
measured this failure mode synthetically (the filter keeps a correlated
zero-signal confound at cos 0.96); C2 shows the real data is the confound
case, fully. The spectra also reveal strong target-independent rank collapse:
top1 grows 39→193 (B=256) and 154→858 (B=1024) while k falls from 11/57 to
~4 — per-sample gradient directions align progressively during training on
real AND permuted targets alike.

Two important scoping notes. First, the *update direction* (unlike the
spectrum) does depend on targets — through residual signs and magnitudes —
so the filter is not literally target-blind as an optimizer; what fails is
the ability of the *engagement diagnostics* to certify that the mechanism
found target signal, which was this gate's pass criterion and the premise
for interpreting any downstream null. Second, the proof is specific to
scalar-output per-sample losses with row normalization; it does not apply to
multi-class cross-entropy (the prior project's MNIST label-noise setting),
which cleanly explains why the same diagnostics were informative there and
are not here — that contrast *is* the mechanistic finding.

## Unexpected Observations

- `consensus_ratio` exceeds 1 on 25–70% of steps: the filter frequently
  *amplifies* the update relative to the mean gradient by discarding
  anti-aligned components — "filtering" here is not shrinkage.
- Strong target-independent rank collapse of the normalized per-sample
  gradient similarity spectrum during training (one direction ends up with
  ~80–90% of the mass).
- After 400 steps, real-target training loss is not below permuted-target
  loss — the task's signal is invisible at this horizon even in-sample.
- C5 gaps were *negative* on real targets (filtered update slightly more
  aligned with other-era gradients), positive on permuted — all within noise.
- The fp32 filter-off equivalence check needed fp64 to confirm cleanly
  (fp32 single-step grad diff 3.7e-8 is eigh nondeterminism-scale, fp64
  5.6e-17 settles it).

## Implications

What this tells us: the pre-registered PASS criterion was not met — the C2
permuted-target control is indistinguishable from real targets, and this is
now understood analytically, not just empirically: **on scalar-output MSE
regression, the Spectral Optimizer's engagement diagnostics measure feature/
Jacobian structure only and are provably target-independent at fixed
parameters.** "Engaged" therefore cannot mean "found target signal" on this
task class, and the planned engagement-gated helps/hurts comparison would
rest on an invalid premise.

Implications: per the plan's fail semantics, do NOT proceed to the
comparison track. The run pivots to the mechanistic-finding deliverable:
(1) the target-independence theorem + fp64/cluster verification, (2) the
C1 confound calibration showing the failure mode synthetically, (3) the
real-data characterization (rank collapse, kept-energy trajectories,
real-vs-permuted statistics), and (4) the contrast with the multi-class
cross-entropy regime where the diagnostic is target-sensitive. This is a
valued outcome under the success criteria ("boundary/degeneracy findings are
explicitly priced as deliverables").

Possible next steps (future work, resource-scoped): target-sensitive
engagement diagnostics for regression (e.g., unnormalized-gradient spectra,
residual-magnitude-weighted similarity, or sign-pattern statistics of
D = diag(sign residual)); a W4-style characterization of the shifted bulk
edge under cross-sectional correlation; a higher-power C5 probe. Any revived
comparison track must use the F1 blocks, F9 bands, F2 mode (hard), and F6
composition (within-era) registered here.

## Data & Artifact Paths

- Numerai parquet (local, outside experiment dir):
  `<run-dir>/data/` — `v5.0_train.parquet` (2.4 GB),
  `v5.0_validation.parquet` (3.8 GB), `v5.0_validation_example_preds.parquet`,
  `shard_train.parquet` (104 MB, the GPU shard), `shard_tuning.parquet`.
- Cluster shard copy: `/ephemeral/t.buckworth/researcher-shards/shard_train.parquet`;
  job dir `/mnt/nw/home/t.buckworth/researcher-runs/<run-id>/exp-001/`.
- Evidence: `run.log` (slurm job 6430), `out/smoke_results.json`,
  `out/smoke_stdout.txt`, `out/protocol.json`, `out/f9_recalibration.json`,
  `out/example_per_era_corr.csv`, `cluster/out/timing.json`,
  `cluster/out/engagement.json`, `cluster/out/spectra.npz`.
- Analysis (this session, local CPU): `src/analyze_engagement.py` →
  `out/analysis.json`, `out/analysis_stdout.txt`;
  `src/verify_target_independence.py` → `out/target_independence_check.txt`.
- Code: `src/` (`cluster_job.py`, `data_prep.py`, `smoke_test.py`,
  `spectral_optimizer.py` [copy], `weight_cov_optimizer_v2*.py` [copies],
  `download_numerai.py`), `run.sbatch`.
