# Steinhardt Decomposition

**Run**: 2026-07-31-spectral-optimizer-for-noise-reduction-on-financial-timeseri
**Question**: Does the existing Spectral Optimizer (per-sample gradient MP-spectral filtering, `/home/titus/pyg/optimizers/experiments/`) improve out-of-sample performance vs tuned AdamW on low-SNR financial prediction (Numerai), across an MLP and one sequence architecture?
**Compute profile**: MATS Slurm free `compute` partition, driven remotely; 1x L40 per job; each experiment <30 min GPU; max 5 experiments. Local machine = orchestration + CPU smoke tests + analysis only.

## Project Components

The project decomposes into nine testable components across four layers:

1. **Infrastructure** — the existing optimizer code integrates with a regression loop (#6), Numerai data downloads and yields a purged-era protocol (#9), an OHLCV fallback exists for the sequence arm (#5).
2. **Feasibility on cluster** — vmap per-sample gradients + B×B eigendecomposition run fast enough that a full training run fits under 30 min on one L40 (#7); the whole 2-arm × 3-seed × matched-tuning design fits in ≤5 experiment slots (#4).
3. **Scientific preconditions** — the filter actually *engages* on financial gradients rather than being a degenerate no-op (#1); the tuned-AdamW baseline lands in the sane per-era-corr band, proving the pipeline finds signal without leakage (#8); the paired per-era design has enough statistical power to support one of the three verdicts (#3).
4. **Scope extension** — vmap works through a recurrent/sequence model for the second-architecture arm (#2).

The single riskiest component is #1 (spectral engagement): the prior project's own results say the filter hurts or degenerates when signal is weak and not gradient-dominant, and low-SNR financial regression is plausibly exactly that regime. It is also cheap to test (~45 min including one short cluster job), so it has by far the highest information rate. Note the asymmetry: per `success-criteria.md`, a *degenerate-filter* result is itself a reportable mechanistic finding, so #1 failing downgrades the deliverable rather than killing it.

## Lambda Table

Ordered by descending lambda. T is wall-clock hours for the minimum viable pass/fail test (including SSH/rsync/queue overhead where a cluster job is needed), not full implementation time.

| # | Component | P_success | Evidence | T (hrs) | lambda | Quick Test | Dependencies | Status |
|---|-----------|-----------|----------|---------|--------|------------|--------------|--------|
| 1 | Spectral engagement on financial gradients | 0.40 | Prior repo README: filter hurts/degenerates on weak-signal tasks (sparse parity); Feldman 2019 long-tail counter-hypothesis; no empirical validation in low-SNR regression | 0.75 | 1.22 | Short training run logging eigenvalues-above-MP-bulk and grad-norm fraction passed, all threshold modes | #6, #9, #7 | PENDING |
| 2 | Sequence-arm vmap feasibility (recurrent model) | 0.35 | torch.func vmap incompatible with cuDNN RNN modules; needs functional/unrolled rewrite; flagged least-tested path in state.md assumption 2 | 1.0 | 1.05 | CPU: vmap(grad) through a hand-rolled GRU cell on within-era sequences; verify vs per-sample loop | #6 | PENDING |
| 3 | Statistical power of paired per-era design | 0.60 | Numerai per-era corr sd ≈ mean (practitioner forum); 3 seeds is the floor; era count drives power (success-criteria risk 2); no direct precedent at this exact scale | 0.5 | 1.02 | Bootstrap power sim from #8's per-era corr vector: achievable CI half-width vs the 0.005 verdict threshold | #8 | PENDING |
| 4 | Full design fits 5-experiment / 30-min budget | 0.80 | MLP-scale + vmap at moderate B comfortably sized for L40 (state.md scope answer), but matched tuning sweeps are the known squeeze (success-criteria risk 5) | 0.25 | 0.89 | Arithmetic from measured step time (#7): trials×time + 2 arms×3 seeds + diagnostics + sequence arm ≤ 5×30 min | #7, #8 | PENDING |
| 5 | OHLCV fallback dataset for sequence arm | 0.80 | Yahoo-style daily OHLCV freely downloadable, widely replicated; but no standardized splits — protocol must be built (success-criteria benchmarks table) | 0.5 | 0.45 | Download ~100 tickers daily OHLCV, build next-period return targets with strict temporal split; check coverage/NaN rates | None (only needed if #2's Numerai within-era framing fails) | PENDING |
| 6 | Spectral Optimizer regression integration | 0.75 | Code exists and worked in home repo (classification); state.md assumption 1: loss_fn swap to MSE + step(inputs, targets) interface are known small adaptations, untested | 0.75 | 0.38 | Local CPU smoke: wrap AdamW, 2-layer MLP, synthetic regression, 200 steps; loss falls, diagnostics emitted, no exceptions | None | PENDING |
| 7 | vmap throughput on L40 within job cap | 0.85 | vmap per-sample grads + B×B eigh at B≤1024 on MLPs is a well-trodden cost (~2-5x plain step, prior repo measured ~2x for streaming variant); cluster venv pins torch 2.5.1 which has torch.func | 0.5 | 0.33 | One `--qos=debug` job: measure steps/sec, project full training run time; must be <20 min | #6 | PENDING |
| 8 | Baseline sanity: tuned AdamW in sane corr band | 0.70 | Practitioner NN models reach mean per-era corr 0.015-0.03 (numerai_forum2021_eras); replicable but our era/feature subsample + small tuning budget may land low; leakage the other failure mode | 1.5 | 0.24 | Small LR/wd sweep (8-12 trials) on MLP, purged validation eras; mean per-era corr in 0.005-0.05, zero-predictor ≈ 0 | #9 | PENDING |
| 9 | Numerai data + era-purged protocol | 0.85 | Public free download via numerapi, heavily replicated by community; era structure documented; only integration effort at risk | 1.0 | 0.16 | Download v5 train+validation locally, parse eras, build purged split with embargo, check memory at chosen feature subset | None | PENDING |

No component qualifies for SKIP (none ≥ 0.9 with multiple replications of *our exact usage*) and none falls below 0.05.

## Component Details

### Component 1: Spectral engagement on financial gradients [lambda = 1.22]

**What**: The filter's mechanism requires the per-sample gradient similarity eigenspectrum to have *some* structure separable from the Marchenko-Pastur bulk — enough directions kept to train, enough filtered to matter. This is the scientific heart of the transfer question: does the coherence-amplifier mechanism even have something to grab onto in a low-SNR financial regime?
**Risks to this component**: (a) All eigenvalues inside the MP bulk — no coherent signal at feasible batch sizes, filter passes ~nothing or falls back to noise directions; (b) spectrum entirely outside the bulk — filter passes ~everything, a no-op; (c) engagement early in training that collapses as the signal-poor loss landscape flattens.
**Evidence for P_success = 0.40**: Theoretical support exists (Coherent Gradients, `chatterjee2020coherentgradients`; GAF found exploitable cross-batch agreement structure, `chaubard2024gradientagreementfiltering`) but zero empirical validation in low-SNR regression. The prior project's own README documents the failure mode directly: the filter hurts when signal is weak and not gradient-dominant (sparse parity), and financial data is the canonical weak-signal regime. Feldman 2019 predicts consensus filtering may suppress rare genuine signal. Rubric band 0.3-0.5 (theory, no empirical validation) — 0.40.
**Quick test**: After #6 and #9 pass, run one short training job (one seed, default-ish hyperparameters, subsampled eras, ~10-15 min GPU) with diagnostics logged every N steps: number of eigenvalues above the MP threshold, fraction of gradient norm passed, for each threshold mode (hard/soft/variance). Reuse the job from #7's throughput measurement if convenient — same run can serve both.
**Pass criterion**: For at least one threshold mode, sustained over a nontrivial fraction of training: eigendirections kept is strictly between 0 and B, and gradient-norm fraction passed lies in roughly [0.1, 0.9] — i.e., the filter is doing real, selective work.
**Fail criterion**: All modes degenerate (≈0% or ≈100% of gradient norm passed) throughout training, at every batch size tried (test at least B ∈ {256, 1024}).
**If the quick test returns FAIL**: The verdict experiments as designed become uninterpretable as helps/hurts, and the deliverable pivots to the mechanistic finding "no gradient-coherent signal separable from the MP bulk on Numerai at feasible batch sizes" — per success-criteria.md this is itself reportable, connected to the weak-signal failure mode and Feldman's theory. Budget shifts from the sequence arm toward characterizing *why* (spectra vs MNIST-label-noise comparison). The project narrows but does not die.

### Component 2: Sequence-arm vmap feasibility [lambda = 1.05]

**What**: The second-architecture arm requires per-sample gradients through a recurrent/sequence model. torch.func vmap does not compose with cuDNN-backed `nn.LSTM`/`nn.GRU`; a functional, unrolled cell implementation (or a small within-era attention model instead) is needed.
**Risks to this component**: vmap failure or unusable throughput through unrolled recurrence; plus the dataset problem — Numerai asset IDs reset each era, so the arm needs within-era sequence framing on Numerai or an OHLCV dataset (#5).
**Evidence for P_success = 0.35**: Flagged as the least-tested code path (state.md assumption 2; success-criteria risk 4). Known documented incompatibility between torch.func transforms and cuDNN RNN modules. Hand-rolled cells do work under vmap in principle, but unrolled per-sample gradients over sequences are memory-hungry and slow, and nobody has run this exact stack. Rubric band 0.3-0.5, at the low end given two independent ways to fail (vmap path + dataset framing).
**Quick test**: Local CPU (this is a correctness smoke test, not an experiment): implement a ~1-layer GRU cell as pure functions, vmap(grad(loss)) over a batch of short within-era sequences (synthetic Numerai-shaped data fine), verify per-sample gradients match a python-loop reference to ~1e-5, and time it. Optionally one 5-min `--qos=debug` GPU job to confirm throughput.
**Pass criterion**: Per-sample gradients correct vs loop reference AND projected full training run (sequence model, subsampled data) <25 min on one L40.
**Fail criterion**: vmap errors that require >2 h of surgery, or projected runtime over the job cap.
**If the quick test returns FAIL**: Drop to the MLP-only minimum viable contribution with the architecture-consistency claim explicitly removed (pre-authorized fallback in success-criteria risk 4). Frees 1-2 experiment slots for the GAF ablation or tail-era analysis. No effect on the primary verdict.

### Component 3: Statistical power of paired per-era design [lambda = 1.02]

**What**: The verdict definitions have teeth: "helps"/"hurts" need a paired per-era CI excluding zero at practical significance ≥0.005 mean per-era corr; "doesn't help" needs a CI narrow enough to *exclude* 0.005 improvements. With per-era corr sd on the order of the mean, the design must have enough eras (and the pairing must cancel enough era variance) for any of the three verdicts to be reachable.
**Risks to this component**: Era-level variance too high; seed-level variance non-negligible at 3 seeds; usable validation era count too small after purging/embargo and subsampling.
**Evidence for P_success = 0.60**: The paired design (same eras, same seeds across arms) is a known strong variance-reduction and Numerai has hundreds of eras — but our budget subsamples eras, and practitioner-reported per-era sd ≈ mean means the unpaired noise floor is large. Related power arithmetic exists (any paired-test literature) but not for this exact metric/scale. Rubric band 0.5-0.7.
**Quick test**: Pure CPU analysis after #8: take the per-era corr vector from the baseline run, bootstrap paired differences under (a) null and (b) injected +0.005 effect with realistic seed noise, and compute achievable CI half-width and detection probability at 3 seeds × available eras.
**Pass criterion**: CI half-width ≤ ~0.005 achievable with the era count the budget allows (so at least one of the three verdicts is always reachable).
**Fail criterion**: CI half-width > ~0.008 even using all available validation eras — no verdict category reachable.
**If the quick test returns FAIL**: First response is free: increase validation era count (era count, not model size, drives power and costs almost no GPU time — success-criteria risk 2). If still underpowered, the claim is scoped honestly to the CI the design supports, and the write-up reports an effect estimate with calibrated uncertainty rather than a categorical verdict. Downgrades, does not kill.

### Component 4: Full design fits the 5-experiment / 30-min budget [lambda = 0.89]

**What**: The minimum viable contribution needs: matched tuning sweep (both arms), main seeded comparison (2 arms × 3 seeds), diagnostics — in ≤3 slots — plus ≥1 slot for the sequence arm and ideally the GAF ablation, each job <30 min GPU on one L40.
**Risks to this component**: Tuning sweeps are the squeeze (success-criteria risk 5); vmap overhead larger than expected inflates per-run time; queue/rsync overhead eats wall-clock.
**Evidence for P_success = 0.80**: state.md scope answer sizes MLP+vmap workloads comfortably within a single L40; the prior repo measured ~2x Adam cost for the streaming variant; era/feature subsampling is pre-authorized. But matched-budget tuning for two arms in shared array-style jobs is exactly the kind of infrastructure that overruns. Rubric band 0.7-0.9.
**Quick test**: Pure arithmetic once #7's measured step time and #8's per-trial time exist: (tuning trials × trial time × 2 arms) + (6 seeded runs) + diagnostics + sequence arm, packed into array-style jobs, vs 5 × 30 min.
**Pass criterion**: The plan packs into ≤5 jobs each ≤25 min (buffer for queue variance), with ≥1 slot spare.
**Fail criterion**: Cannot fit even after subsampling eras/features identically across arms.
**If the quick test returns FAIL**: Execute the pre-registered cut order (success-criteria): Muon baseline first, then the GAF ablation, then the sequence arm — never seeds, never the matched tuning budget. If even minimum viable doesn't fit, record the exact resource ask as FAIL-on-affordability.

### Component 5: OHLCV fallback dataset [lambda = 0.45]

**What**: If the sequence arm can't be framed within-era on Numerai, it needs a public OHLCV next-period return-prediction dataset with a strict temporal split, purge, and embargo built from scratch.
**Risks to this component**: Data quality (survivorship bias, NaN gaps, splits/dividends), no standardized protocol so design errors are on us, and near-zero SNR making the arm uninformative.
**Evidence for P_success = 0.80**: Free daily OHLCV downloads are heavily replicated; the risk is entirely in protocol construction, which the literature review has already specified (temporal split + purge + embargo, per Numerai practitioner standards adapted). Rubric band 0.7-0.9.
**Quick test**: Local: download ~100 liquid tickers' daily OHLCV (10-20 y), compute next-period return targets, check NaN/coverage rates, build the temporal split, and confirm a trivial baseline (historical mean / zero-predictor) evaluates cleanly.
**Pass criterion**: ≥90% coverage after cleaning; split machinery produces leak-free train/val/test with embargo; dataset assembles in <30 min of work.
**Fail criterion**: Data quality issues requiring >2 h of cleaning, or no defensible protocol.
**If the quick test returns FAIL**: Sequence arm falls back to within-era framing on Numerai (small attention/sequence model over feature groups or within-era sample sequences); if that also fails via #2, MLP-only scope per Component 2's fallback.

### Component 6: Spectral Optimizer regression integration [lambda = 0.38]

**What**: Reuse `spectral_optimizer.py` (SpectralConsensusFilter) and `weight_cov_optimizer_v2.py` from `/home/titus/pyg/optimizers/experiments/` in a *new* training loop: per-sample MSE loss_fn (the code assumed classification), the `step(inputs, targets)` interface, wrapping AdamW, outside the home repo, under the cluster's pinned torch 2.5.1.
**Risks to this component**: Hidden classification assumptions (label dtype, loss reduction shape), import-path/coupling issues moving out of the home repo, torch.func API drift vs the torch version the code was written on.
**Evidence for P_success = 0.75**: The code exists and worked in its home repo (MNIST results in README); the author designed it to wrap any base optimizer; the needed adaptations are enumerated and small (state.md assumption 1). But "reusable as designed" is untested outside the repo, and integration is the classically underestimated step. Rubric band 0.7-0.9, low end.
**Quick test**: Local CPU (~50-line script): synthetic regression (y = Xw + noise, Numerai-like shapes: ~300 features, 5-bin-ish targets), 2-layer MLP, per-sample MSE loss_fn, wrap AdamW with SpectralConsensusFilter, run 200 steps; repeat for the streaming variant. Torch version locally should match or be checked against the cluster's 2.5.1 torch.func API.
**Pass criterion**: Loss decreases, spectral diagnostics (eigenvalue counts, norm fractions) are emitted, both variants run without exceptions, filter-off path reproduces plain AdamW to numerical tolerance.
**Fail criterion**: Interface cannot accept a regression loss / new training loop without >2 h of code surgery.
**If the quick test returns FAIL**: Everything downstream blocks. The response is to fix rather than abandon (the code is ours and small), but a large surgery cost must be traded against the experiment budget; if the exact per-batch variant is the blocker, the streaming variant is a designed substitute and vice versa.

### Component 7: vmap throughput on L40 within job cap [lambda = 0.33]

**What**: A full MLP training run — vmap per-sample gradients at B ∈ {256, 1024}, B×B eigendecomposition per step, subsampled Numerai eras, enough epochs to converge — must finish in well under 30 min on one L40 so that seeded/tuning jobs are packable (#4).
**Risks to this component**: vmap memory blow-up at large B × parameter count; eigh cost at B=1024; dataloading from NFS slower than compute.
**Evidence for P_success = 0.85**: MLP-scale per-sample gradients are the best-case for vmap; the prior repo measured ~2x Adam cost for the streaming variant; B×B=1024² eigh is milliseconds on an L40. Data staging to /ephemeral is prescribed by the profile. Rubric band 0.7-0.9, high end.
**Quick test**: One `--qos=debug` sbatch job (<10 min): time 200 steps at each B for filter-on and filter-off; extrapolate a full training run. Can share the job with #1's diagnostics run.
**Pass criterion**: Projected full training run <20 min at the batch size #1 needs.
**Fail criterion**: >30 min projected even after reducing width/epochs/features to the floor that still clears the sane-corr band.
**If the quick test returns FAIL**: Switch primary variant to the streaming rank-k filter (designed for exactly this), reduce B, or subsample harder — all pre-authorized. Only if all of these fail does this become a budget FAIL, with the exact resource ask recorded.

### Component 8: Baseline sanity — tuned AdamW in the sane corr band [lambda = 0.24]

**What**: A tuned-AdamW MLP on the purged Numerai protocol must land at mean per-era corr ≈ 0.005-0.05 with the zero-predictor at ≈0. This validates the *entire* task pipeline (data, loss, splits, metric) and is the reference arm of the main comparison.
**Risks to this component**: Landing at ~0 corr (subsampling destroyed the signal; loss/metric misalignment — plain MSE vs per-era corr is a known issue); landing far above the band (leakage through the purge/embargo — the study's key sanity check); tuning budget too small to find the band.
**Evidence for P_success = 0.70**: Thousands of practitioner replications put NN models in this band on the full dataset, but not at our subsample and 8-12-trial tuning budget; the loss-metric misalignment is documented (`numerai_forum2021_eras`). Rubric band 0.5-0.7 for *our exact configuration*, top end because failures are mostly diagnosable and fixable.
**Quick test**: One array-style sbatch job: 8-12 trials over LR/weight-decay/dropout (~1-2 min each on subsampled eras), plain AdamW, evaluate mean per-era Spearman on purged validation eras plus the zero-predictor.
**Pass criterion**: Best trial's mean per-era corr in [0.005, 0.05]; zero-predictor |corr| < 0.002.
**Fail criterion**: All trials ≈ 0 (< 0.003) or any far above band (> 0.06, leakage signature).
**If the quick test returns FAIL**: ~0 → reduce subsampling, switch loss toward corr-aligned objective (held fixed across arms), retry once. Leakage → fix the purge/embargo before anything else runs; no optimizer comparison is interpretable until this passes.

### Component 9: Numerai data + era-purged protocol [lambda = 0.16]

**What**: Download the free Numerai public dataset (v4.x/v5), parse era structure, build era-purged CV with an embargo gap, choose the feature/era subsample that fits a 32G-RAM job, stage to the cluster's /ephemeral scratch.
**Risks to this component**: API/registration friction, dataset size vs local disk and cluster scratch, era metadata surprises, package installs on the pinned cluster venv (mitigated by downloading locally and rsyncing parquet shards — no numerapi needed on the cluster).
**Evidence for P_success = 0.85**: Public free download, massively replicated by the community; the purged-CV recipe is documented (`numerai_forum2021_eras`). The residual risk is ordinary integration effort. Rubric band 0.7-0.9.
**Quick test**: Local: numerapi (or direct URL) download of train+validation parquet; load, count eras and rows, build the purged split with embargo, verify no era overlap, measure memory at the planned feature subset; rsync one shard to `/ephemeral/t.buckworth/` and load it in a trivial `--qos=debug` job (can piggyback on #7's job).
**Pass criterion**: ≥100 usable validation-side eras after purging at the chosen subsample; loads under 32 GB; shard readable on the cluster.
**Fail criterion**: Dataset not freely obtainable or unusable era structure (near-inconceivable given community replication — this is a checklist item, not a gamble).
**If the quick test returns FAIL**: Fall back to the OHLCV protocol (#5) for the whole study, with the protocol-construction burden that entails and the "leaderboard-equivalent" framing weakened — a significant but survivable scope change.

## Dependency Graph

```
  #6 Integration smoke ──────┬──────────────► #7 vmap throughput ──┐
      (local CPU)            │                  (debug GPU job)    │
                             │                                     ├─► #1 Spectral engagement ─► MAIN EXPERIMENTS
                             └─► #2 Sequence-arm vmap              │      (short GPU run)          (Step 8)
                                   (local CPU)                     │
                                     │  fallback if Numerai        │
                                     ▼  framing fails              │
                                 #5 OHLCV fallback                 │
                                   (local, conditional)            │
                                                                   │
  #9 Numerai data + protocol ───┬──────────────────────────────────┘
      (local + rsync)           │
                                └─► #8 Baseline sanity ─► #3 Power analysis
                                      (GPU array job)       (local CPU)
                                            │                    │
                                            └────────┬───────────┘
                                                     ▼
                                          #4 Budget-fit arithmetic
                                             (no compute)
```

## Parallelisation Plan

**Wave 1 (now, mostly local, no mutual dependencies)**: [#6, #9] — and #2 can start as soon as #6's import/loss plumbing exists (same session).
**Wave 2 (after #6 and #9)**: [#7 + #1 as one shared debug GPU job], [#8 as a parallel array job] — both submittable simultaneously (2 GPUs, within the 6-GPU cap).
**Wave 3 (after #8, local CPU)**: [#3], then [#4] once #7's timings are also in.
**Conditional**: [#5] only if #2 passes on vmap but the Numerai within-era framing fails.

Total pre-experiment gate time: ~4-6 h wall-clock, of which GPU time is two short debug jobs (~25 min combined) — this spends none of the 5 experiment slots if #7/#1/#8 are framed as validation runs (`--qos=debug`, <2 h), leaving all 5 slots for the tuned sweeps, the seeded main comparison, the ablation, and the sequence arm.

## Showstoppers

**None.** No component has P_success < 0.05, and nothing requires compute beyond one L40 per job — the workloads are MLP/small-sequence models with per-sample gradients at B ≤ 1024, comfortably inside the free partition. The two components most likely to fail (#1 at 0.40, #2 at 0.35) both have pre-authorized downgrade paths (mechanistic-finding pivot; MLP-only scope) rather than being fatal, per success-criteria.md.

## Overall Project P_success

**Full minimum-viable verdict** (clean helps/hurts/null with engaged mechanism, adequate power, in budget — components #6, #9, #7, #1, #8, #3, #4):

P = 0.75 × 0.85 × 0.85 × 0.40 × 0.70 × 0.60 × 0.80 ≈ **0.07**

**Informative publishable outcome** (recognizing that #1-fail converts to a reportable mechanistic finding and #3-fail converts to an honestly-scoped effect estimate — both explicitly countable deliverables per success-criteria.md; product over #6, #9, #7, #8, #4 only):

P = 0.75 × 0.85 × 0.85 × 0.70 × 0.80 ≈ **0.30**

**Solid (TMLR-tier) contribution including the sequence arm** multiplies the first figure by P(#2) = 0.35 → ≈ 0.03; the architecture-consistency claim should be treated as a stretch goal, not the plan of record.

(Assumes independence; actual probabilities are correlated — e.g., if #1 passes with strong engagement, #3 likely improves because effects are larger. The 0.07 figure is dominated by the genuine scientific uncertainty in #1 and #3, which is exactly where the uncertainty *should* be concentrated: infrastructure risk is low, and the open question is the science.)
