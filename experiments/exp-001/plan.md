# exp-001: Gate bundle — integration, data/protocol, throughput, spectral engagement

**Lambda driver**: Component #1, spectral engagement on financial gradients (P=0.40, lambda=1.22).
**Bundled components**: #6 (integration, λ=0.38), #9 (Numerai data + protocol, λ=0.16), #7 (vmap throughput, λ=0.33), #1 (spectral engagement, λ=1.22 — the headline pass/fail).
**Fail semantics**: #1-degeneracy is a HARD FAIL-FAST trigger (see below). #6/#9/#7 sub-failures are fixed in place (the code is ours and small), not fail-fast.

## Scientific question

Does the Spectral Optimizer's mechanism *engage* on real financial gradients —
i.e., does the per-sample gradient similarity eigenspectrum have structure
separable from the Marchenko-Pastur bulk, such that the filter does real,
selective work (neither passing ~everything nor ~nothing)? This is the
precondition for the whole helps/hurts comparison. The prior project's README
documents that the filter hurts/degenerates when signal is weak and not
gradient-dominant (sparse parity), and low-SNR financial regression is
plausibly exactly that regime; Feldman 2019 long-tail theory predicts
consensus filtering may suppress rare genuine signal. Outcome genuinely
uncertain.

## Optimizer identity (amendment F11 — binding)

"The Spectral Optimizer" = `SpectralConsensusFilter` in
`/home/titus/pyg/optimizers/experiments/spectral_optimizer.py` — the exact
B×B per-sample-gradient similarity eigendecomposition with Marchenko-Pastur
thresholding (hard/soft/variance modes), wrapping a base optimizer (AdamW).
COPY the needed files into this experiment directory (do NOT modify anything
under `/home/titus/pyg/optimizers` — it is read-only reference). The
streaming variant `weight_cov_optimizer_v2.py` may be smoke-tested in #6, but
a #7-forced swap to it is a SCOPE CHANGE that must be reported as such in
results.md, never a silent fallback.

## Sub-component tasks and pass criteria

### #6 — Regression integration smoke (local CPU)

Wrap AdamW with SpectralConsensusFilter in a fresh training loop outside the
home repo: synthetic regression (y = Xw + noise, Numerai-like shapes: ~300
features, continuous ~5-bin-style target), 2-layer MLP, per-sample MSE
loss_fn, 200 steps. Also run the streaming variant once.

PASS requires ALL of:
- Loss decreases; spectral diagnostics (eigenvalues kept, grad-norm fraction
  passed) are emitted; both variants run without exceptions.
- Filter-off path reproduces plain AdamW to numerical tolerance.
- **C1 spiked-covariance unit test** (verifies diagnostics are RIGHT, not
  merely emitted):
  (a) pure i.i.d.-noise gradients → filter keeps ≈0 directions;
  (b) planted spike (coherent direction) → filter keeps it;
  (c) i.i.d. noise + correlated zero-signal confound → MEASURE whether the
  filter keeps the confound (report the number; this calibrates the
  "engaged ≠ meaningful" caveat).

FAIL: interface cannot accept a regression loss / new loop without >2 h of
code surgery. (Response: fix in place; trade surgery cost against budget.)

### #9 — Numerai data + era-purged protocol (local, then rsync one shard)

Download Numerai public dataset (v5) train+validation parquet locally (numerapi
or direct URLs). Parse eras; choose a feature/era subsample that loads <32 GB.
Build the purged split with:
- **F1 (binding)**: validation-side eras split into a TUNING block and a
  temporally later, embargoed FINAL-VERDICT block. No tuning, threshold-mode
  selection, or peeking ever touches the verdict block. Record the block
  boundaries in results.md — downstream experiments must reuse them.
- **F5 (binding)**: embargo ≥ ceil(target_horizon_days / era_spacing_days)
  eras between train/tuning/verdict blocks.
- **F9 (binding)**: recompute the corr sanity band and leakage gate from
  CURRENT Numerai v5 example-model validation stats (not 2021 forum numbers);
  record the recalibrated band for exp-003.

PASS: ≥100 usable validation-side eras total across the two blocks after
purging; loads <32 GB; one shard rsynced to
`/ephemeral/t.buckworth/` on the cluster and readable in the debug job.

### #7 — vmap throughput on L40 (shared --qos=debug GPU job)

Time 200 steps at B ∈ {256, 1024}, filter-on and filter-off, on the real
(subsampled) Numerai data; extrapolate a full training run.

PASS: projected full training run <20 min at the batch size #1 needs.
FAIL response (pre-authorized): reduce B / subsample harder; a swap to the
streaming variant invokes F11 scope-change rules — report it, don't silently
do it.

### #1 — Spectral engagement (SAME debug GPU job, the headline)

Short training run (one seed, defaultish hyperparameters, subsampled eras,
~10-15 min GPU), logging every N steps, for EACH threshold mode
(hard/soft/variance), at B ∈ {256, 1024}:
- eigenvalues above the MP threshold (count kept vs B);
- fraction of gradient norm passed;
- **F6 (binding)**: all diagnostics under BOTH batch compositions —
  within-era minibatches AND mixed-era minibatches (two configs, same job).
  The main-comparison batch composition is then chosen and justified in
  results.md.
- **C2**: permuted-target null control — same diagnostics with targets
  permuted (pure noise by construction); compare spectra to calibrate where
  the MP bulk edge sits on this data's correlation structure.
- **C3**: two cosine diagnostics per logged step — (i) filtered-update vs
  plain mean-gradient cosine, (ii) filtered vs unfiltered update cosine.
- **C5**: era-identity probe of the kept subspace — does the kept gradient
  component predict within the current era much better than across eras?
  (Direct check on "coherent = era factor, not signal".)
- **F2 (binding)**: at the end, PRE-REGISTER the threshold mode for all
  downstream experiments from these engagement diagnostics ONLY — before any
  out-of-sample performance is seen. Record the choice and the rule used in
  results.md. Do NOT compute or report any validation/OOS performance of the
  filtered runs in this experiment.

PASS (headline): for at least one threshold mode, sustained over a
nontrivial fraction of training: 0 < eigendirections kept < B AND grad-norm
fraction passed in ~[0.1, 0.9], at B ∈ {256, 1024} under at least one batch
composition, AND C3 mean-gradient cosine ≤ ~0.95 (filter is not just
mean-gradient smoothing), AND C2 permuted-target spectra are distinguishable
from real-target spectra.

FAIL (headline — HARD FAIL-FAST): all modes degenerate (≈0% or ≈100% of
grad norm passed) at both batch sizes and both compositions, OR C3
mean-gradient cosine > 0.95 everywhere. If this fires: do NOT proceed to the
comparison track. Report the degeneracy characterized against the C2
permuted-target null and (qualitatively) the prior project's MNIST
label-noise spectra. The run pivots to a mechanistic-finding deliverable.

## Compute & workflow

Local machine: #6, #9, all analysis/plotting. Cluster (MATS, remote via
`ssh mats`): ONE shared `--qos=debug` GPU job for #7+#1 (~15-20 min GPU,
--partition=compute --gres=gpu:1 --cpus-per-task=8 --mem=32G, realistic
--time under 2h). Follow the staging workflow in the compute profile
(rsync to /mnt/nw/home/t.buckworth/researcher-runs/<run-id>/exp-001/,
mkdir -p logs BEFORE sbatch, poll squeue, rsync results back, copy the slurm
log to run.log). Use the cluster venv (source ~/venv/bin/activate, torch
2.5.1 pinned — do not install/upgrade torch); HF/dataset scratch under
/ephemeral/t.buckworth/. Numerai parquet download happens LOCALLY; rsync
shards to the cluster (no numerapi install on the cluster).

## Deliverables

- `results.md`: per-sub-component PASS/FAIL, headline verdict PASS/FAIL for
  #1, the F2 pre-registered threshold mode + rule, F1 block boundaries, F9
  recalibrated bands, C1 unit-test table, C2 spectra comparison, C3 cosine
  traces, C5 probe result, measured step times and projected run time,
  diagnostics plots/CSVs (small files only).
- `run.log`: full slurm stdout/stderr of the debug job.
- Code used (scripts + sbatch file) left in this directory.
- Clean up: no model checkpoints, no files >50 MB in this directory. The
  Numerai parquet may live locally OUTSIDE the experiment dir (e.g.
  `<run-dir>/data/` is fine) and on /ephemeral — record paths in results.md.
