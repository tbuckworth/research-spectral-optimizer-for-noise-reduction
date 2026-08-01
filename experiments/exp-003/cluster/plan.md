# exp-003: Baseline sanity, power analysis, budget fit, and F4 go/no-go gate

**Bundled components**: #8 (baseline sanity, λ=0.24), #3 (statistical power, λ=1.02 — the lambda driver), #4 (budget fit, λ=0.89).
**Fail semantics**: a LEAKAGE signature in #8 is a HARD FAIL-FAST trigger — fix the purge/embargo before anything downstream runs; nothing is interpretable until fixed. Power/budget failures re-scope per F4/F7 (pre-authorized), they do not stop the run.

## Prerequisites (read first)

- `../exp-001/results.md` — REQUIRED. Reuse verbatim: the F1 tuning/verdict
  block boundaries, the F5 embargo size, the F9 recalibrated corr sanity band
  and leakage gate, the F2 pre-registered threshold mode, the measured step
  times and projected run time (#7), and the chosen batch composition. Do NOT
  re-derive or alter any of these.
- `../../challenge/limitation-triage.md` — binding amendments F1, F3, F4, F5,
  F7, F9, F12.
- Data: the Numerai subsample and split artifacts produced by exp-001 (paths
  recorded in exp-001/results.md, locally under `<run-dir>/data/` and on the
  cluster under `/ephemeral/t.buckworth/`; re-rsync the shard if /ephemeral
  was wiped).
- Hyperparameter priors: `/home/titus/pyg/optimizers/README.md` (F12 — the
  spectral-arm search center for later experiments; document the transfer).

## Sub-component tasks and pass criteria

### #8 — Tuned-AdamW baseline sanity (one GPU array-style job)

Small hyperparameter sweep for the plain-AdamW MLP arm: 8-12 trials over
LR / weight-decay / dropout (~1-2 min each on the subsampled eras), evaluated
by mean per-era Spearman correlation, plus a zero-predictor control.

- **F1 (binding)**: the sweep touches TUNING-block eras ONLY. The verdict
  block is never loaded, evaluated, or peeked at in this experiment.
- **F12 (binding)**: define the tuning match in compute; record trials × time.
  Log under-exploration signatures: best-config-on-grid-boundary,
  non-monotone sweep shape. These carry into exp-004's verdict rules.
- **F9**: judge the result against the recalibrated band from exp-001, not
  2021 forum numbers.
- **F5**: as soon as per-era corrs exist, compute lag-1 autocorrelation of
  the per-era corr series for the best trial; record it (it sets the block
  length below).

PASS: best trial's mean per-era corr inside the F9-recalibrated sane band on
tuning-block eras; zero-predictor |corr| < 0.002; no leakage signature.
FAIL modes:
- All trials ≈ 0 (< 0.003): reduce subsampling and/or switch to a
  corr-aligned loss (held fixed across arms downstream), retry ONCE.
- Any trial far above band (leakage signature): HARD FAIL-FAST — diagnose and
  fix the purge/embargo first; document the defect and the fix; re-run the
  sweep only after the fix.

### #3 — Statistical power of the paired per-era design (local CPU)

Bootstrap power simulation from the best trial's per-era corr vector:
- **F5 (binding)**: use a MOVING-BLOCK bootstrap with block length ≥ the
  target-horizon overlap implied by the lag-1 autocorrelation measured in #8
  (and ≥ the embargo-derived overlap from exp-001). Naive iid bootstrap is
  not acceptable for the headline numbers (may be shown as a comparison).
- Simulate paired differences under (a) the null and (b) an injected effect
  at the F3 threshold, with realistic seed noise at 3 seeds.
- **F1**: power is computed for the VERDICT block's era count (from
  exp-001's block boundaries), not the tuning block's.
- **F3 (binding, pre-register here)**: verdict threshold =
  min(0.005, 0.25 × realized tuned-baseline mean per-era corr). Compute it
  from #8's result and RECORD IT in results.md before any arm comparison is
  ever unblinded. This number is frozen for exp-004.

PASS: block-bootstrap CI half-width ≤ ~F3-threshold achievable at 3 seeds ×
verdict-block era count (some verdict category reachable).
FAIL: CI half-width > ~0.008 using ALL available verdict-block eras. First
response is free (increase verdict-block era count if more eras exist);
otherwise the F4 gate below re-scopes the endpoint.

### #4 — Budget-fit arithmetic (no compute)

Using #7's measured step times (exp-001) and #8's per-trial times:
pack (tuning both arms) + (2 arms × 3 seeds) + C4 mechanism controls +
diagnostics [+ sequence arm if exp-002 passed] into the remaining experiment
slots at ≤25 min/job.
- **F7 (binding)**: use the inverted conditional cut order under pressure:
  Muon → GBT → sequence arm → GAF-style ablation → random-subspace control;
  NEVER cut seeds, matched tuning, or the verdict-block separation. The last
  mechanism-discriminating control (C4 random-subspace or GAF) is protected:
  if it doesn't fit as a full job, check array co-scheduling before treating
  it as a slot.

PASS: minimum viable design (+protected control) fits.
FAIL: cannot fit even after identical-across-arms subsampling → execute F7
cut order; if minimum viable still doesn't fit, record FAIL-on-affordability
with the exact resource ask.

### F4 — Joint go/no-go gate (local CPU, the exit criterion)

Joint block-bootstrap: given the realized baseline level, its per-era
dispersion/autocorrelation, and the verdict-block era count, estimate
P(at least one verdict category — helps / hurts / doesn't-help — is
reachable) under the frozen F3 threshold.

- ≥ 0.6 → GO: exp-004 proceeds as planned.
- < 0.6 → NO-GO: do NOT proceed to the seeded comparison as designed.
  Re-scope the endpoint (pre-authorized: honestly-scoped effect estimate
  with calibrated uncertainty instead of a categorical verdict) and record
  the re-scoped endpoint in results.md for exp-004 to implement.

## Compute & workflow

One GPU array-style job (`--qos=debug` if <2h, --partition=compute
--gres=gpu:1 --cpus-per-task=8 --mem=32G, logs/ created before sbatch),
staged per the compute-profile rsync workflow to
mats:/mnt/nw/home/t.buckworth/researcher-runs/<run-id>/exp-003/.
All of #3, #4, F4 are local CPU on the returned per-era corr vectors.

## Deliverables

- `results.md`: per-sub-component PASS/FAIL; best-trial config + mean per-era
  corr + zero-predictor corr; sweep table (all trials); F12 under-exploration
  signature check; lag-1 autocorrelation + chosen block length; the FROZEN F3
  threshold; power-sim table (CI half-widths, detection probabilities); #4
  packing arithmetic with the F7 cut decisions; the F4 gate number and
  GO/NO-GO decision; leakage-gate verdict.
- `run.log`: full slurm stdout/stderr.
- Per-era corr vectors as small CSVs; code + sbatch script in this directory.
- No files >50 MB; no checkpoints kept.
