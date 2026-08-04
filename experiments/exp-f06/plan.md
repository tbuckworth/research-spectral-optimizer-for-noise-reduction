# exp-f06 — EU-1: Pre-flight (--qos=debug) — throughput/VRAM/realized rank grid + B1 additions (CLUSTER, 1× L40)

**Component**: #8 in `<run-dir>/decomposition.md` (lambda 0.18, P=0.70) — READ ITS FULL SECTION ("Component 8") plus amendments **D1, D4, D6, D7** in the "Round-2 Challenge Addendum", and the EU-1 row of the Experiment-Unit Packing Plan. Also carries component #6's short-schedule stability check and the on-cluster #5 identity guard. Read the frozen criteria's compute-feasibility and A2/A4 sections in `<run-dir>/success-criteria.md`.

**Run dir**: `<run-dir>` = `/media/titus/big/researcher-output/2026-08-03-followup-follow-up-spectral-optimizer-redo-pxp-filter-walk-`

## Inputs (must exist before launch)

- `<run-dir>/experiments/exp-f01/out/protocol-draft.json` — fold boundaries (EU-1 uses the LARGEST shard: fold-3 refit = train+embargo+valid rows), the fold-1 VALID slice for the convergence run's proxy gate ratio, and the **D7 per-fold VALID-vs-TEST offsets** that calibrate the proxy. Read exp-f01/results.md.
- `<run-dir>/experiments/exp-f02/src/spectral_filter.py` — the run's canonical filter copy (with the CPU-fp64 eigh fallback wired and A8/B3/D1 diagnostics module beside it). Read exp-f02/results.md, especially its "interface notes" (local torch is 2.11; cluster pins 2.5.1 — watch for any flagged API drift).
- `<run-dir>/experiments/exp-f03/results.md` + src/ — the per-phase persistence pattern (D6 requires EU-1's phases individually persisted so an interrupted job resumes at the failed phase, not from scratch).
- Parent code (READ-ONLY, adapt into your src/): parent `experiments/exp-001/src/` (data prep), `exp-003/src/` (shard build, sweep harness) at `/media/titus/big/researcher-output/2026-07-31-spectral-optimizer-for-noise-reduction-on-financial-timeseri/`.

## Task

One Slurm submission (EU-1) on the MATS free `compute` partition, 1× L40, `--qos=debug` (target < 2 h wall). Everything below is packed into that single job as **ordered, individually persisted phases (D6)** — each phase writes its results to NFS as it completes:

**Phase 0 — staging (before sbatch, from local)**: Check whether the parent run already staged the v5.0 parquet on the cluster (look under `/mnt/nw/home/t.buckworth/researcher-runs/2026-07-31-spectral-optimizer-for-noise-reduction-on-financial-timeseri/` and any data/ inside it) — reuse if present; otherwise rsync the needed parquet from `<run-dir>/data/` (they are symlinks — rsync with `-L`). Stage code to `/mnt/nw/home/t.buckworth/researcher-runs/<run-id>/exp-f06/`. Inside the job, build shards on `/ephemeral/t.buckworth/` (wiped on reboot — copy small outputs back to NFS; HF_HOME=/ephemeral/t.buckworth/hf).

**Phase 1 — shard build**: full v5.0 feature set (~2.3k features), int8 residency, the fold-3 refit shard (largest) + the fold-1 TRAIN/VALID shards needed by the convergence run. Record build time, shard sizes, /ephemeral usage.

**Phase 2 — timing table**: s/step for arm A (plain AdamW) and arm B (SpectralGradientFilter, `normalize="none"`) at rank ∈ {8, 32, 128, 512, 2048} at BOTH architectures: planned (~600k params) and fix-ladder rung-2 (~2.5–3M params, e.g. 2304-1024-256-1; basis p×k at k=2048 ≈ 24 GB — if it does not fit, record that fact, it decides whether A5 rung 2 is available). Also time the **eigh-every-N-steps** and **GPU-fp32-eigh** variants at the expensive ranks (B1(iii) — named documented variants). Record VRAM peaks per cell. **D4**: record kept-norm fraction per grid point (feeds the stage-1 LR correction rule).

**Phase 3 — identity re-assert + short-schedule stability**: on-cluster alpha=0 fp64 identity check (guards torch-2.5.1 compat); 500-step runs at grid extremes (rank 8 and largest feasible; adaptive effrank) — no NaN/Inf, eigh-fallback firing count logged, realized k(t) of adaptive configs + basis-rotation-rate diagnostic logged (feeds A2 checkability and #2's design).

**Phase 4 — full-length arm-A convergence run (LAST, D6)**: generous step budget, checkpointed, VALID-score-vs-step series streamed to NFS, plateau-detection stop. Yields: (i) **measured steps-to-convergence** (the anchor for packing arithmetic and the refit stopping rule — fixed OUTSIDE the 12 trials per A4); (ii) **proxy gate ratio**: arm-A VALID-slice mean per-era corr vs the example model's VALID-slice mean, mapped through the D7 offset from protocol-draft.json. **Pre-authorized fallback**: if phase 4 outgrows the debug window, it continues as EU-1b (normal queue, `--time=6h`, resumable, charged to the EU-5 reserve) — the sbatch/checkpoint structure must make that continuation trivial. If truncated, record the checkpointed lower bound + the pre-registered extrapolation rule.

## Cluster workflow (compute profile, binding)

1. Write code + `run.sbatch` locally under `<run-dir>/experiments/exp-f06/`; `#SBATCH` lines first: `--partition=compute --qos=debug --gres=gpu:1 --time=02:00:00 --job-name=spectral-eu1 --cpus-per-task=8 --mem=32G --output=logs/slurm-%j.out`. Inside: `source ~/venv/bin/activate`, `python -u`. NEVER upgrade torch (cluster pins 2.5.1+cu121).
2. `rsync -avP` (add `-L` for the data symlinks if staging data) to `mats:/mnt/nw/home/t.buckworth/researcher-runs/2026-08-03-followup-follow-up-spectral-optimizer-redo-pxp-filter-walk-/exp-f06/`.
3. BEFORE submitting: `ssh mats 'squeue -u t.buckworth'` — if a job named spectral-eu1 is already queued/running, resume waiting instead of resubmitting. Then `ssh mats 'cd .../exp-f06 && mkdir -p logs && sbatch run.sbatch'` (mkdir logs BEFORE sbatch).
4. Poll `ssh mats 'squeue -u t.buckworth'` sleeping 60 s between polls; PD (Resources) waits of tens of minutes are normal. Do NOT end while the job is queued/running; if you must stop, record the job id in results.md.
5. Pull results back with `rsync -avP mats:.../exp-f06/ <run-dir>/experiments/exp-f06/` and copy the full slurm log into `run.log`. Do NOT pull multi-GB shards back — metrics/JSON/CSV only (exclude shards in the rsync).
6. `ssh mats 'scontrol show job <id>'` for stuck jobs; an empty .out on a running job is stdout buffering; nvidia-smi in-job lists all 8 GPUs but you own only $CUDA_VISIBLE_DEVICES.

## Pass criterion
Arm A trains at full scale; convergence run plateaus within the job (or EU-1b continuation documented and submitted per D6, charged to EU-5); realized rank grid keeps ≥ 4 points including ≥ 512 (else the A2 kill-scope wording is triggered and recorded in results.md for immediate pre-registration); VRAM ≤ 44 GB; no unexplained NaN; second-architecture + eigh-variant timings recorded; kept-norm fractions recorded; proxy gate ratio computed with the D7 calibration.

## Fail criterion
Full feature set unfittable even with int8 residency + reduced batch (→ documented feature reduction, mandated limitation), or arm B > 10× arm A at every rank ≥ 32 even under the timed variants (→ symmetric trial shrink; if that still fails, FAIL-on-affordability with the exact resource ask).

## Outputs
- `out/eu1-timing.json` (the full timing/VRAM/kept-norm table), `out/eu1-stability.json` (fallback counts, k(t), rotation rates), `out/eu1-convergence.json` (steps-to-convergence, VALID curve file, proxy gate ratio raw + calibrated), realized-rank-grid statement.
- `results.md`: PASS/FAIL, the realized rank grid, measured steps-to-convergence, proxy gate ratio (calibrated), whether rung-2 architecture is available for the A5 ladder, eigh-variant policy inputs, VRAM table summary, and explicit flags for anything that triggers A2 wording. Full slurm log → `run.log`.

## Constraints
- NEVER modify files outside `<run-dir>` locally; on the cluster, work ONLY under `/mnt/nw/home/t.buckworth/researcher-runs/<run-id>/` and `/ephemeral/t.buckworth/`.
- Never run heavy work in an ssh shell on the dev node — everything compute goes through sbatch.
- Parent's cluster dir is READ-ONLY (reuse staged data by absolute path or copy; never modify).
- No new venvs; the cluster job uses `source ~/venv/bin/activate`.
