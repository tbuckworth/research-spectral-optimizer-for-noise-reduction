# exp-f03 — Component #7: Fold-job resumability kill-test + refit checkpoint roundtrip (LOCAL CPU, zero GPU) [B2]

**Component**: #7 in `<run-dir>/decomposition.md` (lambda 0.22, P=0.85) — READ ITS FULL SECTION ("Component 7"). Also read the B2 amendment description in the change log at the top of that file.

**Run dir**: `/media/titus/big/researcher-output/2026-08-03-followup-follow-up-spectral-optimizer-redo-pxp-filter-walk-`
**Parent sweep harness (READ-ONLY source)**: `/media/titus/big/researcher-output/2026-07-31-spectral-optimizer-for-noise-reduction-on-financial-timeseri/experiments/exp-003/src/sweep.py` (and `analyze.py` beside it). Copy what you adapt into `<run-dir>/experiments/exp-f03/src/`; never edit the parent's files.
**Filter**: use the run's canonical copy at `<run-dir>/experiments/exp-f02/src/spectral_filter.py` if it exists yet; otherwise copy `/home/titus/pyg/optimizers/spectral_filter.py` to your own src/ (READ-ONLY original). exp-f02 may be running concurrently — do not write into exp-f02/.

## Task

Local CPU only. All work inside `<run-dir>/experiments/exp-f03/` (src/, out/, run.log).

The fold jobs (EU-2/3/4) are multi-hour batched Slurm submissions on a shared partition; B2 requires that a `--time` kill or node failure costs one trial, not one fold. Verify the machinery BEFORE any fold submission:

1. **Per-trial persistence kill-test**: adapt the parent exp-003 sweep harness so each trial's result is durably appended to disk (JSON-lines or per-trial files + fsync) the moment the trial completes. Run it on 6 synthetic fast trials; SIGKILL the process mid-trial-4 (from a supervisor script — actually send SIGKILL, do not simulate); verify trials 1–3 are intact on disk; restart; verify the resume runs exactly trials 4–6 and never re-runs a completed trial (no double-count, no duplicate rows).

2. **Refit checkpoint roundtrip including filter state**: train a small model with `SpectralGradientFilter` active for N steps; checkpoint model + base-optimizer state + FILTER state (streaming basis, any EMA/covariance factors, step counters) + RNG streams (torch, numpy, python, and any separate generators); restore into a fresh process; verify bit-continuation for 10 further steps in fp64 (trajectories identical to an uninterrupted run). If the filter class has no built-in state_dict, write save/restore helpers in src/ (these become the fold jobs' checkpoint code — a deliverable).

3. **Resume-on-restart discipline**: demonstrate the combined pattern the fold jobs will use — on start, scan persisted trials, skip completed, resume any checkpointed refit — as one runnable harness entrypoint (deliverable code).

## Pass criterion
Kill-test passes (completed trials survive SIGKILL, resume runs exactly the remainder); checkpoint roundtrip is bit-continuous in fp64 including filter state; resume never re-runs a completed trial.

## Fail criterion
Persistence or restore cannot be made reliable within ~2h of surgery → per the decomposition, the consequence is structural: fold jobs must be split into sweep-job + refit/eval-job submissions from the outset (report this clearly; it changes EU packing).

## Constraints
- NEVER modify files outside `<run-dir>`. Parent run dir and `~/pyg/` are READ-ONLY.
- Write `results.md` with PASS/FAIL per sub-check, evidence (file listings before/after kill, trajectory diff numbers), and a short spec of the deliverable resume pattern for the fold jobs. Full stdout to `run.log`.
