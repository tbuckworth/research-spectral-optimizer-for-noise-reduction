# Experiment f03 Results

## Component Tested
Component #7 (decomposition, round 2, B2): Fold-job resumability — per-trial
persistence kill-test, refit checkpoint roundtrip including
`SpectralGradientFilter` state, and the combined resume-on-restart discipline
for the EU-2/3/4 fold jobs. Local CPU only (pre-authorized zero-GPU Wave-1
component); no cluster, no Slurm, no GPU touched (`CUDA_VISIBLE_DEVICES=""` in
every worker process, device hardcoded `cpu`).

## Verdict: PASS

All three sub-checks pass:

| Sub-check | Result |
|-----------|--------|
| 1. Per-trial persistence kill-test (actual SIGKILL mid-trial-4) | PASS |
| 2. Refit checkpoint roundtrip, fp64 bit-continuation incl. filter state | PASS |
| 3. Resume-on-restart discipline (skip completed, resume checkpointed refit) | PASS |

## Setup
- Environment: Python 3.12.3, torch 2.11.0+cu128 (CPU-forced), numpy 1.26.4,
  host titus-MS-7A59. `torch.set_num_threads(1)`,
  `torch.set_default_dtype(torch.float64)` in all workers.
- Duration: ~90 s total wall (kill-test ~25 s across two attempts, roundtrip
  ~10 s), well under the 30-min target.
- Resources used: local CPU only, ~280 KB of artifacts under `out/`.
- Filter source: `exp-f02/src/spectral_filter.py` did not exist yet
  (concurrent run), so per the plan the canonical
  `~/pyg/optimizers/spectral_filter.py` was copied to
  `exp-f03/src/spectral_filter.py` (read-only original untouched).

## What Was Tested

**Finding that motivated the adaptation**: the parent exp-003 `sweep.py`
(READ-ONLY source) accumulates all trial results in an in-memory `trials`
list and writes `sweep_results.json` once at the end of `main()`. Its "append
behaviour" does **not** survive SIGKILL — a `--time` kill would lose every
completed trial of a fold job. The B2 requirement is therefore a real change,
not a verification of existing behaviour.

The adapted harness (`src/harness.py`, keeping the parent's grid →
`train_one` → record structure on 6 synthetic fp64 trials) appends each
trial's record to `out/trials.jsonl` via `src/trial_store.py` — a single
`os.write` on an `O_APPEND` fd, then `fsync` of file and directory — the
moment the trial completes, and on start scans the store and skips completed
trial ids. A supervisor (`src/kill_test.py`) launched it, waited for the
`trial_4.started` marker, and sent an **actual SIGKILL** (verified
`rc == -9`) 0.5 s into trial 4 (trials run ~2.5 s each); it then verified
on-disk state, restarted the harness, and verified the resume. The same
supervisor killed a `SpectralGradientFilter` refit mid-run (after the step-20
checkpoint) and verified restart-resume-completion.

The roundtrip test (`src/roundtrip_compare.py` + `src/refit_worker.py`) ran
three **separate processes**: (a) an uninterrupted 35-step fp64 reference run
of a small MLP trained with AdamW + `SpectralGradientFilter(rank=8, warmup=5)`
recording the full flat parameter vector and loss for steps 26–35; (b) a
25-step run checkpointing model + AdamW state + full filter state (V, S,
proj_k, step_count, grad_mean, hparams) + all four RNG streams (torch global
CPU, dedicated data-order generator, numpy, python) via `src/checkpoint.py`
(atomic tmp+fsync+rename); (c) a fresh process **deliberately initialized
with a different seed (999)**, restored from the checkpoint, run 10 further
steps. The filter class has no `state_dict`; `src/checkpoint.py` supplies the
save/restore helpers and is a deliverable for the fold jobs.

## Results

### Raw Output

Kill-test (from `run.log`):
```
CHECK [killed-by-sigkill]: PASS rc=-9
trials.jsonl after kill: ids=[1, 2, 3], torn_lines=0
CHECK [survivors-1-2-3]: PASS ids=[1, 2, 3]
CHECK [resume-skips-1-2-3]: PASS skipped=[1, 2, 3]
CHECK [resume-runs-4-5-6]: PASS ran=[4, 5, 6]
CHECK [final-ids-1..6-no-dupes]: PASS ids=[1, 2, 3, 4, 5, 6]
CHECK [no-completed-trial-rerun]: PASS started-in-both=[4] (allowed only for the killed in-flight trial)
CHECK [refit-killed-by-sigkill]: PASS rc=-9
[refit] RESUMED from checkpoint at step 20 (filter step_count=20)
[refit] COMPLETE at step 60 (resumed_from=20) final loss 0.935735
KILL-TEST RESULT: ALL PASS
```

Roundtrip (all 10 steps identical; first and last shown):
```
  step N+ 1: bitwise_equal=True  max|dparam|=0.000e+00  loss_ref=1.0219518443441875 loss_res=1.0219518443441875 loss_equal=True
  ...
  step N+10: bitwise_equal=True  max|dparam|=0.000e+00  loss_ref=0.85722033643986517 loss_res=0.85722033643986517 loss_equal=True
CHECK [trajectory-bit-continuous-10-steps]: PASS
CHECK [filter-V-bitwise]: PASS shape ref=(353, 8) res=(353, 8)
CHECK [filter-S-bitwise]: PASS
CHECK [filter-grad_mean-bitwise]: PASS
CHECK [filter-step_count]: PASS ref=35 res=35
ROUNDTRIP RESULT: ALL PASS
```

### Metrics
| Metric | Target | Actual | Pass? |
|--------|--------|--------|-------|
| Process killed by real SIGKILL | rc = −9 | rc = −9 (both sweep and refit kills) | Y |
| Trials on disk after mid-trial-4 kill | exactly {1,2,3}, parseable | {1,2,3}, 0 torn lines | Y |
| Resume runs | exactly {4,5,6}, skips {1,2,3} | ran [4,5,6], skipped [1,2,3] | Y |
| Duplicate trial records | 0 | 0 (6 unique ids 1..6) | Y |
| Completed trial re-run on resume | never | none (only killed in-flight trial 4 re-ran) | Y |
| Refit resume after kill | resumes from checkpoint > 0, completes | resumed_from=20, completed 60/60 | Y |
| 10-step continuation, fp64 params | bitwise identical | bitwise_equal=True all 10 steps, max\|Δparam\| = 0.0 | Y |
| Losses over continuation window | identical | equal to 17 significant digits, all 10 steps | Y |
| Filter state after continuation (V, S, grad_mean, step_count) | bitwise identical | all bitwise equal (V: 353×8) | Y |
| Surgery time budget | reliable within ~2 h | ~1 h including one kill-timing fix | Y |

### Analysis

The pass criterion is met on all three legs. The kill was real (SIGKILL,
rc −9, sent by a supervisor process), landed mid-trial-4, and cost exactly the
in-flight trial: trials 1–3 were durably on disk with no torn lines, and the
restarted harness ran exactly 4–6 with zero duplicate records. The refit leg
showed the same semantics one level up: a kill after the step-20 checkpoint
cost 0–9 steps of work, and the restart resumed from step 20 and completed.

The checkpoint roundtrip is bit-continuous in fp64 across a **process
boundary** with a deliberately mis-seeded fresh initialization before restore
— so the checkpoint demonstrably carries *all* trajectory-determining state:
model params, AdamW moments, the filter's streaming basis V, singular values
S, grad_mean EMA, step counter, and every RNG stream. Ten continuation steps
matched the uninterrupted reference bitwise in parameters, losses, and final
filter state. Note the sanity condition for exactness on the cluster: fixed
thread count (the harness pins `torch.set_num_threads(1)` for the fp64
verification; the GPU fold jobs need the analogous determinism flags only if
bit-level — rather than statistical — resume fidelity is demanded there).

One negative finding of record: the parent harness as written would **not**
have survived a Slurm kill (results held in memory until end-of-run), which
confirms the decomposition's decision to treat B2 as a component with a
kill-test rather than an assumption. The consequence for EU-2/3/4 packing is
the good branch of the fail criterion: fold jobs do NOT need to be split into
sweep-job + refit/eval-job submissions; the resumable single-submission design
stands.

### Deliverable resume pattern for the fold jobs (spec)

1. **Per-trial persistence** (`src/trial_store.py`): append one JSON line per
   completed trial via single `os.write` on an `O_APPEND` fd + `fsync(fd)` +
   `fsync(dirfd)`, keyed by `trial_id`. A torn *trailing* line (kill
   mid-write) is detected and dropped on load; a torn line anywhere else
   raises.
2. **Refit checkpoints** (`src/checkpoint.py`): every K steps save
   `{step, model.state_dict, optimizer.state_dict, filter_state_dict
   (V, S, proj_k, step_count, grad_mean, hparams), rng (torch CPU + data
   generator + numpy + python)}` — written atomically (tmp → fsync →
   `os.replace` → dir fsync) so a kill mid-checkpoint preserves the previous
   checkpoint. Restore asserts hparam equality before loading.
3. **On every (re)start** (`src/harness.py --phase all`): scan
   `trials.jsonl`, skip completed trial_ids, run the remainder; then, if
   `refit_done.json` absent, resume the refit from `refit_ckpt.pt` if
   present, else start fresh; write the done-marker with fsync on
   completion. Marker files (`state/trial_<id>.started`) flag in-flight
   trials. This also defends against duplicate submissions after
   orchestration interruptions (the scan is idempotent).
4. On GPU fold jobs, add `map_location` handling and (if bit-fidelity is
   required) CUDA RNG state + determinism flags; statistically-faithful
   resume needs only items 1–3.

## Unexpected Observations
- **Attempt 1 kill-timing miss** (preserved in `run.log` appendix): at
  TRIAL_STEPS=400 the synthetic trials ran ~0.25 s each, so the supervisor's
  0.5 s post-marker delay let trials 4–5 complete and the kill landed in
  trial 6. The persistence machinery itself behaved correctly even in that
  run (5 survivors, 0 torn lines, resume ran exactly the remainder, no
  duplicates); only the scripted kill placement was off. Fixed by lengthening
  trials to ~2.5 s (TRIAL_STEPS=4000). Incidentally this was a second,
  differently-placed successful kill/resume cycle.
- `fsync` cost is negligible at per-trial granularity (6 records, ~1 KB
  total) — no reason to batch writes on NFS for jobs with 12-trial sweeps.
- The refit-done marker check (`refit_done.json`) matters: without it a
  restarted job would re-enter the refit loop after completion.

## Implications

What this tells us: the criterion was met — a `--time` kill or node failure
on a fold job costs one in-flight trial (or ≤ K refit steps), not the fold,
and the `SpectralGradientFilter`'s undocumented state (streaming basis,
grad_mean EMA, counters) is fully and exactly checkpointable with the
`src/checkpoint.py` helpers. Next steps: EU-2 (fold-1 job) adopts this
pattern directly — `trial_store.py` + `checkpoint.py` + the
`harness.py --phase all` restart discipline port into the fold-job script;
`--time` set at 2–3× projection per B2; folds 2–3 re-projected from fold-1
realized wall before submission. No sweep-job/refit-job split is needed, so
EU packing is unchanged.
