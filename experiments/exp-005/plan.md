# exp-005: Sequence-arm architecture-consistency test (conditional stretch goal — AUTHORIZED)

**Component**: #2 outcome (exp-002 PASS) extended to real data; #5 (OHLCV
fallback) is internal fallback only. Tests whether exp-004's verdict
transfers to a recurrent architecture on the SAME dataset.
**Authorization check (both required, both hold)**: exp-002 PASSED (vmap
through hand-rolled GRU verified); exp-003 #4 arithmetic left room ("exp-005
2-task array 13.7 min/task fits=True; F7 cuts: none").
**Fail semantics**: NOT fail-fast. On FAIL: cut per F7, note the removed
claim explicitly as a limitation, carry to Future Work as W2 with its
resource ask. Never blocks the run.

## Context you must read first

- `../exp-004/results.md` — the MLP verdict this experiment tests for
  architecture consistency: headline "hurts" (−0.00527 nc, CI [−0.00886,
  −0.00181]) downgraded per F12 to "no evidence of benefit under the
  affordable tuning budget"; C4 random-subspace control indistinguishable
  from the spectral filter.
- `../exp-004/src/verdict_job.py` + `src/analyze_verdict.py` — reuse this
  machinery (arms, diagnostics logging, per-era CSV output, bootstrap
  analysis) with the GRU model swapped in.
- `../exp-002/gru_vmap_test.py` — the verified hand-rolled functional GRU
  cell (pure tensor ops, composes with torch.func vmap(grad)); its
  `spectral_optimizer.py` is byte-identical to the source repo.
- `../exp-001/out/protocol.json` — F1 blocks (authoritative).
- `../exp-004/ref/` — f9_recalibration.json, protocol.json (copy to ref/).

## Design

**F13 (binding claim naming)**: run on **within-era Numerai framing** —
same shards as exp-004 — so the result supports the "architecture
consistency" claim. Do NOT switch to OHLCV unless the Numerai framing
fails; if OHLCV is used, the claim renames to "second setting" (F13).

**Sequence framing on real Numerai rows**: each row's 705 medium features
are reshaped to a T×D sequence (use T=15, D=47; 15×47=705, no padding),
consumed by the exp-002-style hand-rolled 1-layer functional GRU
(hidden ~64, dropout on the head, sequence-to-one scalar regression head).
Rationale (state it in results.md): the claim under test is
optimizer-effect consistency across ARCHITECTURES on the same data/task —
the sequence construction only needs to give a genuinely recurrent compute
graph over real Numerai features; it is not claimed to be the natural
architecture for the task.

**Arms** (same train shard, same 3 seeds {0,1,2}, identical data order per
seed across arms, 2000 steps @ B=1024 within-era batches — F6):
1. filter_off: GRU + plain AdamW (frozen config, see tuning below) × 3.
2. filter_on: GRU + SpectralConsensusFilter (hard, mp_factor 2.0 — F2,
   unchanged) wrapping the same AdamW config × 3.
3. C4 random-subspace control × 3 (same matching rule as exp-004: random
   orthonormal sample-subspace, k(t) and update-norm ratio(t) matched to
   the same-seed filter_on run) — include it; it was load-bearing in
   exp-004 and cost ~1/4 of filter_on. GAF arm: optional, only if the
   budget clearly allows; cut it first.

**GRU config tuning (F1: tuning-block eras ONLY)**: a small sweep (≤4
trials, seed 0, e.g. lr ∈ {3e-4, 1e-3} × dropout ∈ {0, 0.2}, wd=1e-3) on
the TUNING shard (`/ephemeral/t.buckworth/researcher-shards/
shard_tuning.parquet`, eras 0579–0966; if wiped, rebuild is NOT worth it —
fall back to reusing t07: lr 1e-3, wd 1e-3, dropout 0.2, and say so).
Freeze the best config BEFORE any verdict-shard evaluation; no config
changes afterwards. The spectral arm gets NO tuning sweep (same F12
affordability asymmetry as exp-004 — record it).

**Correctness guard (exp-002 caveat, mandatory)**: at job start on the
cluster (torch 2.5.1), re-run the B=64 vmap-vs-loop per-sample-gradient
assert for the GRU (seconds). If max abs diff > 1e-5, abort and report —
do not train on incorrect gradients.

**Evaluation**: predict once per (arm, seed) on the verdict shard
(`shard_verdict.parquet`, eras 0971–1225, 255 eras) after all configs
frozen; save per-era numerai_corr and spearman CSVs exactly like exp-004.
Diagnostics (k, ratio, cos_vs_mean, kept_energy_frac) logged every ~50
steps in filtered arms.

**Verdict machinery (identical to exp-004, frozen)**: paired per-era
diffs, circular moving-block bootstrap L=4, 10,000 resamples, 95% CI;
F3 threshold 0.00398 (nc) / 0.00391 (sp); three-way verdict rule; F12
downgrade rule applies to any "hurts" (spectral arm affords ~1 trial —
signature already on record); F10 corr-Sharpe via the same joint
machinery; F8 tail breakdown (both splits, with the regression-to-mean
caveat on the baseline-corr split). Adapt `analyze_verdict.py` (drop the
GAF columns if that arm is cut). Analysis runs locally on CPU.

## Compute & workflow (compute profile — binding)

MATS Slurm cluster, driven remotely over SSH (`ssh mats '<cmd>'`); you are
NOT on the cluster. FREE `compute` partition only, `--qos=debug` (this is
a validation-scale run well under 2h). Budget: ≤2 array tasks at ≤25
min/task (exp-003 projected 13.7 min/task; exp-002 measured eigh ≈3% of
step cost, vmap-GRU step ≈0.117 s projected on L40 → 2000 steps ≈ 4 min
filtered arm; a single job may well fit everything — prefer ONE job if
projected <25 min total, else a 2-task array split by arms).

Workflow per profile: (1) write code + `run.sbatch` locally under this
directory (`src/`, logs to `logs/slurm-%j.out`); all #SBATCH lines first:
`--partition=compute --qos=debug --gres=gpu:1 --cpus-per-task=8 --mem=32G
--time=00:50:00 --job-name=exp005-seq --output=logs/slurm-%j.out`;
(2) `rsync -avP` this dir to `mats:/mnt/nw/home/t.buckworth/researcher-runs/2026-07-31-spectral-optimizer-for-noise-reduction-on-financial-timeseri/exp-005/`;
(3) `ssh mats 'cd .../exp-005 && mkdir -p logs && sbatch run.sbatch'`
(mkdir logs BEFORE sbatch); (4) poll `ssh mats 'squeue -u t.buckworth'`
with sleeps (no busy-loop; wait out queue contention, do not double-submit);
(5) rsync results back; copy the slurm log to `run.log`.

Shards: reuse `/ephemeral/t.buckworth/researcher-shards/` (train, tuning,
verdict already there); job script should re-copy from
`../exp-001/data/shard_train.parquet` and `../exp-004/data/shard_verdict.parquet`
if ephemeral was wiped (tuning shard: skip sweep if gone, per above).
Inside the job: `source ~/venv/bin/activate`, `python -u`. NEVER
install/upgrade torch (2.5.1 pinned). `HF_HOME` not needed (no HF models).

Other constraints: no root installs; no venv creation; never modify files
outside the run directory; no local-GPU fallback (orchestration/analysis
only locally); clean up any large intermediates (no checkpoints; no file
>50 MB in this directory); target <30 min job runtime.

## PASS / FAIL

PASS: GRU trains within cap on within-era Numerai; correctness assert
passed on-cluster; paired filter_on vs filter_off comparison returned with
the same inference machinery (any verdict category OR an honest no-verdict
effect estimate counts — the deliverable is the architecture-consistency
comparison itself); diagnostics confirm the filter stayed engaged;
realized GRU baseline reported vs the MLP baseline and F9 band.
FAIL: within-era framing unworkable, correctness assert fails on 2.5.1, or
the arm exceeds the remaining budget. On FAIL: report per F7/W2 (cut +
limitation + resource-scoped future-work entry), never retry-loop.

## Deliverables

- `results.md`: architecture-consistency verdict (does the GRU arm
  reproduce the MLP's direction? same category or not), paired CIs, arm
  levels incl. zero-pred sanity, engagement diagnostics summary, F12
  wording where applicable, exact configs + seeds + timings, explicit F13
  claim-naming statement, comparison table vs exp-004's MLP numbers.
- `run.log` (full slurm stdout), per-era CSVs in `out/`, code + sbatch in
  this dir. No checkpoints; nothing >50 MB.
