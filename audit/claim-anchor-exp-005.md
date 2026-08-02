# Experiment 005 Results — Sequence-Arm Architecture-Consistency Test (GRU)

## Component Tested
Component #2 outcome extended to real data: does exp-004's MLP verdict on
the Spectral Optimizer (SpectralConsensusFilter on tuned AdamW vs tuned
AdamW alone) transfer to a recurrent architecture on the SAME dataset,
shards, seeds, and frozen inference machinery? (Conditional stretch goal;
both authorization conditions held: exp-002 PASS, exp-003 budget room.)

## Verdict: PASS

The deliverable — the architecture-consistency comparison itself — was
returned in full: the GRU arm trained within cap, the on-cluster
correctness assert passed, the paired filter_on vs filter_off comparison
came back through the identical frozen machinery, engagement diagnostics
confirm the filter did real work, and the realized GRU baseline is
reported against the MLP baseline and the F9 band.

**The GRU reproduces the MLP's verdict: same category, same direction,
similar magnitude.**

- GRU headline (filter_on − filter_off), 255 verdict eras, 3 seeds,
  moving-block bootstrap L=4, 10,000 resamples, 95% CI:
  **−0.00484, CI [−0.00758, −0.00211]** → pre-registered category
  **HURTS** (CI excludes 0, |mean| > F3 threshold 0.00398).
- MLP (exp-004): −0.00527, CI [−0.00886, −0.00181] → HURTS.
- **F12 downgrade applies identically** (pre-registered, binding):
  reportable claim = "no evidence of benefit under the affordable tuning
  budget", raw hurts numbers reported alongside.
- **C4 spectral-vs-random null replicates**: filter_on − c4_random =
  −0.00011, CI [−0.00233, +0.00215] → no detectable difference. The
  MP-eigenselection again adds nothing measurable beyond a random
  sample-subspace projection matched on k(t) and update-norm ratio(t).

**F13 claim-naming statement (binding)**: this experiment ran on the
**within-era Numerai framing** — the same train shard (eras 0425–0574) and
verdict shard (eras 0971–1225) as exp-004, each row's 705 medium features
reshaped to a T=15 × D=47 sequence (15×47=705, no padding). OHLCV was NOT
used. The supported claim is therefore **"architecture consistency"** (not
"second setting"). Rationale stated per plan: the claim under test is
optimizer-effect consistency across architectures on the same data/task;
the sequence construction only needs to give a genuinely recurrent compute
graph over real Numerai features — it is not claimed to be the natural
architecture for the task.

## Setup

- **Cluster job**: Slurm job **6614** on the MATS cluster, one NVIDIA L40,
  `--partition=compute --qos=debug --gres=gpu:1`, torch 2.5.1+cu121,
  total job time **6.6 min** (well under the 25-min cap; job waited ~1 min
  in queue on Resources). Full stdout in `run.log` (= `logs/slurm-6614.out`).
  Analysis run locally on CPU (`src/analyze_seq.py`; log
  `out/analyze_seq.log`; machine-readable `out/seq_analysis.json`).
  A local CPU smoke test (30 steps, B=256, seed 0, eval on train shard
  only) validated the code before submission (`out/smoke/`).
- **Architecture (the only change vs exp-004)**: hand-rolled 1-layer
  functional GRU from exp-002 (pure tensor ops, vmap-compatible), hidden
  64, dropout 0.2 on the final hidden state, sequence-to-one scalar head;
  21,761 params (vs MLP's 197k). `src/spectral_optimizer.py` verified
  byte-identical to the exp-002/source-repo copy.
- **Frozen inputs honored verbatim**: F1 blocks (hard-asserted against
  `ref/protocol.json`); F2 hard mode, mp_factor 2.0; F6 within-era
  batches, B=1024; 2000 steps; seeds {0,1,2} with identical per-seed data
  order across arms (same Sampler); F3 = 0.00398 (nc) / 0.00391 (sp);
  F5 L=4, 10,000 resamples; F12 rule pre-registered.
- **GRU config = t07 (lr 1e-3, wd 1e-3, dropout 0.2), frozen in code
  before submission — the plan's pre-authorized fallback.** The tuning
  sweep was skipped because `/ephemeral/.../shard_tuning.parquet` appeared
  wiped when checked from the login node. Honesty note: the job log later
  showed the shard DOES still exist on the worker node's `/ephemeral`
  (node-local disks differ between login and worker nodes). The t07
  fallback was already frozen and submitted on the wiped-shard premise;
  re-running a sweep after the verdict shard had been evaluated would
  violate the freeze-before-unblinding rule, so the t07 result stands and
  the untuned-GRU-config limitation is recorded (see Implications). The
  spectral arm got no tuning sweep either (same F12 affordability
  asymmetry as exp-004, recorded).
- **Arms**: filter_off (plain AdamW t07), filter_on (SpectralConsensusFilter
  hard/2.0), C4 random-subspace control (exp-004 matching rule verbatim:
  random orthonormal sample-subspace, k(t) and update-norm ratio(t)
  matched to same-seed filter_on). **GAF arm cut** (plan: optional, cut
  first).
- **Correctness guard (mandatory, exp-002 caveat)**: B=64 vmap-vs-loop
  per-sample-gradient assert re-run at job start on cluster torch 2.5.1,
  on real train rows, eval mode (dropout inactive so both paths are
  deterministic): **max abs diff 4.768e-07 ≤ 1e-5 → PASS** (4.6 s).
  Training proceeded on verified gradients.

## Results

### Raw Output (key lines; full log in run.log, analysis in out/analyze_seq.log)
```
GUARD max abs diff (all params): 4.768e-07 -> PASS (tol 1e-05)
zero-predictor: mean spearman +0.00057, mean numerai_corr +0.00063 over 255 eras

--- arm levels (mean per-era corr over verdict block) ---
  filter_off : nc +0.00586 (seeds ['+0.00623', '+0.00674', '+0.00461']), sp +0.00677, Sharpe 0.328
  filter_on  : nc +0.00102 (seeds ['+0.00298', '-0.00118', '+0.00127']), sp +0.00047, Sharpe 0.086
  c4_random  : nc +0.00113 (seeds ['-0.00202', '+0.00448', '+0.00093']), sp +0.00113, Sharpe 0.111

F9: GRU baseline nc +0.00586 vs band [0.0087, 0.0522] -> OUTSIDE; leakage gate fired: False

  filter_on - filter_off: nc -0.00484 CI [-0.00758, -0.00211] -> hurts; sp -0.00630 CI [-0.00915, -0.00342]
  c4_random - filter_off: nc -0.00473 CI [-0.00776, -0.00175] -> hurts; sp -0.00564 CI [-0.00854, -0.00270]
  filter_on - c4_random : nc -0.00011 CI [-0.00233, +0.00215] -> doesnt_help
  headline per-seed 0: -0.00325 CI [-0.00642, -0.00006]
  headline per-seed 1: -0.00792 CI [-0.01248, -0.00328]
  headline per-seed 2: -0.00335 CI [-0.00681, -0.00001]

=== GRU-ARM VERDICT: hurts -> F12 DOWNGRADE: no evidence of benefit under
    the affordable tuning budget ===
F10 corr-Sharpe: on 0.086 off 0.328 diff -0.242 CI [-0.414, -0.073] (excludes 0: True)

--- architecture consistency: GRU (this exp) vs MLP (exp-004) ---
  quantity                    MLP (exp-004)  GRU (exp-005)
  baseline nc (filter_off)         +0.00641       +0.00586
  filter_on nc                     +0.00113       +0.00102
  c4_random nc                     +0.00229       +0.00113
  headline diff (on-off)           -0.00527       -0.00484
  on-c4 diff                       -0.00116       -0.00011
  same category: True; same direction: True
```

### Metrics (plan PASS criteria)
| Metric | Target | Actual | Pass? |
|--------|--------|--------|-------|
| GRU trains within cap on within-era Numerai | job < 25 min | 6.6 min total (all 9 runs + eval) | Y |
| Correctness assert on-cluster (torch 2.5.1) | max abs diff ≤ 1e-5 | 4.768e-07 | Y |
| Paired on-vs-off comparison via frozen machinery | verdict category or honest effect estimate | HURTS: −0.00484, CI [−0.00758, −0.00211] (F12-downgraded wording) | Y |
| Diagnostics confirm filter engaged | selective, non-degenerate throughout | 100% of logged steps 0<k<B; k_med 60–63; kept_energy_med 0.946–0.985 (<0.995); ratio_med 0.80–0.85 | Y |
| Realized GRU baseline vs MLP baseline and F9 band | reported | +0.00586 vs MLP +0.00641; BELOW band [0.0087, 0.0522]; leakage gate not fired | Y |
| F13 claim naming | explicit statement | within-era Numerai → "architecture consistency" (OHLCV not used) | Y |

### Architecture-consistency comparison table (the deliverable)

| Quantity (nc, 255 verdict eras, 3 seeds) | MLP (exp-004, 197k params) | GRU (exp-005, 21.8k params) |
|---|---|---|
| filter_off baseline | +0.00641 | +0.00586 |
| filter_on | +0.00113 | +0.00102 |
| c4_random | +0.00229 | +0.00113 |
| zero-predictor sanity | +0.00063 | +0.00063 (same seeded predictor) |
| headline diff (on−off) | −0.00527 [−0.00886, −0.00181] | −0.00484 [−0.00758, −0.00211] |
| headline category | hurts → F12 downgrade | hurts → F12 downgrade |
| spearman variant | −0.00607 [−0.00927, −0.00292] | −0.00630 [−0.00915, −0.00342] |
| filter_on − c4_random | −0.00116 [−0.00366, +0.00134] (null) | −0.00011 [−0.00233, +0.00215] (null) |
| F10 corr-Sharpe diff (on−off) | −0.200 [−0.416, −0.004] | −0.242 [−0.414, −0.073] |
| per-seed headline diffs | −0.00473 / −0.00589 / −0.00521 | −0.00325 / −0.00792 / −0.00335 |
| filter engagement regime | k_med 1–4, ratio_med 9.5–13.9 (amplifies) | k_med 60–63, ratio_med 0.80–0.85 (mild attenuation/rescale) |

All three per-seed GRU headline diffs are negative with CIs excluding zero
(seeds 0 and 2 marginally). Direction is seed-consistent, as in exp-004.

### F8 tail-era breakdown (descriptive; iid bootstrap within buckets)

By baseline per-era corr quartile (on−off, nc): Q1 (worst) +0.01277
[+0.00982, +0.01575], Q2 −0.00058, Q3 −0.00944 [−0.01234, −0.00654],
Q4 (best) −0.02204 [−0.02485, −0.01921] — the same strong monotone pattern
as the MLP, with the recorded caveat that bucketing on the baseline's own
realized corr mechanically induces regression-to-the-mean; descriptive
only. The dispersion split (not outcome-conditioned): Q1 −0.00977
[−0.01348, −0.00573], Q2 −0.00140, Q3 −0.00398, Q4 −0.00415 — harm is NOT
concentrated in high-dispersion (tail) eras (if anything the largest point
harm is in the LOWEST-dispersion quartile), again no Feldman-style tail
pattern; and corr-Sharpe went down (−0.242), so no stability trade either.

### Engagement diagnostics

filter_on was selective on 100% of logged steps in all 3 seeds (0 < k < B),
k_median 60–63 of 1024, kept_energy_frac median 0.946–0.985 (below the
0.995 no-op guard), update-norm ratio median 0.80–0.85 (range up to ~1.6),
cos-vs-mean median 0.66–0.78. The pre-authorized FAIL condition (silent
disengagement) did not occur; the verdict is attributable to an engaged,
selective mechanism. C4 matched the same k/ratio trajectories exactly
(visible step-by-step in run.log).

### Timings
Per-run train times on one L40: filter_off 19.4–22.9 s, filter_on
59.2–67.2 s, c4_random 39.1–43.1 s; guard 4.6 s; total job 6.6 min
(exp-003's budget arithmetic projected 13.7 min/task — realized well
under; single job used instead of a 2-task array, as the plan preferred
for projected <25 min).

## Unexpected Observations

1. **The filter operates in a completely different regime on the GRU yet
   produces the same outcome.** On the MLP the filter collapsed to k≈1
   eigendirection and amplified the update ~10×; on the GRU it kept k≈60
   directions at roughly unit norm ratio (0.8–0.85, mild attenuation).
   Despite these opposite engagement signatures, the headline effect is
   nearly identical (−0.00484 vs −0.00527) and the spectral-vs-random null
   replicates even more tightly (−0.00011). This strengthens exp-004's
   mechanistic account: the harm tracks the generic
   subspace-projection/rescaling intervention class, not any particular
   spectral selection or amplification behavior.
2. **Login-node vs worker-node `/ephemeral` divergence**: the tuning-shard
   "wiped" determination was made from the login node, but the worker's
   node-local `/ephemeral` still had all three shards (visible in
   run.log). The t07 fallback was therefore invoked on a premise that was
   false on the execution node. The comparison remains protocol-valid
   (config frozen in code before any verdict evaluation; identical across
   all arms), but the GRU tuning sweep the plan preferred was skippable
   only under the wiped premise — recorded as a limitation below.
3. The untuned t07 GRU baseline (+0.00586) lands within ~9% of the tuned
   MLP baseline (+0.00641) despite an arbitrary 15×47 reshape of tabular
   features and 9× fewer parameters — the within-era sequence framing is
   a genuinely workable (if unnatural) architecture for this task.
4. F9: the GRU baseline sits BELOW the tuning-block sanity band
   [0.0087, 0.0522], exactly as the MLP baseline did (+0.00641) — the
   known verdict-block regime drift from exp-004, not a new anomaly; the
   leakage gate did not fire.
5. Seed spread is larger for the GRU filtered arms (filter_on s1 went
   negative, −0.00118; c4_random s0 negative, −0.00202) — plausible for a
   smaller model under update-restriction, but with 3 seeds this is an
   observation, not an estimate.

## Implications

What this tells us: the criterion was met — the architecture-consistency
deliverable is in hand, and it is affirmative. On the same data, shards,
seeds, and frozen inference machinery, a recurrent GRU reproduces the
MLP's result in category (hurts → F12-downgraded to "no evidence of
benefit under the affordable tuning budget"), direction (all 6 per-seed
diffs across both experiments negative), magnitude (−0.00484 vs −0.00527,
~83% of the baseline's edge erased), the corr-Sharpe deterioration, the
absence of a dispersion-tail pattern, and — most tellingly — the
spectral-vs-random-subspace null. The run's overall negative verdict is
not an MLP artifact; combined with exp-001's target-independence proof and
exp-004's controls, the study now supports an architecture-consistent
claim: per-sample gradient "consensus" filtering degrades OOS performance
on this low-SNR financial task, and its spectral machinery specifically
adds nothing beyond matched generic subspace projection, across two
architectures with very different filter-engagement regimes.

Limitations to carry forward (for Step 10/11): (a) the GRU config was
t07 transferred from the MLP, not swept — the plan-preferred ≤4-trial
tuning-block sweep was skipped on a wiped-shard premise that turned out
false on the worker node; a hurts-vs-undertuned-baseline reading is
already covered by the (conservative, one-sided) F12 downgrade, and the
resource ask to close it is small (one ~10-GPU-min job: 4-trial GRU sweep
on the tuning shard + re-run of this experiment with the swept config,
frozen before re-evaluation). (b) Sequence framing is a construct
(reshaped tabular features), stated per plan/F13 — the claim is
architecture consistency of the optimizer effect, not sequence-modeling
relevance. (c) 3 seeds; GAF arm cut (was cut-first by plan; its exp-004
result already covered the sign-agreement class).

Next steps: proceed to the run's write-up path — no kill criterion or
retry triggered; both experiments' verdicts and controls are mutually
consistent.
