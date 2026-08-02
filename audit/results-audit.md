# Results Audit — round 1

## Overall disposition: HONEST-NEGATIVE
## audit_exit_reason: true-null — adequately-powered honest negative; load-bearing result reproduced under fresh-seed re-execution on an independently constructed evaluation split; no claim-flipping defect found

## Re-execution summary

**Yes — the load-bearing experiment (exp-004) was actively re-executed on the
MATS cluster** (audit code and artifacts: `audit/rerun-exp-004/`; Slurm jobs
6616, 6618, 6620; ~9 min total L40 time), with:

- **Fresh seeds {10, 11, 12}** (producer used {0, 1, 2}), running the
  producer's own frozen `verdict_job.py` (seeds changed, dual-shard eval
  added; diff preserved in `audit/rerun-exp-004/src/rerun_job.py`).
- **A self-constructed evaluation split**: the verdict-era range 0971–1225 is
  the frozen pre-registration and was kept, but the audit built its own
  652,800-row shard directly from the raw `v5.0_validation.parquet` with an
  independent row subsample (RNG seed 424242 vs producer's 0), by audit code
  (`build_audit_shard.py`). The producer's shard was additionally
  cross-checked row-by-row against the raw parquet on 3 random eras (100% of
  rows found in raw; per-era counts consistent) — no fabrication, no
  out-of-block eras.

**Result: the headline reproduces in category, direction, and magnitude.**

| Quantity (mean per-era numerai_corr, 255 verdict eras, 3-seed mean) | Original (seeds 0–2) | Re-run (seeds 10–12, producer shard) | Re-run (seeds 10–12, audit-built shard) |
|---|---|---|---|
| filter_off (tuned AdamW) | +0.00641 | +0.00614 | +0.00495 |
| filter_on (spectral) | +0.00113 | +0.00068 | −0.00067 |
| headline diff (on−off), MBB L=4 95% CI | −0.00527 [−0.00886, −0.00181] | **−0.00546 [−0.00892, −0.00204]** | **−0.00562 [−0.00911, −0.00201]** |
| category vs frozen F3=0.00398 | hurts | **hurts** | **hurts** |
| filter_on − c4_random (C4 null) | −0.00116 [−0.00366, +0.00134] | +0.00054 [−0.00161, +0.00275] | −0.00114 [−0.00334, +0.00108] |

Pooled over all 6 seeds (0–2 + 10–12): −0.00537, 95% CI [−0.00853, −0.00222].
All CIs above are from the auditor's own independent moving-block-bootstrap
implementation (different code, different RNG), not the producer's.

**Re-execution surfaced one new defect** (Finding 5): the first re-run
attempt (job 6616, unmodified producer code) **crashed** with
`torch.linalg.eigh: failed to converge` on an ill-conditioned similarity
matrix at seed 10, and again in the `kept_energy_fraction` diagnostic at
seed 11 (job 6618). A mathematically-equivalent CPU-fp64 eigh fallback
(audit-side patch only; fired 4 times in 18,000 filtered steps of job 6620)
let the run complete. This is a robustness defect of the method's
implementation, not a validity defect of the reported results — the
producer's original jobs (6439, 6614) completed without hitting it, and the
patched re-run reproduces the original conclusions.

## Findings

### Finding 1 — exp-004 — Headline verdict (HURTS → F12 downgrade): claim ↔ evidence and re-execution
- **Verdict**: SUPPORTED
- **Severity**: — (no defect)
- **Evidence**: Every number in `results.md` traces to a primary artifact.
  Arm/per-seed means match `run.log` exactly (e.g. `filter_off_s0: mean
  spearman +0.00719, mean numerai_corr +0.00530 (train 2.8s)`; `filter_on_s0:
  ... +0.00057`). The bootstrap outputs match `out/analyze_verdict.log`
  (`filter_on - filter_off: nc -0.00527 CI [-0.00886, -0.00181] -> hurts`)
  and were re-derived by independent auditor code from
  `out/per_era_numerai_corr.csv`: −0.00527 [−0.00887, −0.00173]; robustness
  cross-checks agree (paired t-test p=1.5e-4, Wilcoxon p=1.8e-4). Re-executed
  with fresh seeds and an audit-built split: reproduces (see table above).
  F3=0.00398 re-derived from exp-003 artifacts: 0.25 × mean(+0.01962,
  +0.01286, +0.01533) = 0.003985 ✓. Freeze-before-unblinding corroborated by
  git history (frozen `verdict_job.py` + amended plan committed 2026-08-01
  17:36, before the verdict job ran; results pulled 2026-08-02). No grader
  gaming signatures anywhere in the code (the only `sys.exit` is exp-005's
  legitimate abort-on-bad-gradients guard, exit code 2 *before* results).
  F1 splits hard-asserted in code against `protocol.json`; embargo 4 eras =
  ceil(20d/5d) ✓; feature scaling uses fixed constants (no leakage via
  normalization); smoke runs evaluated on train eras only (blinding held).
- **Would fixing this plausibly flip PASS/FAIL?**: N/A — no defect. As
  negative-result triage: this is an adequately-powered genuine negative
  (F4 gate pre-registered power 0.91; realized CI half-width ≈0.0035 <
  2×F3), reproduced across 6 seeds and 2 independently constructed eval
  subsamples. Do not loop to re-confirm.
- **Finding to hand back**: none. The F12-downgraded wording ("no evidence of
  benefit under the affordable tuning budget", raw hurts numbers alongside)
  is exactly the pre-registered claim — no criteria drift.

### Finding 2 — exp-004/exp-005 — C4 mechanism claim ("MP-eigenselection indistinguishable from random subspace")
- **Verdict**: SUPPORTED (with a wording bound for the write-up)
- **Severity**: Low
- **Evidence**: Re-derived from CSVs: MLP −0.00116 [−0.00366, +0.00134]; GRU
  −0.00011 [−0.00233, +0.00215] (audit's own bootstrap, matching
  `analyze_verdict.log` / `analyze_seq.log` lines). Replicated in the audit
  re-run at fresh seeds on both shards (+0.00054 [−0.00161, +0.00275];
  −0.00114 [−0.00334, +0.00108]). The C4 control implementation is sound:
  random orthonormal sample-subspace with the same-seed filter_on k(t) and
  update-norm-ratio(t) trajectories matched step-by-step (full 2000-step
  trajectories verified present in `traj_filter_on_s*.npz`).
- **Would fixing this plausibly flip PASS/FAIL?**: No.
- **Finding to hand back**: reporting precision — "indistinguishable" is an
  equivalence claim *at the F3 resolution* (CIs lie within ±0.004);
  spectral-vs-random differences smaller than ~0.002–0.003 cannot be
  excluded. The paper should state the claim with its CI, not as identity.
  Likewise "GAF-style" should stay flagged as a sign-agreement variant, not
  Chaubard et al.'s exact microbatch-cosine GAF.

### Finding 3 — exp-001 — Target-independence proof (co-primary deliverable)
- **Verdict**: SUPPORTED
- **Severity**: — (no defect)
- **Evidence**: The producer's fp64 check (`target_independence_check.txt`:
  max |eig(real) − eig(permuted)| = 8.88e-16) re-verified by the auditor with
  an **independent construction** (different seed/architecture/width/batch,
  `audit/rerun-exp-004/verify_proof_independent.py`): normalized-spectrum
  diff 2.1e-14, plus two scoping controls the producer's script lacked —
  the UNNORMALIZED spectrum is strongly target-dependent (diff ≈ 61) and a
  2-output MSE head breaks the invariance (diff ≈ 0.61) — confirming the
  theorem is non-vacuous and correctly scoped to scalar-output MSE with row
  normalization, which is exactly the deployed code path
  (`spectral_optimizer.py` lines 74–79; verified byte-identical to
  `/home/titus/pyg/optimizers/experiments/spectral_optimizer.py`, F11 ✓).
  The step-0 cluster spectra claim re-checked from `spectra.npz` (real vs
  permuted max diff 2.6e-4 on top1 39.4 — fp32 roundoff ✓). C2/C5/C1 tables
  in results.md match `out/analysis_stdout.txt` line-for-line.
- **Would fixing this plausibly flip PASS/FAIL?**: N/A.
- **Finding to hand back**: none. One orchestration note, examined for
  criteria drift and cleared: exp-001's headline logged FAIL on the C2
  conjunct while its pre-registered *fail-fast* clause (degeneracy) did not
  fire; the orchestrator proceeded to exp-004 under an amended interpretation
  frame. This was recorded before unblinding, left all verdict machinery
  unchanged, *narrowed* the permissible interpretation (target-blind
  subspace regularization; C4 made load-bearing), and the outcome was
  negative — so the deviation was conservative, transparent, and did not
  move goalposts toward a favorable result. The paper must keep exp-001's
  headline as a logged FAIL-on-C2 (mechanistic finding), not relabel it PASS.

### Finding 4 — exp-005 — GRU architecture-consistency replication
- **Verdict**: SUPPORTED
- **Severity**: — (no defect)
- **Evidence**: All results.md numbers trace to `run.log` (job 6614:
  `filter_off_s0 ... +0.00623`, `filter_on_s1 ... -0.00118`, `GUARD max abs
  diff ... 4.768e-07 -> PASS`) and re-derive from the per-era CSVs with
  audit code: headline −0.00484 [−0.00757, −0.00205], C4 null −0.00011,
  Sharpe diff −0.242 [−0.414, −0.073] (robust). F13 satisfied (within-era
  Numerai framing; "architecture consistency" naming is per the binding
  amendment). The t07-fallback config was frozen in code pre-submission and
  identical across arms — protocol-valid; the false "wiped shard" premise is
  honestly disclosed in results.md. Not independently re-executed (the
  load-bearing experiment is exp-004); log-and-re-derivation evidence only,
  which is sufficient for a replication arm whose parent claim was re-executed.
- **Would fixing this plausibly flip PASS/FAIL?**: No — re-tuning the GRU
  baseline would, if anything, strengthen the hurts direction (a better
  baseline widens the gap); the conservative F12 wording already covers the
  under-tuned-spectral-arm direction.
- **Finding to hand back**: carry the untuned-GRU-config limitation (already
  recorded) into Limitations + Future Work with its ~10-GPU-min resource ask.

### Finding 5 — exp-004 (method implementation) — Seed-dependent `linalg.eigh` non-convergence crash (found by re-execution)
- **Verdict**: FIXABLE-DEFECT (robustness/reporting only — does not affect any reported number)
- **Severity**: Medium
- **Evidence**: Audit job 6616 (producer code verbatim, fresh seed 10):
  `torch._C._LinAlgError: linalg.eigh: The algorithm failed to converge
  because the input matrix is ill-conditioned or has too many repeated
  eigenvalues (error code: 1)` at `spectral_optimizer.py` line 79, ~step
  1400; job 6618 hit the same in the `kept_energy_fraction` diagnostic at
  seed 11. Root cause is the run's own documented rank collapse (one
  eigenvalue carrying ~80–90% of mass ⇒ near-degenerate trailing spectrum on
  cuSOLVER fp32). With a mathematically-equivalent CPU-fp64 fallback the
  full re-run completed (fallback fired 4× in 18,000 filtered steps) and
  reproduced all conclusions.
- **Would fixing this plausibly flip PASS/FAIL?**: No — the patched re-run
  reproduces the identical verdict; the defect is availability (crash), not
  bias.
- **Finding to hand back** (defect class): "numerical-robustness: the
  spectral filter's eigendecomposition can fail stochastically by seed in
  fp32 on GPU under the rank-collapse regime this very study documents; the
  original results are 6-of-6-seed lucky-complete. Disclose in the paper's
  implementation notes; any follow-up must handle eigh non-convergence."

### Finding 6 — exp-004 — MLP corr-Sharpe secondary claim is bootstrap-RNG fragile
- **Verdict**: FIXABLE-DEFECT (write-up wording; auditor guidance suffices)
- **Severity**: Low
- **Evidence**: Producer: diff −0.200 CI [−0.416, −0.004] "excludes 0: True"
  (`analyze_verdict.log`). Auditor's independent bootstrap on the same data:
  [−0.416, **+0.003**] — the upper bound straddles zero across bootstrap
  RNGs. The GRU Sharpe claim is robust (−0.242 [−0.414, −0.073], reproduced).
  Per-seed headline note (already disclosed by producer): seed-2 CI touches
  zero ([−0.01069, +0.00023]); direction is 6/6-seed consistent including
  the audit seeds.
- **Would fixing this plausibly flip PASS/FAIL?**: No — F10 corr-Sharpe is
  secondary; the headline category never depended on it.
- **Finding to hand back**: report the MLP corr-Sharpe CI as *marginal*
  (boundary within bootstrap noise), not as a clean exclusion; the GRU
  Sharpe result can carry the stability-worsens statement.

### Finding 7 — exp-004 — Verdict-block baseline below the pre-registered sanity band (regime drift)
- **Verdict**: TRUE-NULL (context limitation, honestly reported — not a re-run trigger)
- **Severity**: Medium (for external validity of the claim, not its correctness)
- **Evidence**: `analyze_verdict.log`: `F9: baseline nc +0.00641 vs band
  [0.0087, 0.0522] -> OUTSIDE; leakage gate fired: False`. The frozen anchor
  (success-criteria.md) conditions an "informative outcome" on working-arm
  corr landing ≈0.01–0.04; the realized verdict-block baseline (+0.0064,
  ~2.3× the F9-recalibrated band floor short) is below it, making the frozen
  F3 threshold ≈62% of the realized baseline rather than the intended 25%.
  The baseline still carries real signal (per-era t ≈ 4.9 vs zero-predictor
  +0.0006), the CI excludes zero regardless of F3, and the audit re-run
  confirms the level (+0.0061). The producer disclosed this exactly rather
  than re-fitting the threshold — the correct handling.
- **Would fixing this plausibly flip PASS/FAIL?**: No. A stronger baseline
  (full-scale data/model) could change the *magnitude/relevance* of the
  effect, not rescue the filter at this operating point — and that is
  priced as future work W1, beyond the compute profile.
- **Finding to hand back**: the paper must scope the claim to this regime:
  a low-capacity subsampled-Numerai pipeline whose verdict-block edge sits
  below the practitioner sanity band; the hurts magnitude (~82% of baseline
  edge) is relative to a small edge.

### Finding 8 — run-level — Remaining carried limitations verified as honestly recorded
- **Verdict**: SUPPORTED (disclosures match artifacts)
- **Severity**: Low
- **Evidence**: C5 probe inconclusive (all Wilcoxon p ≥ 0.078, n=8,
  `analysis_stdout.txt` lines 20–27 — matches "inconclusive, carried");
  F8 baseline-quartile split flagged descriptive/regression-to-mean in the
  analysis code itself (`analyze_verdict.py` f8_note) before results;
  F12 under-exploration signatures recorded in exp-003 before unblinding
  (`analyze.log` line 17, sweep table); single spectral operating point
  (~1 matched-compute trial vs AdamW's 12) is arithmetic from measured
  times (63.1s vs 3.65s/run), correctly derived.
- **Would fixing this plausibly flip PASS/FAIL?**: No.
- **Finding to hand back**: none beyond the limitation-triage rows below.

## Summary table

| Finding | Experiment | Verdict | Severity | Flips verdict if fixed? |
|---------|-----------|---------|----------|-------------------------|
| 1 | exp-004 | SUPPORTED (re-executed, reproduces) | — | N/A |
| 2 | exp-004/005 | SUPPORTED (wording bound) | Low | No |
| 3 | exp-001 | SUPPORTED (independently re-verified) | — | N/A |
| 4 | exp-005 | SUPPORTED | — | No |
| 5 | exp-004 impl. | FIXABLE-DEFECT (robustness disclosure) | Medium | No |
| 6 | exp-004 | FIXABLE-DEFECT (Sharpe wording) | Low | No |
| 7 | exp-004 | TRUE-NULL (regime-drift scope limit) | Medium | No |
| 8 | run-level | SUPPORTED | Low | No |

No finding warrants a remediation re-run: the two FIXABLE-DEFECTs are
write-up-level (disclosure/wording) and neither could flip the verdict.

## Unresolved findings for the write-up

These MUST appear in the paper's Limitations and the email:

1. **Regime drift / below-band baseline** (Finding 7): the verdict-block
   baseline (+0.0064) sits below the practitioner sanity band; the F3
   relative bar realized at ~62% instead of the intended 25%. Claim is
   scoped to this low-edge regime.
2. **Affordability asymmetry** (F12, pre-registered): spectral arm had ~1
   matched-compute configuration vs AdamW's 12 trials; the reportable claim
   is "no evidence of benefit under the affordable tuning budget", not
   "hurts" simpliciter — keep the downgraded wording everywhere.
3. **GRU config not tuned** (t07 fallback, frozen pre-unblinding; premise
   later found false on the worker node) — disclosed; ~10-GPU-min fix ask.
4. **C5 era-identity probe inconclusive** (n=8 probes, no significant gap in
   either direction).
5. **eigh non-convergence fragility** (Finding 5, new from this audit):
   seed-dependent crash risk in the released implementation.
6. **MLP corr-Sharpe CI is marginal** (Finding 6); rely on the GRU arm for
   the stability statement.
7. **C4 "indistinguishable" = equivalence at ±0.004 resolution**, not
   identity (Finding 2).
8. **F8 baseline-quartile gradient is descriptive only**
   (regression-to-the-mean); the unconditioned dispersion split is the
   evidentially meaningful one (no Feldman-tail pattern).

## Limitation triage

| Limitation | Disposition | If fixable now: what + cost | If future-work: resources a fix would need |
|-----------|-------------|------------------------------|--------------------------------------------|
| 3 seeds per arm (headline) | fix-now-free | Already fixed by this audit: cite the fresh-seed re-execution — 6-seed pooled headline −0.00537 [−0.00853, −0.00222] and audit-shard replication (`audit/rerun-exp-004/`) in the paper's robustness paragraph | — |
| F8 quartile split regression-to-mean | fix-now-free | Re-analysis of existing CSVs: bucket eras by *held-out-seed* baseline corr (e.g. bucket on seed 0, compute diff on seeds 1–2, rotate); descriptive claim only, no verdict change | — |
| F3 calibration gap (~62% relative bar) | fix-now-free | Reporting fix: state both the frozen absolute threshold and the realized relative bar; note the CI-excludes-zero verdict is threshold-independent | Structural fix (threshold defined on realized verdict-block baseline) requires a new pre-registration on fresh eras — future protocol |
| MLP corr-Sharpe marginal CI | fix-now-free | Wording fix per Finding 6; GRU Sharpe carries the claim | — |
| C4 wording ("indistinguishable") | fix-now-free | State as equivalence CI within ±F3 | Tighter bound: more seeds (~6 × 80 s GPU per extra seed-pair) if a follow-up wants <0.002 resolution |
| eigh non-convergence (Finding 5) | fix-now-free (disclosure) | Implementation note in paper; the audit's CPU-fp64 fallback patch is in `audit/rerun-exp-004/src/` for any follow-up | — |
| GRU baseline untuned (t07 fallback) | fix-now-cheap | One ~10-GPU-min job: ≤4-trial GRU sweep on tuning-block shard, freeze, re-run exp-005 arms. Not claim-critical (would likely strengthen the negative); only worth spending if a remediation round is opened for other reasons — otherwise carry as future work | — |
| No FNC (feature-neutral corr) reported | fix-now-cheap | Re-run eval saving per-row predictions + local neutralization (~6 min GPU + CPU); was a solid-tier (TMLR) item, not minimum-viable | — |
| C5 probe power | fix-now-cheap | ~5-GPU-min higher-power probe (more probes/steps); largely superseded by the target-independence proof — optional | — |
| Single spectral operating point (mp_factor 2.0/hard/B=1024) | future-work | — (a wider sweep inside this run would break the pre-registered matched-compute design) | ~30 GPU-min for a 12-trial spectral sweep at a *declared* enlarged budget for both arms (re-registration needed); fits an L40 |
| Below-band baseline / subsampled low-capacity pipeline (B-conditionality, A5/W1) | future-work | — | Full-scale non-subsampled Numerai verdict at B≈4096: est. 10–20 L40-hours or A100-class (exceeds 30-min cap and free-partition etiquette) |
| Sequence arm is reshaped-tabular construct (F13) | future-work | — | True second setting: OHLCV dataset with both MLP and GRU arms (W2): 2–4 experiment slots' GPU, ~1–2 h L40 total |
| No Muon / GBT context baselines (designated cuts, W3) | future-work | — | ~2 experiment slots (~30–60 min L40); Muon tuning sweep at matched budget |
| Single dataset (Numerai v5), single loss (MSE) | future-work | — | Second financial dataset + corr-aligned loss ablation; ~2–3 slots. Note the target-independence theorem is loss-specific: multi-output/CE settings need new engagement diagnostics (exp-001 future-work list) |
| MP-null under cross-sectional correlation (W4) | future-work | — | CPU-only simulation study (no GPU); days of CPU, no cluster constraint |

### Audit artifacts

- `audit/rerun-exp-004/rederive_local.py` — independent re-derivation of all
  headline/C4/Sharpe/per-seed numbers from primary CSVs.
- `audit/rerun-exp-004/verify_proof_independent.py` — independent proof
  check with scoping controls.
- `audit/rerun-exp-004/build_audit_shard.py`, `data/shard_verdict_audit.parquet`
  — self-constructed eval split + producer-shard-vs-raw cross-check.
- `audit/rerun-exp-004/src/`, `run.sbatch`, `logs/slurm-{6616,6618,6620}.out`,
  `out/` — the re-execution (crash evidence + completed reproduction).
- `audit/rerun-exp-004/analyze_rerun.py` — re-execution analysis
  (fresh-seed and audit-shard headlines, 6-seed pool).
