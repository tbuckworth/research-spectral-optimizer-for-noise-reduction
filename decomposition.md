# Steinhardt Decomposition

**Run**: 2026-08-03-followup-follow-up-spectral-optimizer-redo-pxp-filter-walk-
**Status**: ROUND 2 — regenerated after the Step 6 challenge loop-back (verdict MAJOR_REVISIONS). Replaces the round-1 decomposition, which is recorded as superseded in `state.md`. Everything the challenge did not fault is preserved from round 1; the B-series amendments (B1–B4, step 6 decision) and the decomposition-facing consequences of the round-2 criteria (A1–A8) are integrated below and indexed in the change log.
**Question**: Does the p×p `SpectralGradientFilter` (`~/pyg/optimizers/spectral_filter.py`, class at line 52, `filter_grad()` at line 303) improve out-of-sample performance vs tuned AdamW on Numerai v5, under a 3-fold expanding-window walk-forward protocol with refit, a hard 0.60×-example-model baseline gate (A5 semantics: floored denominator, near-miss bands, fix ladder), and a genuine rank sweep at matched 12-trial budget per the A4 staged allocation?
**Compute profile**: MATS Slurm free `compute` partition, driven remotely; 1× L40 per job; max 5 submitted experiment units; short trials batched inside jobs. Local machine = orchestration + CPU sims/smoke tests + analysis only (the B3 arm-C simulation is explicitly a local CPU task).

## Round-2 Change Log (B-series, visibly implemented)

| Amendment | Where it lands |
|---|---|
| **B1** — EU-1 expanded: one FULL-LENGTH arm-A convergence run (real steps-to-convergence + proxy gate ratio on a VALID slice); arm-B timing at the largest plausible gate-fix architecture; timing of eigh-every-N-steps and GPU-fp32-eigh variants | Component #8 (quick test + pass criteria), EU-1 row of the packing plan, component #4 (proxy gate ratio as early warning), component #1 (packing arithmetic anchored on measured steps-to-convergence, not the ~500×-extrapolated 3 s parent anchor) |
| **B2** — resumability BEFORE any submission: per-trial NFS persistence (kill-test the parent exp-003 harness's append), refit checkpoints, `--time` at 2–3× projection, folds 2–3 re-projected from fold-1 REALIZED wall, >~16 h folds split into sweep-job + refit/eval-job | New component #7; binding requirements on EU-2/3/4 in the packing plan |
| **B3** — arm C redesigned for parameter space (A7): invariants = k(t) + update-norm ratio + basis-rotation rate; preferred design = random orthogonal rotation of B's own realized basis; ~20-min LOCAL CPU sim REQUIRED before porting code; principal-angle rotation-rate diagnostic logged in all arm-B runs | Component #2 (rewritten; the CPU sim is its quick test); rotation-rate logging added to components #5 and #8 |
| **B4** — power-sim consequence wired: if P(CI excludes 0 \| true effect ±0.005) < 0.6 on ≥ 2 prospective folds, the A1 power-qualified kill wording is load-bearing; it goes into frozen `protocol.json` regardless (the sim sets write-up prominence) | Component #3 (rewritten fail semantics); protocol-freeze checklist |

Round-1 elements preserved unchanged and still binding: gate-first fold-1 structure (fold 1 runs alone; gate evaluated before folds 2–3 are spent); EU packing EU-1 pre-flight / EU-2 fold 1 / EU-3+4 folds 2–3 parallel / EU-5 reserve; cut order on overflow (cap rank grid → reduce seeds, floor 3 → reduce folds, all reported); rank-grid capping is a documented design realization, not a failure; a structural alpha=0 identity failure is STOP-and-report; no rescue components exist downstream of a clean NULL/HURTS; data is banked (SKIP, symlink, never re-download).

**Decay/warmup decision (A8 authorization, exercised here)**: the criteria permit Step 5 to trade one pruned grid dimension for `decay ∈ {0.99, 0.999}`. Decision: **exercised, as a stage-2 probe** — one of the stage-2 refinement trials runs the stage-1 winner's configuration at `decay=0.999` (0.99 is the repo default already in every trial); if it wins on VALID it carries into refit. `energy_threshold` stays pruned to {0.90} and `adaptive="gap"` stays dropped, per A4. `warmup` remains at the repo default (100) and **stays a declared limitation** — the probe partially lifts the covariance-memory-horizon limitation for decay only, and the write-up says exactly that.

## Project Components

This is a follow-up run: the parent's PASS infrastructure carries over and is deliberately **not** re-tested as high-lambda components. Already banked: Numerai v5 data on local disk (~6.3 GB in parent `data/` — symlink, never re-download), era-purge/embargo arithmetic (parent `exp-001/src/data_prep.py`), shard building + 12-trial AdamW sweep harness (`exp-003/src`), per-era eval + moving-block bootstrap + subspace-matching logic (`exp-004/src`), independent bootstrap re-derivation + CPU-fp64 eigh-fallback patch (`audit/rerun-exp-004/src`). Parent `src/` comes from the parent run dir (`/media/titus/big/researcher-output/2026-07-31-spectral-optimizer-for-noise-reduction-on-financial-timeseri/`) or a clone of the parent repo — `prior/experiments/` here holds only `results.md`/`plan.md`. Banked ≠ untouched: the exp-003 harness is banked as *code*, but its crash-persistence behaviour is exactly what B2 requires verified, so that verification is a component (#7), not an assumption.

What is genuinely NEW and uncertain decomposes into five layers (round 1's four, plus the logistics layer the pre-mortem showed was unpriced):

1. **New object** — `SpectralGradientFilter` has never been run on Numerai data (#5: integration + alpha=0 identity + the A8 planted-subspace check the identity does not cover), and its mechanism control must be **re-derived**, not ported, for parameter space (#2 — the round-1 "mechanical port" framing was the challenge's top critical defect).
2. **New scale** — full v5.0 feature set (~2.3k features vs the parent's 705), ALL usable eras per fold, re-tuned step counts, and a rank grid reaching 2048 where the per-step (k+1)×(k+1) CPU eigh is a cubic-cost hazard (#8 throughput/VRAM/realized grid, #6 long-schedule numerics, #1 packing).
3. **New protocol** — 3-fold expanding-window walk-forward with per-fold refit, THE assertion `min(test_eras) − max(refit_train_eras) == 5`, per-fold gate denominators computed before freeze, and the example-preds coverage assertion (#9); per-fold inference power at ~110 TEST eras under the A6 hierarchical bootstrap, with the MDE machinery that feeds the A1-qualified kill criterion (#3).
4. **The scientific gate** — the 0.60×-example-model baseline gate has never been attempted; the claim that the parent's 27% shortfall was protocol-induced (state.md assumption 3) is untested (#4). Still the run's dominant genuine uncertainty (~half the pre-mortem's failure mass). B1 buys this bit one experiment unit earlier via the EU-1 proxy gate ratio.
5. **Logistics as a first-class risk** — pre-mortem scenario 4: fold jobs are multi-hour batched submissions on a shared partition, sized off a cost model that was extrapolated ~500× in round 1. B1 replaces the extrapolation with a measured convergence run; B2 makes fold jobs resumable so a Slurm kill costs a trial, not a fold (#7).

The asymmetry inherited from the criteria stands: the WINS/HURTS/NULL *content* of the verdict is not a component — all categories, including the pre-registered qualified forms (power-limited NULL, scope-limited, budget-limited, baseline-construction negative), are informative outcomes. P_success below measures only whether an *interpretable* outcome is reachable. The parent's HURTS is scoped to the B×B variant (its target-independence theorem depends on row normalization) and is **not** used as evidence about this optimizer anywhere in this table. Binding scope exclusions honored: no GRU arm, no live submission, MLP-only, no rescue downstream of a clean kill.

## Lambda Table

Ordered by descending lambda. T = wall-clock hours for the minimum viable pass/fail test (including rsync/queue overhead for cluster tests), not full implementation time.

| # | Component | P_success | Evidence | T (hrs) | lambda | Quick Test | Dependencies | Status |
|---|-----------|-----------|----------|---------|--------|------------|--------------|--------|
| 1 | Design packs into ≤5 experiment units (incl. 5-seed option, staged sweep, resumable structure) | 0.70 | B1 convergence run replaces the ~500× cost extrapolation (up); but packing now also carries the 5-seed target, arm C rotation machinery, and split-job contingency (down); packing has not been trivial in any run so far | 0.25 | 1.43 | Arithmetic from #8's measured s/step AND measured steps-to-convergence: pre-flight + 3 resumable fold jobs + reserve ≤ 5 units, each within `--time` at 2–3× projection | #8 | PENDING |
| 2 | Arm C re-derived for parameter space (invariants: k(t), norm ratio, basis-rotation rate) | 0.65 | Parent matching logic proven twice but in B×B sample space (k/B ≈ 3–50% energy); at p≈600k a random k-subspace captures ~k/p ≈ 0.3% (challenge critical #1); redesign specified (A7) but never implemented in any space | 0.75 | 0.57 | **B3 REQUIRED pre-port CPU sim (~20 min runtime, local)**: captured-energy fraction + realized norm-amplification at k ∈ {8, 512, 2048}, p ≈ 600k, for the candidate designs; norm-matched C must still descend on the #5 synthetic task | #5 (for replay/logging format; the energy sim itself has no dependency) | PENDING |
| 3 | Per-fold inference power at ~110 TEST eras + MDE machinery (A6 hierarchical bootstrap) | 0.70 | Parent sim: half-width 0.00219 at 255 eras/3 seeds → ~0.0033 scaled to 110 eras; but recent-window lag-1 ACF may approach the 0.763 the parent measured for the example model (longer blocks, smaller effective n), and hierarchical seed resampling widens CIs; 5 seeds narrows | 0.75 | 0.48 | Local CPU: parent per-era vectors restricted to prospective ~110-era fold windows (from #9 boundaries); hierarchical era-block × seed bootstrap; null and ±0.005-injected scenarios at 3 and 5 seeds; per-fold MDE computed | #9 | PENDING |
| 4 | Baseline gate: tuned AdamW ≥ 0.60× floored example-model denominator (≥2 of 3 folds) | 0.55 | Untested (state.md assumption 3); parent hit 27% under broken protocol; parent's best tuned MLP +0.0196 on older/easier eras; 0.60× ≈ 0.014 absolute sits inside the practitioner NN range but on the hardest regime; A5 bands/ladder make the reserve spend efficient without changing the underlying uncertainty; fold gates correlated | 2.5 | 0.24 | Fold-1 job phase 1 (gate-first): 12-trial A sweep on VALID → refit → single TEST eval → realized ratio + THE assertion printed. Early warning one unit sooner: EU-1's B1 proxy gate ratio on a VALID slice | #8, #9, #1 | PENDING |
| 5 | SpectralGradientFilter integration + alpha=0 identity + planted-subspace check + eigh fallback + A8 diagnostics | 0.80 | Code exists and ran in home repo (H3/H5/H7); never on Numerai/regression at this p; identity is by-construction but covers only the no-op path — hence the added planted-subspace check (A8); fallback patch exists (parent audit Finding 5) | 1.0 | 0.22 | Local CPU: synthetic regression; loss falls; alpha=0 soft identity == plain AdamW in fp64; hard top-k recovers a planted dominant direction; forced-LinAlgError fallback fires and recovers; cosine/kept-norm/rotation-rate logging emitted | None | PENDING |
| 6 | Numerical stability over long re-tuned schedules | 0.80 | Rank-1 SVD + small eigh well-conditioned in home repo at short horizons; parent audit Finding 5 (fp32 eigh fails stochastically under rank collapse) mitigated by mandated CPU-fp64 fallback with logged count | 1.0 | 0.22 | Short-schedule check inside #8 (500-step runs at grid extremes); fully retired by fold-1 full schedules in #4 | #8 | PENDING |
| 7 | Fold-job resumability + realized-wall projection (B2) | 0.85 | Parent exp-003 harness exists and "appends results" per its design, but crash-survival of the append is unverified; refit checkpointing is standard torch engineering; the risk is discipline, not feasibility | 0.75 | 0.22 | Local kill-test: run the sweep harness on synthetic trials, SIGKILL mid-trial, verify completed trials persisted and a resume skips them; refit checkpoint save/restore roundtrip | None | PENDING |
| 8 | Full-scale throughput/VRAM + realized rank grid + B1 additions (convergence run, second architecture, eigh variants) | 0.70 | New scale: ~2.3k features, ~600k-param MLP, 4–6M-row shards; p×k basis ≈ 5 GB fp32 at k=2048 (fits 48 GB); per-step 2049×2049 CPU eigh may blow the ~2×-Adam claim; nothing at this p/rows/rank has ever run | 2.0 | 0.18 | One `--qos=debug` job (EU-1): largest-shard build; s/step for A and B at rank {8,32,128,512,2048} at BOTH the planned and the largest plausible gate-fix architecture; eigh-every-N + GPU-fp32-eigh timings; VRAM peaks; 500-step stability + fallback count; on-cluster alpha=0 re-assert; **one full-length arm-A convergence run with VALID-slice proxy gate ratio**; adaptive-config realized k(t) + rotation-rate on short arm-B runs | #5, #9 | PENDING |
| 9 | Walk-forward fold arithmetic + protocol.json + THE assertion + per-fold gate denominators + coverage assertion | 0.85 | Parent era arithmetic PASSed (exp-001); expanding-window layout is new deterministic code; 643+ usable eras comfortably cover 3×(110+96+8); denominators are a 10-minute pandas job on parent `out/example_per_era_corr.csv` | 1.0 | 0.16 | Local CPU: boundaries from realized usable-era list; per fold assert `min(test)−max(refit_train)==5`, contiguity, non-overlap, coverage ≥0.95, every TEST era present in example preds; compute per-fold example-model denominators with the 0.010 floor; write `protocol.json` incl. raw-era↔usable-index mapping | None | PENDING |
| 10 | Numerai v5 data + example preds (banked by parent) | 0.97 | Downloaded and used end-to-end by parent (exp-001 PASS); files verified on local disk | 0.1 | 0.30* | Symlink from parent `data/`; row/era-count sanity only | None | **SKIP** |

*#10 marked SKIP per the rubric (successfully replicated in the parent run); its lambda is not used for ordering and it appears only so the reuse decision is explicit.

## Component Details

### Component 1: Design packs into ≤5 experiment units [lambda = 1.43]

**What**: Per fold: 12-trial arm-A VALID sweep; 12-trial arm-B VALID sweep per the **A4 staged allocation** (stage 1: one trial per point of the realized rank grid {8, 32, 128, 512, 2048 as feasible} + `adaptive="effrank"` at transferred LR, 7–8 trials; stage 2: 4–5 trials refining LR/alpha around the winner, **one of which is the decay=0.999 probe** per the decision above); from-scratch refits at selected hps (paired seeds × arms A/B/C — **seed target 5, floor 3, per A6**: the packing arithmetic explicitly evaluates the 5-seed option, the only authorized upward power lever); one TEST evaluation pass. All of it must pack into ≤ 5 submitted units with pre-flight and reserve included, under the B2 resumable-job structure.
**Risks to this component**: Arm-B trial cost at high rank inflating fold jobs; converged step count much larger than projected; the 5-seed target and arm C's rotation machinery adding load; queue variance; a >16 h fold forcing a B2 split that consumes unit count.
**Evidence for P_success = 0.70**: Round 1's estimate is retained with offsetting round-2 changes: the B1 full-length convergence run removes the largest unknown (the ~500×-extrapolated step count — the pre-mortem's cross-cutting theme 3), which raises P; the added content (5 seeds, arm C rotation, staged sweep, split contingency) lowers it. Packing has failed to be trivial in every run so far. Rubric band 0.7–0.9, low end.
**Quick test**: Pure arithmetic immediately after #8 returns measured s/step AND measured steps-to-convergence: (12 + 12 trials) × trial time + (seeds × 3 arms) × refit time + eval, per fold; check each fold job fits one 1-GPU submission with `--time` set at 2–3× the projection (B2), and pre-flight + 3 folds + reserve ≤ 5 units. Evaluate at seeds = 5 first; drop to 3 only if 5 does not fit, and record which.
**Pass criterion**: 1 pre-flight + 3 fold jobs + 1 reserve = 5 units; each fold job ≤ ~8 h projected wall at `--time` ≤ ~16 h (2× margin); seeds = 5 if it fits, else 3 documented.
**Fail criterion**: A fold cannot fit one unit even after capping the rank grid at the largest measured-feasible point and cutting to 3 seeds.
**If the quick test returns FAIL**: Execute the preserved cut order — cap rank grid (documented design realization; triggers the A2 kill-scope wording if K_max < 512) → reduce seeds (floor 3) → reduce folds (decision rule then reads over gated folds) — all reported. A fold projecting > ~16 h is split into sweep-job + refit/eval-job (B2), with the reserve absorbing the extra submission; if two folds need splitting, the cut order applies first. Record the exact resource ask if even the minimum fails (FAIL-on-affordability).

### Component 2: Arm C re-derived for parameter space [lambda = 0.57] (B3/A7)

**What**: The mechanism control, **redesigned rather than ported**. Round 1 priced a mechanical port of the parent's sample-space matching; the challenge showed that design's semantics do not survive the space change: at p ≈ 600k, a fixed random k-subspace captures ~k/p ≈ 0.3% of gradient energy, so norm-matching it means 30–250× amplification — noise injection, not a control, and the mechanism-honesty clause becomes vacuous. The A7 matching invariants are now **k(t), the update-norm-ratio trajectory, AND the basis-rotation rate** (measured as principal angles between B's realized basis at t and t−Δ — a diagnostic now logged in all arm-B runs, per B3). Candidate designs, evaluated by the required sim:
- (a) **Preferred (A7)**: a random orthogonal rotation of B's own realized basis — C's basis = R·U_B(t) with R a fixed random orthogonal transform applied implicitly (candidate implementations: random signed permutation composed with a subsampled randomized Hadamard transform — O(p log p) per apply, exactly orthogonal, norm-preserving). Matches k(t) and rotation rate by construction; the sim measures its captured energy and required amplification.
- (b) Random rotation **within the span** of B's tracked covariance factorization / recent-gradient history — selects non-spectrally inside the space where gradient energy actually lives, preserving the energy budget if (a)'s captured energy proves ~k/p.
- (c) Re-drawn random basis at B's measured rotation rate — matches rotation dynamics but not span; weakest candidate.
**Risks to this component**: (a) may still capture ~k/p energy (a Haar-rotated adapted basis is a uniformly random subspace at each instant), pushing the design to (b); implicit-transform implementation cost; replaying B's realized k(t)/ratio trajectories for adaptive configs; C reported in absolute terms must actually train, not collapse.
**Evidence for P_success = 0.65**: The invariant list is now derived for this space and the parent's replay plumbing is proven logic, but no implementation of any candidate exists in any space, and the challenge's energy arithmetic shows the naive version fails — this is design work with one round of empirical feedback (the sim), not adaptation. Rubric band 0.5–0.7.
**Quick test (REQUIRED before porting any arm C code — B3)**: Local CPU, ~20 min runtime: synthetic gradient stream at p = 600k with planted low-rank + noise structure (or gradients harvested from #5's synthetic task); for each candidate at k ∈ {8, 512, 2048}: captured-energy fraction, realized norm-amplification factor needed to match a logged arm-B kept-norm trajectory, principal-angle rotation-rate match, and whether the norm-matched control still descends on the synthetic task. The sim's output **determines the final control design within the A7 invariants**, in writing, before any cluster code.
**Pass criterion**: At least one candidate matches all three invariants and descends on the synthetic task with a loss curve within ~2× of arm A's (the pre-mortem's straw-control pivot indicator); design decision documented.
**Fail criterion**: Every candidate either requires noise-injection-scale amplification with no descent, or cannot match the invariants without >2 h surgery.
**If the quick test returns FAIL**: Arm C is required for the criteria's "solid contribution" bar but the minimum viable verdict (B−A) survives without it. Per the symmetric honesty clause (A7), with no fair control any B ≫ C mechanism claim is unavailable and a B-over-A win claims only "adaptive low-rank projection helps"; C's absence and the sim evidence are reported plainly. The round-1 fallback (fixed random subspace matched on time-averaged k/norm) is **no longer acceptable** — the criteria explicitly exclude it as norm-matched noise injection.

### Component 3: Per-fold inference power + MDE machinery [lambda = 0.48] (B4/A1/A6)

**What**: The decision rule needs per-fold 95% CIs on paired (B−A), (B−C), (C−A) per-era differences under the **A6 hierarchical bootstrap** (moving-block over era blocks with seed-level resampling nested inside; block length from each fold's own lag-1 ACF, recomputed, never inherited; RNG fixed; stability across ≥ 2 RNG seeds), plus the **standing per-fold MDE table** (95% CI half-width of the seed-averaged fold-mean difference) that feeds the A1 power-qualified kill criterion. This component builds and validates that machinery and measures the design's realized power before any TEST touch.
**Risks to this component**: CI half-widths at ~110 autocorrelated eras too wide to resolve ±0.005 (pre-mortem scenario 2 called the power-limited NULL the modal gate-passing outcome); recent-era ACF near the example model's 0.763 rather than the parent tuning block's 0.247; hierarchical resampling widening CIs relative to the parent's era-only machinery.
**Evidence for P_success = 0.70**: Parent exp-003 measured half-width 0.00219 at 255 eras/3 seeds; √(255/110) scaling gives ~0.0033 — below 0.005 but without the hierarchical seed term and at the lower ACF. Downgraded from round 1's 0.85 because the challenge showed the recent-window ACF risk and the seed-variance term were unpriced, and B4's pass threshold is now explicit. Rubric band 0.5–0.7, top.
**Quick test**: Local CPU, no new compute: parent per-era vectors (exp-001 `out/example_per_era_corr.csv`, exp-004 per-era diff series) restricted to the prospective ~110-era fold windows from #9; recompute lag-1 ACF and block length per window; hierarchical bootstrap under null and ±0.005-injected scenarios at 3 and 5 paired seeds; emit per-fold MDE.
**Pass criterion**: P(CI excludes 0 | true effect ±0.005) ≥ 0.6 on ≥ 2 of 3 prospective folds (the B4 threshold); median MDE ≤ ~0.005; conclusions stable across 2 bootstrap RNG seeds.
**Fail criterion**: The B4 condition fails — P(detect ±0.005) < 0.6 on ≥ 2 prospective folds.
**If the quick test returns FAIL**: **Wired, per B4 — this is not a proceed-with-a-caveat branch.** The A1 power-qualified kill wording goes into frozen `protocol.json` in either case (it is mandatory); a FAIL here means it is load-bearing: the run's most probable clean outcome is a power-limited NULL, the write-up must lean on the MDE table and the resourced decisive-test ask as primary content, and seeds go to 5 if #1's arithmetic permits (the only upward lever). The kill criterion cannot fire on a NULL whose per-fold MDE > 0.005 — that wording is frozen before any TEST touch, which is exactly what this component certifies.

### Component 4: Baseline gate — tuned AdamW ≥ 0.60× floored denominator [lambda = 0.24]

**What**: THE scientific gate and the dominant genuine uncertainty, now under full A5 semantics: per-fold denominators computed locally before freeze with the 0.010 absolute floor (low-signal flag if floored); realized ratio bands ([0.50, 0.60) near-miss → one targeted fix then accept; [0.45, 0.50) → fix ladder within the reserve's budget and stopping rule; < 0.45 → structural gap, reserve buys diagnosis + data-scaling learning curve, not repair); the 5-rung fix ladder with budgets 4/4/2/1/1 (regularization first per parent F12 signatures; network size available **only if** EU-1 timed arm B at the larger architecture — B1; era-recency weighting authorized); retry semantics (gate ratio only, VALID re-selection after fixes, TEST-touch count logged). Component defined as: gate passes on ≥ 2 of 3 folds.
**Risks to this component**: The parent shortfall was intrinsic (MLP-vs-GBT-ensemble gap on the hardest regime) rather than protocol-induced; 12 trials too few at the new scale; step count mis-set (mitigated: fixed by the EU-1 convergence run, outside the 12 trials, per A4); the denominator floor flagging a low-signal fold.
**Evidence for P_success = 0.55**: Unchanged from round 1 — the A5 machinery makes failure *cheaper and more informative*, not less likely. For: refit ending 5 eras before TEST, all usable eras, full features, and measured step counts are exactly the mechanisms that should close the 27% → 60% gap; practitioner NNs reach 0.015–0.03 vs the example model's ~0.0235. Against: the parent's best tuned MLP hit 0.0196 on older, easier eras; the gate eras are the hardest regime; fold gates share common causes so P(≥2/3) ≈ per-fold P. Rubric band 0.5–0.7, bottom.
**Quick test**: Fold-1 job phase 1 (gate-first, preserved from round 1): 12-trial A sweep on fold-1 VALID, refit at the selected config under the pre-registered stopping rule (steps scaled by rows_refit/rows_train), single TEST eval, realized ratio + THE assertion printed. ~2.5 h wall including queue. **Early warning one unit earlier (B1)**: EU-1's full-length arm-A run reports a proxy gate ratio on a VALID slice — if the proxy is < ~0.5, the fold-1 TEST gate will almost certainly fail, and that is known before EU-2 is sized.
**Pass criterion**: Fold-1 realized ratio ≥ 0.60 (and subsequently on ≥ 2 of 3 folds; [0.50, 0.60) resolves per the near-miss band).
**Fail criterion**: Ratio < 0.60 after the genuine sweep, refit, and the band-appropriate ladder spend.
**If the quick test returns FAIL**: Follow the pre-registered bands — no open-ended repair. Near-miss: one targeted rung, one re-check, accept. [0.45, 0.50): ladder within EU-5 under its stopping rule (futility: two consecutive rungs improving the ratio < 0.02 combined). < 0.45: structural — EU-5 buys the data-scaling learning curve, converting the failure into quantitative evidence on assumption 3. Terminal failure on ≥ 2 folds → the criteria's baseline-construction negative, reported with realized ratios, ladder trajectory, and TEST-touch counts. No optimizer comparison on failed-gate folds.

### Component 5: Filter integration + alpha=0 identity + planted-subspace + eigh fallback [lambda = 0.22] (A8)

**What**: The object of the run: `filt = SpectralGradientFilter(model, base_opt, rank=k)`, `filter_grad()` between `backward()` and `step()`, `normalize="none"` (H7), all sweep knobs reachable, CPU-fp64 eigh fallback wired from the start with a logged firing count. Verification depth per A8: (i) alpha=0 soft identity reproduces plain AdamW (the no-op path); (ii) **planted-subspace correctness check** — hard top-k recovers a planted dominant gradient direction on a synthetic task, exercising the top-k/adaptive/energy-threshold code paths the identity leaves untouched; (iii) seeded zero-predictor; (iv) the run-long diagnostics implemented here: per-step filtered-vs-unfiltered cosine, kept-norm fraction, realized k(t), and the **basis-rotation-rate (principal-angle) diagnostic** (B3) — all emitted in every arm-B run, with the selected config's distance-from-identity reported per fold. Arm B/C RNG consumption from a separate generator so seed pairing does not silently break.
**Risks to this component**: Hidden interface assumptions (model introspection, parameter grouping, device placement of the streaming basis); torch 2.5.1 (cluster pin) vs local API drift; identity not holding exactly; fallback patch not transplanting cleanly from the parent's B×B context to the (k+1)×(k+1) eigh; a filtering-path bug the identity cannot see (the planted-subspace check exists precisely for this).
**Evidence for P_success = 0.80**: Code exists at the verified path and ran in its home repo (H3/H5/H7); usage is four lines; but "worked in home repo" is one environment, and the parent's equivalent smoke needed fp64 to settle its identity cleanly. Rubric band 0.7–0.9.
**Quick test**: Local CPU (~1 h): 2-layer MLP, synthetic Numerai-shaped regression; (a) filtered training runs 200 steps, loss falls, all diagnostics emitted; (b) alpha=0 fp64 identity vs plain AdamW: 50-step parameter trajectories bit-identical or ≤ 1e-12; (c) planted-subspace: hard top-k recovers a planted dominant direction (cosine to planted direction > ~0.9, mirroring the parent's C1(b) calibration); (d) monkeypatched `torch._C._LinAlgError` → fallback fires, is logged, training continues; (e) zero-predictor eval ≈ 0 corr on synthetic eras.
**Pass criterion**: All five sub-checks pass; identity holds in fp64; rotation-rate diagnostic produces sane principal angles.
**Fail criterion**: Identity fails structurally (alpha=0 is not a no-op), planted direction not recovered, or integration needs > 2 h of surgery.
**If the quick test returns FAIL**: A structural identity failure is **STOP-and-report** per the brief — no file substitution, no fallback optimizer; nothing downstream is trustworthy until this passes. A planted-subspace failure is a filtering-path bug: fix in the run's copy and document, or STOP if it resists — an engagement-diagnostics deliverable without a working filter is the parent's failure shape and is not repeated here.

### Component 6: Numerical stability over long re-tuned schedules [lambda = 0.22]

**What**: The streaming rank-1 SVD factorization and per-step eigh must stay well-conditioned over the measured full schedules (state.md assumption 6), with the CPU-fp64 fallback firing rarely and recoverably, count logged and reported per run.
**Risks to this component**: Rank collapse over long horizons driving eigh non-convergence (parent audit Finding 5's failure class); decay-EMA interaction with much longer horizons (partially probed by the decay=0.999 stage-2 trial); silent basis degeneracy rather than loud failure (the A8 diagnostics — kept-norm, rotation rate — are the visibility instruments).
**Evidence for P_success = 0.80**: The mandated fallback is a working patch (parent `audit/rerun-exp-004/src/`); the eigh is small ((k+1)×(k+1), CPU); home-repo runs completed at comparable step counts on other tasks; nothing at this schedule length on this data exists. Rubric band 0.7–0.9.
**Quick test**: Short-schedule portion inside #8 (500-step runs at grid extremes, fallback count logged); fully retired by the EU-1 full-length arm-A run's schedule scale and fold-1's full arm-B schedules in #4.
**Pass criterion**: No NaN/Inf; fallback fires on < 1% of steps and every firing recovers; loss curves sane at all grid points run.
**Fail criterion**: Unrecoverable eigh failures or NaN at multiple grid points/seeds.
**If the quick test returns FAIL**: Localized failures (specific rank/adaptive configs) are excluded from the realized grid and documented (feeding the A2 scope wording if the exclusions push K_max < 512); global instability blocks arm B → implementation-infeasibility finding for this optimizer at this scale, treated as STOP-and-report, not substitution. The EU-1-timed eigh-every-N and GPU-fp32-eigh variants (B1) are the named, documented engineering options if instability is cost-coupled — if used, the write-up reports results under the named variant, never silently.

### Component 7: Fold-job resumability + realized-wall projection [lambda = 0.22] (B2, new in round 2)

**What**: Pre-mortem scenario 4 made atomic fold jobs a priced risk: a `--time` kill or shared-node failure must cost one trial, not one fold and one experiment unit. **Before any fold submission**: (i) per-trial result persistence to NFS as each trial completes — the parent exp-003 harness's append behaviour verified by kill-test, not assumed; (ii) refit checkpoints (model + optimizer + filter state + RNG streams) so a killed refit resumes; (iii) `--time` set at 2–3× the projection (24 h cap leaves room; a generous limit on the free partition costs queue priority, not money); (iv) resume-on-restart logic that skips completed trials (also the defense against duplicate submissions after orchestration-session interruptions — job IDs recorded in state.md per the compute profile); (v) folds 2–3 wall-clock **re-projected from fold-1 REALIZED wall time, not EU-1 numbers**, before their submission; (vi) any fold projecting > ~16 h split into a sweep-job + a refit/eval-job.
**Risks to this component**: The harness buffers results in memory and writes at exit (append does not survive SIGKILL); filter state (streaming basis) makes refit checkpoints larger and fiddlier than plain torch; resume logic subtly double-counting trials.
**Evidence for P_success = 0.85**: All standard engineering with existing scaffolding; the parent harness ran clean but was never killed mid-flight, so its crash behaviour is genuinely unknown — hence a component, not an assumption. Rubric band 0.7–0.9.
**Quick test**: Local (~45 min): run the adapted sweep harness on 6 synthetic fast trials; SIGKILL it mid-trial-4; verify trials 1–3 are on disk and a restart runs exactly trials 4–6; checkpoint a refit (including `SpectralGradientFilter` state) at step N, restore, verify bit-continuation for 10 steps in fp64.
**Pass criterion**: Kill-test and checkpoint roundtrip both pass; resume never re-runs a completed trial.
**Fail criterion**: Persistence or restore cannot be made reliable in > 2 h of surgery.
**If the quick test returns FAIL**: Do not submit multi-hour atomic folds on hope — restructure into smaller submissions (sweep-job + refit/eval-job per fold) from the outset and accept the unit-count pressure via the preserved cut order. This is a logistics defect with a mechanical fix; it does not touch the science.

### Component 8: Full-scale throughput/VRAM + realized rank grid + B1 additions [lambda = 0.18]

**What**: EU-1, the pre-flight `--qos=debug` job, expanded per B1. Original scope: full v5.0 feature set (~2.3k features → ~600k-param MLP), largest (fold-3 refit) shard, arm-B s/step across the rank grid {8, 32, 128, 512, 2048}, VRAM peaks, short-schedule stability, on-cluster alpha=0 re-assert (also covers the torch-2.5.1 compat risk). **B1 additions**: (i) **one full-length arm-A convergence run** — a generous step budget with VALID-slice score monitored to plateau, yielding the real steps-to-convergence (fixed outside the 12 trials per A4, and the anchor for #1's packing arithmetic and the refit stopping rule's scale) and a **proxy gate ratio** on the VALID slice (vs the example model's VALID-slice mean — the gate bit one unit early); (ii) **arm-B timing at the largest plausible gate-fix architecture** (fix-ladder rung 2, ~2.5–3M params, e.g. 2304-1024-256-1: basis p×k ≈ 24 GB at k=2048) — without this measurement, A5 restricts gate fixes to p-preserving levers and rung 2 is skipped; (iii) **eigh-every-N-steps and GPU-fp32-eigh variant timings** — the named engineering variants that could buy back rank 512/2048 if the cubic eigh wall materializes (documented variants if ever used, never silent); (iv) realized k(t) of adaptive configs on the short runs, so "the feasibility cap binds the adaptive arms" is checkable (A2), plus the rotation-rate diagnostic feeding #2's design.
**Risks to this component**: Rank-2048 (and possibly 512) measured infeasible — expected-manageable: the realized grid is capped and documented, with the A2 kill-scope machinery (K_max < 512 → "no evidence at feasible ranks", decided and written into `protocol.json` at that moment, before any fold job — not after a NULL arrives); int8 shard residency + cast overhead; basis VRAM stacking with activations; NFS→/ephemeral staging of multi-GB shards; the convergence run revealing a step count that strains packing.
**Evidence for P_success = 0.70**: Parent numbers anchor the small scale (1.19 ms/step filter-off; filter ~2× at default rank per the repo); the arithmetic (5 GB basis, 11–15 GB int8 shard) fits 48 GB on paper; nothing at this p/rows/rank has run. P prices "the design is feasible with a defensible realized grid and a measured cost anchor", not "rank 2048 is feasible". Rubric band 0.5–0.7 / 0.7–0.9 boundary.
**Quick test**: The EU-1 job itself (~45–75 GPU-min projected, `--qos=debug` if it stays under 2 h; wall ~2 h including shard build, rsync, queue): everything listed under "What".
**Pass criterion**: Arm A trains at full scale and the convergence run plateaus within the job; realized rank grid retains ≥ 4 points **including ≥ 512** (else the A2 wording is triggered and pre-registered immediately); VRAM ≤ 44 GB; no unexplained NaN; second-architecture and variant timings recorded.
**Fail criterion**: Full feature set does not fit even with int8 residency and reduced batch (→ documented feature reduction, a mandated limitation, not silent), or arm B is > 10× arm A at every rank ≥ 32.
**If the quick test returns FAIL**: Feature-set reduction is the pre-authorized, must-be-documented fallback; grid capping is expected and documented, with kill-scope wording per A2. If arm B is order-of-magnitude slow at all useful ranks even under the timed variants, trials shrink symmetrically for both arms (matched budget preserved) and the reduction is reported; if that still fails, record the exact resource ask (FAIL-on-affordability).

### Component 9: Fold arithmetic + protocol.json + assertions + gate denominators [lambda = 0.16] (A5/A8)

**What**: Compute 3 expanding-window folds from the realized usable-era list: TRAIN (all usable eras to T) / embargo 4 / VALID (~96 eras, selection only) / embargo 4 / TEST (~110 eras); TEST blocks contiguous, non-overlapping, most recent data, coverage ≥ 0.95; boundaries written to `protocol.json` together with the **raw-era ↔ usable-index mapping** (A8). THE assertion, hard-asserted and printed per fold: `min(test_eras) − max(refit_train_eras) == E + 1 == 5`. **Companion assertion (A8)**: every TEST era of every fold present in `data/v5.0_validation_example_preds.parquet`. **A5 denominators**: the example model's mean per-era numerai_corr on each prospective TEST block computed locally from parent `out/example_per_era_corr.csv` the moment boundaries exist — before freeze, touching example preds only; per-fold denominators with the 0.010 floor (and any low-signal flag) recorded in `protocol.json`.
**Risks to this component**: Era-list gaps breaking the exact ==5 arithmetic; off-by-one at the refit boundary; coverage vs V≈96/S≈110 targets not simultaneously satisfiable; a denominator landing under the floor (handled: flagged, not fatal).
**Evidence for P_success = 0.85**: Parent era arithmetic PASSed (exp-001: E=4, purged splits, 643 usable eras — comfortably enough for 3×(110+96+8)); the expanding-window layout is new but deterministic; the denominator computation is a 10-minute pandas job on an existing file. Rubric band 0.7–0.9.
**Quick test**: Local CPU (~1 h): fold-boundary script against the realized era list; per fold assert ==5, disjointness, contiguity, non-overlap, coverage; assert example-preds coverage; compute and floor the denominators; emit `protocol.json` with the mapping; eyeball printed boundaries and denominators.
**Pass criterion**: All asserts pass on all 3 folds; `protocol.json` written with boundaries, mapping, and denominators; targets met within the coverage constraint.
**Fail criterion**: The ==5 condition cannot be satisfied exactly for some fold given era-list gaps.
**If the quick test returns FAIL**: The assertion is defined on the usable-era index sequence — if raw numbering has gaps, the boundary definition (not the assertion) is adjusted so the semantic condition holds, with the adjustment and the mapping documented in `protocol.json` before any training run (the mentor's requirement). This is arithmetic; it gets fixed, not worked around.

### Component 10: Numerai v5 data + example preds [SKIP]

Downloaded and used end-to-end by the parent (exp-001 PASS): `v5.0_train.parquet` (2.4 GB), `v5.0_validation.parquet` (3.8 GB), `v5.0_validation_example_preds.parquet`, `v5.0_features.json`, `out/example_per_era_corr.csv` in the parent run's directories. Action: symlink into this run's `data/`, verify row/era counts match parent-recorded values. Never re-download.

## Dependency Graph

```
 #10 data symlink [SKIP]
      │
      ├──────────────────────────────┬─────────────────────────────┐
      ▼                              ▼                             ▼
 #9 fold arithmetic            #5 filter integration          #7 resumability
   + protocol.json               + identity + planted-           kill-test +
   + THE + coverage asserts      subspace + fallback             refit checkpoints
   + A5 gate denominators        + A8/B3 diagnostics             (local CPU)
      │        │                 (local CPU)                       │
      │        ▼                     │                             │
      │   #3 power sim + MDE         ├────────────► #2 arm C       │
      │   (hierarchical bootstrap,   │              B3 CPU sim →   │
      │    B4 wiring; local CPU)     │              design freeze  │
      │        │                     │              (local CPU)    │
      ▼        │                     ▼                             │
      └────────┼────► #8 EU-1 pre-flight ◄─────────────────────────┘
               │        (throughput/VRAM/rank grid + B1: full-length
               │         arm-A convergence + proxy gate ratio,
               │         2nd-architecture timing, eigh variants;
               │         carries #6 short-schedule check)
               │                     │
               │                     ▼
               └──────► #1 packing arithmetic (local, no compute)
                                     │
                                     ▼
                    PROTOCOL FREEZE (protocol.json — see checklist)
                                     │
                                     ▼
                    #4 baseline gate, fold 1 ─── retires #6 (long schedules)
                    (EU-2: fold-1 job, gate-first, B2-resumable)
                                     │ gate passed; re-project from
                                     ▼ fold-1 REALIZED wall (B2)
                    folds 2 + 3 (EU-3, EU-4, parallel) ── EU-5 reserve
                                     │                    (A5 ladder / structural-gap
                                     ▼                     learning curve / forced
                    verdict per pre-registered rule        reruns / B2 fold split)
                    (kill criterion: four conditions —
                     execution + A1 power + A2 scope + A3
                     under-exploration; clean kill ⇒ STOP)
```

## Parallelisation Plan

**Wave 1 (now, all local, no mutual dependencies)**: [#9, #5, #7] — #10 symlink first (minutes). #3 as soon as #9 emits prospective boundaries; #2's energy sim as soon as #5's synthetic task and logging format exist (same session; the sim itself can start on synthetic streams immediately).
**Wave 2 (after #5, #9; #2 and #3 results in hand)**: [#8] — EU-1, one `--qos=debug` job carrying the B1 additions and #6's short-schedule check.
**Wave 3 (after #8, no compute)**: [#1] packing arithmetic (seeds 5-vs-3 decided here); then **protocol freeze** per the checklist below, before any TEST touch.
**Wave 4 (after freeze)**: [#4] fold-1 job (EU-2), gate-first, B2-resumable. Fold 1 runs **alone** — its gate outcome is the highest-information remaining bit and fold gates are correlated.
**Wave 5 (after fold-1 gate passes)**: folds 2 and 3 (EU-3, EU-4) in parallel (2 GPUs, within the 6-GPU cap), sized from fold-1 realized wall (B2).
**Conditional**: EU-5 reserve — A5 ladder / structural-gap learning curve on a failed-gate fold, forced reruns, or absorption of a B2 fold split. All bootstrap/verdict/MDE analysis is local CPU and consumes no experiment unit.

## Protocol-Freeze Checklist (what must be in `protocol.json` before EU-2)

Frozen before any TEST touch, per the round-2 criteria — the decomposition's job is to ensure every input exists by Wave 3:

1. Realized fold boundaries + raw-era↔usable-index mapping + coverage (#9).
2. Per-fold gate denominators with the 0.010 floor and any low-signal flags (#9, A5).
3. Gate bands, fix ladder with per-rung budgets 4/4/2/1/1, stopping rule, retry semantics; whether rung 2 (network size) is available — determined by whether #8 timed the second architecture (B1/A5).
4. A4 staged arm-B allocation: realized rank grid (from #8), transferred LR, stage boundaries, the decay=0.999 stage-2 probe, pruning statement.
5. Converged step count (from #8's B1 convergence run) and the refit stopping rule (steps × rows_refit/rows_train).
6. **A1 power-qualified kill wording — mandatory (B4)**; #3's sim result recorded alongside, determining its write-up prominence.
7. A2 kill-scope wording tied to the realized grid; if K_max < 512, the "no evidence at feasible ranks" form is written in at the moment the grid is realized, not after results.
8. A3 under-exploration signatures (grid-boundary; non-monotone with range < ~2× across-seed sd; ≥ 4× LR shift) and the HURTS downgrade rule.
9. A6 inference spec: hierarchical bootstrap, seed count (5 or 3, from #1), block-length rule, RNG seeds, MDE definition.
10. Arm C final design from #2's sim, its invariants, and the symmetric honesty clause's evidence requirements (A7).
11. Eigh-variant policy: base configuration is per-step CPU-eigh with fp64 fallback; eigh-every-N / GPU-fp32-eigh usable only as named, documented variants with #8's timings attached (B1).

## Experiment-Unit Packing Plan (binding sizing, per the Step 4 criteria)

The 5-unit cap counts Slurm submissions, with short trials batched inside each job's wall time:

| Unit | Content | Est. wall | Components |
|------|---------|-----------|------------|
| EU-1 | Pre-flight `--qos=debug`: shard build + rank-grid throughput at planned AND gate-fix architectures + eigh-variant timings + VRAM + short-schedule stability + on-cluster identity assert + **B1 full-length arm-A convergence run + VALID-slice proxy gate ratio** + adaptive k(t) and rotation-rate logging | ~1–2 h | #8, #6(short), #5(guard), feeds #1/#2/#4 |
| EU-2 | Fold 1, gate-first, **B2-resumable** (per-trial NFS persistence, refit checkpoints, `--time` = 2–3× projection): A sweep(12) → A refit → A TEST eval + gate + assertions; if gate passes: B staged sweep(12), refits A/B/C × seeds (target 5, floor 3), one TEST pass | ~3–6 h projected (from EU-1 measurement; `--time` 2–3×) | #4, #6(long), main comparison fold 1 |
| EU-3 | Fold 2 (same structure; submitted only after fold-1 gate pass; sized from fold-1 REALIZED wall) | re-projected | main comparison |
| EU-4 | Fold 3 (same structure, largest shard; same re-projection; split into sweep-job + refit/eval-job if > ~16 h projected, absorbing EU-5) | re-projected | main comparison |
| EU-5 | Reserve: A5 fix-ladder retry / structural-gap diagnosis + data-scaling learning curve / forced rerun / B2 fold-split absorption | ≤ 4 h | contingency |

Fold jobs exceed the profile's ~30-GPU-min *preference* but batch what would otherwise be ~40+ short runs each, per the criteria's explicit sizing instruction; each requests 1 GPU inside the 24 h cap. Overflow cut order (preserved, reported if used): cap rank grid at the largest measured-feasible point → reduce seeds (5 → 3 floor) → reduce folds.

## Showstoppers

**None.** No component sits below P = 0.05, and nothing requires more than 2 concurrent L40s against a 6-GPU cap. Named expected-manageable hazards, not showstoppers: (a) rank-2048 (possibly 512) time-infeasible — capping is a documented design realization, with the A2 kill-scope wording pre-registered the moment the grid is realized; (b) baseline gate unreachable on ≥ 2 folds — produces the criteria's defined baseline-construction negative plus, for a structural gap, the data-scaling learning curve; (c) power insufficient at ~110 eras — produces the pre-registered power-limited NULL with the MDE table and a resourced ask (B4 makes this wiring frozen, not narrative). Also binding: a structural alpha=0 identity failure (#5) is STOP-and-report — a user-decision point, not a component to engineer around; a clean NULL/HURTS through all four kill conditions ends the line with no rescue components downstream.

## Overall Project P_success

**Full unqualified-verdict deliverable** (WINS/HURTS/NULL per the pre-registered rule with no A1/A2/A3 qualification needed: gate passed ≥ 2/3 folds, adequate power, K_max ≥ 512, fair arm C, resumable execution — components #1–#9):

P = 0.70 × 0.65 × 0.70 × 0.55 × 0.80 × 0.80 × 0.85 × 0.70 × 0.85 ≈ **0.06**

**Informative outcome** (recognizing that under the round-2 criteria a #4 gate failure converts to the pre-registered baseline-construction negative, a #3 power shortfall converts to a power-limited NULL, a #2 failure downgrades mechanism attribution without killing the B−A verdict, and a capped grid converts to the scope-limited form — product over the components whose failure has no defined conversion: #1, #5, #6, #7, #8, #9):

P = 0.70 × 0.80 × 0.80 × 0.85 × 0.70 × 0.85 ≈ **0.23**

(Assumes independence; realistically correlated — #8 passing at comfortable margins raises #1 and #4 together, and the three fold gates share common causes, which is priced into #4 directly. The round-2 number for the unqualified verdict is *lower* than round 1's 0.09 — honestly so: round 1 was counting outcomes the challenge showed could be artifacts (an under-explored HURTS, a power-limited NULL, a truncated-grid kill) as full verdicts. The informative-outcome figure is unchanged at ~0.23 because the amendments converted those failure shapes into pre-registered, publishable qualified findings rather than reducing their probability. The uncertainty remains concentrated where it should be — #4, the never-tested scientific gate — and the plan still buys that bit as early and cheaply as possible: proxy ratio in EU-1, gate-first fold 1, bands and ladder pre-registered before the ratio exists.)

## Round-2 Challenge Addendum (Step 6, binding — D-series)

The round-2 challenge (mentor verdict MINOR_REVISIONS; construct gate OK) surfaced
thirteen fix-now-free amendments, triaged in `challenge/limitation-triage.md` (full
rationale and sources there — this section is the operative plan change). All are
zero-GPU wording/logging/sim-extension/job-structuring changes, legitimate only
before the protocol freeze, and all are **binding on Steps 7–9**. None adds an
experiment unit; extra submissions (EU-1b, fold splits) charge to the EU-5 reserve
as already accounted. The protocol-freeze checklist gating EU-2 is extended by the
items below (bringing it from 11 to 16 items, D-items grouped where they land).

- **D1 (→ #5, #8, protocol.json)**: 4th under-exploration signature — arm B's
  selected config still improving at step cutoff (VALID slope over final segment
  above a pre-registered threshold) → A3 downgrade path. Log VALID-score-vs-step
  for every arm-B trial. Claim language: "at matched step budget".
- **D2 (→ #2)**: arm C expected primary = candidate (b) with strictly larger
  history span (rotate within r > k ambient space, truncate to k). B3 sim gains
  distinctness criteria vs B (principal-angle floor, update-cosine ceiling) and a
  positive-control-for-the-control: acceptable C must FAIL the #5 planted-subspace
  task where B succeeds. Empty-middle fallback pre-registered: drop C, scope to
  (B−A), redirect C's refit-seed budget toward 5 seeds.
- **D3 (→ #3, protocol.json)**: MDE honesty — derive 0.005 in one sentence in
  protocol.json; report as "CI half-width (≈50%-power MDE)"; connect the A1 kill
  condition to the B4 sim's P(detect ±0.005); state the kill bound in relative
  terms vs realized arm A. #3's sim adds a CI coverage check at 3 and 5 seeds;
  sub-nominal 3-seed coverage makes 5 seeds a validity lever in #1's packing.
- **D4 (→ #8, A4 stage 1, checklist item 4)**: stage-1 transferred LR scaled per
  grid point by EU-1-measured kept-norm fraction; one pre-registered stage-1 trial
  is a kept-norm-corrected LR probe at the modal rank (frozen correction rule).
  Checklist item 4 is written as a function of EU-1's kept-norm diagnostics. Extra
  signature: stage-1 winner within one grid step of the identity-distance minimum.
- **D5 (→ #9, protocol.json)**: deterministic refit-extension rule beside the
  stopping rule (loss slope over last 5% of steps > threshold → extend once by a
  stated factor, logged). Cross-fold trial-step semantics fixed by a one-line rule.
- **D6 (→ #8 / EU-1 structure)**: EU-1 restructured as ordered, individually
  persisted phases — (1) shard build; (2) timing table (both architectures + eigh
  variants); (3) identity re-assert + stability; (4) convergence run LAST,
  checkpointed, VALID-score series streamed to NFS, plateau-detection stop.
  Pre-authorized fallback: phase 4 outgrowing the debug window continues as EU-1b
  (normal queue, --time=6h, resumable, charged to EU-5). Truncation anchor
  pre-registered: checkpointed lower bound + stated extrapolation rule, declared.
- **D7 (→ #4, EU-1 prep)**: proxy gate ratio calibrated BEFORE EU-1 submission
  using the example model's own VALID-vs-TEST offset per fold (local pandas job on
  out/example_per_era_corr.csv). Reserve pre-commitment: calibrated ratio < 0.60 →
  EU-5 earmarked for the fix ladder before EU-2 launches; EU-2 phase 2 (arms B/C)
  conditional in the sbatch structure. Proxy gates sizing only.
- **D8 (→ #3, checklist)**: pooled cross-fold secondary estimand pre-registered
  (concatenated era-level paired (B−A) under one hierarchical MBB; cannot overturn
  the per-fold rule). Pre-freeze decisiveness checkpoint: power sim FAIL AND
  realized K_max < 512 → considered branch (2 folds with proportionally longer
  TEST blocks, reported) or an explicit protocol.json declaration that the run
  targets a bound.
- **D9 (→ protocol.json)**: A3 signature 2's sd source named (arm-A refit sd at
  the same fold, or a duplicated stage-1 trial at rank ≈ 32); stage-1 enumeration
  reconciled with trial count (slack = D4 probe + optional D9 duplicate).
- **D10 (→ #9)**: after the three gate denominators are computed pre-freeze:
  either record all comfortably above the 0.010 floor, or pre-register the
  flagged fold's role in the decision rule and gate-failure tally before the
  number can motivate the choice.
- **D11 (→ #2/#6)**: post-hoc invariant-match report (realized principal-angle and
  norm-ratio between B and C over the full schedule, per fold) pre-registered as
  the fair-control evidence the symmetric honesty clause consumes.
- **D12 (→ protocol.json)**: kill wording instantiated on all axes — "dead for
  this problem class as instantiated (Numerai v5 tabular regression, MLP,
  p ≈ 600k, k ≤ K_max)".
- **D13 (→ all EUs)**: protocol.json is the single operational spec; every fold
  job prints which checklist items it satisfies. Stage-2 priority rule: sharp
  stage-1 LR sensitivity → the decay=0.999 probe is the trial dropped.

Future-work rows FW1–FW4 (power ceiling, generality axes, rank-grid ceiling,
decay/warmup study) carry to Step 11's Future Work section via
`challenge/limitation-triage.md`; they are not components of this run.
