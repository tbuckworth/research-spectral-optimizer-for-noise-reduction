# Pre-Mortem Analysis

## Setting

It is six months from now. The research on "Spectral Optimizer redo: p×p filter, walk-forward split, rank sweep" has been completed, and the run did not deliver a valid answer to the motivating question. Under this run's criteria a clean WINS, HURTS, or NULL is success, so "did not hold" here means one of two things: the run ended without an interpretable verdict (again — this would be the second consecutive run on this question to do so), or it delivered a verdict whose meaning collapsed on later inspection. Both paths are reconstructed below.

One calibration note specific to this follow-up: the parent run is not a hypothetical prior — it is empirical evidence about how this line actually fails. It failed through (a) an object chosen by filename match and frozen, (b) a tripped pre-registered safeguard (baseline below its own sanity band) reinterpreted as "regime drift" instead of treated as a stop, (c) an asymmetric tuning budget rationalized by cost, and (d) a broken protocol that every individual gate passed. The parent's own pre-mortem predicted the threshold-miscalibration failure (its Scenario 1) and it materialized anyway, because the reinterpretation happened after the evidence arrived. The scenarios below therefore emphasize a specific shape: **pre-authorized flexibility clauses (grid capping, seed cuts, "proceed with the limitation stated") interacting with hard terminal rules (the baseline gate, the kill criterion) to produce outcomes that are procedurally clean and scientifically wrong.** These are systemic interactions the decomposition's per-component P values do not price.

## Scenarios (ranked by priority)

### 1. The gate did what gates do — and the run ended with a second consecutive non-answer [Likelihood: High | Severity: High]

**What happened**: Fold 1 ran gate-first as designed. The 12-trial AdamW sweep on VALID, refit on all usable eras, single TEST evaluation — and the realized ratio printed 0.44. The corrected protocol had genuinely closed most of the parent's gap (27% → 44%: refit, all eras, full features, re-tuned steps all helped, confirming the shortfall was *partly* protocol-induced), but the remainder was intrinsic. The Numerai example model is a mature GBT ensemble refined over years; a single MLP given 12 trials, evaluated on the hardest recent-era regime, did not reach 60% of it. EU-5, the sole reserve unit, was spent on baseline fixes — but "more features, more steps, larger network, LR schedule, target transform, better regularization" is a six-dimensional repair space, and 12 more trials found 0.51. The fold gates share common causes, so folds 2 and 3 were never viable once fold 1 settled. The run reported the criteria's baseline-construction negative: "the gate was unreachable under this budget."

That outcome is defined, honest, and explicitly priced by the criteria — but it means the two-run arc ends with: *we still do not know whether the Spectral Optimizer helps, because we could not build a baseline the protocol accepts.* The motivating question survives both runs unanswered. What made this hard to avoid is that assumption 3 ("the parent's shortfall was protocol-induced, not intrinsic") is tested for the first time at the exact moment it is load-bearing, and the 0.60× constant was set from a practitioner range (NNs at 0.015–0.03 vs example ~0.0235) that was never conditioned on this architecture, this trial budget, or the recent-era regime. Note the parent's best-ever tuned MLP reached 0.0196 mean per-era corr on *older, easier* eras with a favorable setup — 0.60× of the example model on recent eras is likely an absolute bar in the 0.012–0.018 range, i.e., at or above the parent's best under materially harder conditions.

There is also a neglected two-sided calibration issue: the gate's denominator is a per-fold random variable. If the example model itself scores low on some recent 110-era TEST block (regime drift hits it too), the gate becomes easy to pass but the fold carries little signal; if it scores high, the gate may be unreachable. Nobody has looked at the denominator per prospective fold, and it is computable locally, for free, today.

**Root cause**: The 0.60× threshold and the "fix the baseline with remaining budget" recovery path were both specified without any empirical anchor under the corrected protocol — the run's dominant uncertainty (P=0.55) has an open-ended repair plan funded by a single reserve unit, and the correlated fold gates make "≥ 2 of 3" effectively one Bernoulli draw.

**Early warning signs**:
- **Available now, zero compute**: the example model's mean per-era numerai_corr computed locally on each of the three prospective TEST blocks from `v5.0_validation_example_preds.parquet` + the fold boundaries from #8. This fixes the absolute bar per fold before any GPU spend.
- **EU-1**: one full-schedule arm-A trial at plausible hyperparameters scored on a VALID slice — a proxy ratio a full experiment unit earlier than the plan currently learns it.
- **Fold-1 phase 1**: the VALID proxy ratio printed before the refit (already planned) — if it is < ~0.5, the TEST gate will almost certainly fail.

**Mitigation**:
- Compute the per-fold denominator locally before freezing `protocol.json`; if the implied absolute bar on any fold exceeds ~0.018, the gate constant is probably miscalibrated for this budget and the time to say so is before pre-registration, not after fold 1.
- Pre-specify the baseline-fix ladder now: the ordered list of fixes, the trial allocation for each, and a stopping rule — replacing the open-ended "spend remaining budget fixing the baseline". Prioritize the fixes with parent evidence behind them (the parent's F12 signatures showed the baseline under-regularized at its grid boundary → regularization and network size first).
- Add a near-miss band to the pre-registration: realized ratio in [0.50, 0.60) means "one targeted fix, then accept the outcome"; ratio < 0.45 means the gap is structural and the reserve should buy diagnosis, not repair.
- If the gate fails terminally: spend any remaining budget on a data-scaling learning curve for the baseline (corr vs training-era count). That converts the baseline-construction negative into new quantitative evidence about assumption 3 — the one question this failure mode can still answer.

**Pivot indicator**: Fold-1 realized ratio < 0.45 after the genuine 12-trial sweep and refit → do not spend EU-5 on incremental fixes (12 trials will not close a 15-point ratio gap); go directly to the baseline-construction negative with a diagnosis, preserving the reserve for the learning-curve evidence.

**Load-bearing assumption**: state.md assumption 3 — the parent's baseline shortfall was protocol-induced rather than intrinsic to "single MLP, 12 trials, recent-era regime vs a GBT ensemble yardstick".

---

### 2. NULL by power, killed by rule [Likelihood: Medium | Severity: High]

**What happened**: The gate passed (ratios 0.63–0.71 — the protocol fixes worked). The sweep ran genuinely at matched budget. The per-fold paired differences came back +0.0021, +0.0009, −0.0004, every 95% CI straddling zero with half-widths around 0.0035–0.005. Verdict: NULL. The kill criterion's preconditions were all satisfied — gate passed, sweep genuinely ran — so it fired as written: *the line is dead for financial tabular regression; write the clean negative and STOP.* Six months later a reader computed what the design could actually see: with ~110 TEST eras per fold (vs the parent's 255) and recent-era autocorrelation near the 0.763 the parent measured across all validation eras (not the 0.247 on its older tuning block), the moving-block bootstrap's effective sample size per fold was small, and the minimum detectable effect per fold was ~0.004–0.008 per-era corr — 15–35% of the example model's entire edge, larger than almost any single training-pipeline change ever produces in this domain. The study was structurally capable of detecting only an implausibly large effect. The "dead line" verdict actually meant "effect smaller than ±0.005, sign unresolved" — but the kill criterion had already terminated the line, by design, with no follow-up permitted.

This is the modal end-state conditional on the gate passing, and it was visible in the plan's own numbers: component #2's pass criterion is P(detect ±0.005) ≥ 0.6 — a bit better than a coin flip, at an effect size that is itself optimistic for an optimizer-level intervention — and #2's fail branch explicitly says "the run proceeds with the limitation stated: the ≥2-of-3 rule will mostly resolve to NULL". So a #2 FAIL does not stop anything; it converts into a foreseeable power-limited NULL, and the frozen criteria treat that NULL identically to an adequately-powered one when the kill criterion fires.

**Root cause**: The decision rule's NULL conflates "evidence of no effect" with "insufficient power to resolve the sign", and the kill criterion is conditioned on execution-validity (gate passed, sweep ran) but not on evidential sufficiency (was the design capable of a non-NULL outcome at plausible effect sizes). This is the same class of error as the parent's — a safeguard verifying that procedure happened, treated as verifying that the conclusion is licensed — one level up.

**Early warning signs**:
- Component #2's local power simulation (already planned, zero GPU): median CI half-width > ~0.004, or lag-1 ACF on the prospective TEST-era windows > ~0.5 (recomputed on the recent-era windows, which the plan already requires). Both visible before any cluster job.
- Fold-1 realized CI half-width after EU-2 — if it exceeds ~0.004, folds 2–3 will resolve NULL almost regardless of the true effect.

**Mitigation** (all zero-compute, and all must land **before** `protocol.json` freezes — this is a wording change to the criteria's kill criterion, legitimate now, illegitimate after TEST is touched):
- Add a power qualification to the kill criterion: NULL is terminal only if the realized per-fold MDE (from the fold's own block-bootstrap, reported regardless) is ≤ a pre-registered bound (e.g., 0.005). Otherwise the verdict is "NULL (power-limited): |B−A| bounded within ±MDE; line not declared dead", and the resourced ask for a decisive test goes to future work.
- Report the per-fold MDE next to every CI as a standing table — this makes the distinction auditable rather than narrative.
- If #1's packing arithmetic permits, raise paired seeds to 5 (seeds are the only power lever; the criteria already authorize seeds as the flexible margin, currently only downward).

**Pivot indicator**: If the #2 simulation shows P(CI excludes 0 | true effect ±0.005) < 0.6 on ≥ 2 of the 3 prospective folds, stop and amend the kill-criterion wording before pre-registration — after freeze, the honest options collapse to executing a kill the design predetermined.

**Load-bearing assumption**: That a NULL under the pre-registered rule at ~110 autocorrelated TEST eras per fold constitutes evidence of absence strong enough to terminate the research line.

---

### 3. The feasible rank grid quietly stopped covering the hypothesis [Likelihood: Medium | Severity: High]

**What happened**: EU-1 measured what the decomposition flagged: at p ≈ 600k, arm B's per-step cost was fine at ranks 8/32/128, ~8× arm A at 512 (the O(p·k) basis update and re-orthogonalization, not just the eigh), and ~40× at 2048, where the 2049×2049 CPU eigh alone cost ~0.5 s/step against a ~14k-step schedule. The packing arithmetic did what the brief pre-authorizes: capped the realized grid at {8, 32, 128} plus the adaptive rules, documented as a design realization ("capped at what is feasible for p"), not a failure. The sweep ran genuinely at matched budget over that grid; VALID selected rank 32; the verdict came back NULL/HURTS; the kill criterion's preconditions read satisfied — the sweep *did* genuinely run — and the line was declared dead.

The problem surfaced in review: 128 directions out of 600k is 0.02% of parameter space. Prior finding H3 ("rank ≤ 4 destabilizes, ~10 fastest") comes from problems with p in the hundreds-to-thousands; the equivalent useful-rank region at p ≈ 600k is plausibly in the thousands — exactly the part of the grid feasibility removed. The adaptive rules did not rescue this: effrank on a 600k-dimensional streaming covariance naturally selected k in the high hundreds, and the same cost wall (or an explicit cap) clamped it into the tested region. So the kill verdict silently generalized "dead at k ≤ 128" to "dead", and — because the kill criterion forbids follow-ups — the one regime where the mechanism had room was never tested and never can be under this line.

**Root cause**: Two individually reasonable clauses interact without a joint definition: the brief's "grid capped at what is feasible for p" (making truncation a documented non-failure) and the criteria's "kill if the sweep genuinely ran" (making the truncated sweep kill-qualifying). Nobody defined the minimum realized grid that still constitutes *testing the hypothesis* rather than testing its cheapest corner.

**Early warning signs**:
- EU-1's s/step curve across the rank grid — the whole point of the pre-flight; the warning is specifically "arm B > ~10× arm A at every rank ≥ 512".
- The adaptive rules' realized k(t) on the EU-1 short runs: if effrank/gap naturally chooses k above the feasible cap, the cap binds the adaptive arms too, and the entire sweep lives in the truncated region. One logged number.

**Mitigation**:
- Pre-register the scope rule now, in `protocol.json`: the kill verdict's wording is automatically scoped to the realized grid ("dead for k ≤ K_max on this problem"), and if K_max < a pre-registered floor (proposal: 512), a NULL/HURTS downgrades from "line dead" to "no evidence at feasible ranks; higher ranks are a costed future-work ask" — the kill fires only above the floor.
- EU-1 should also time two cheap engineering variants that could buy back rank 512/2048: eigh every N steps instead of every step, and GPU-fp32 eigh with the existing CPU-fp64 fallback. Either is a modification of the object under test, so if used it must be a named, documented variant in the write-up — but knowing the price now is free.
- Log realized k(t) for adaptive configs in every run so the "cap binds the adaptive arms" condition is checkable from artifacts, not memory.

**Pivot indicator**: EU-1 shows the feasible grid tops out below 512 → the kill-scope wording is decided *at that moment*, written into `protocol.json` before any fold job is submitted. Deciding it after a NULL arrives is the parent's "regime drift" move with new nouns.

**Load-bearing assumption**: That the useful-rank region for a 600k-parameter model lies inside the compute-feasible portion of the grid — the implicit premise that makes "cap at what is feasible" harmless.

---

### 4. The fold job was one atomic 14-hour Slurm submission, and the run died of logistics [Likelihood: Medium | Severity: Medium]

**What happened**: The mandated step-count re-tuning settled at ~3 epochs over the fold-3 refit shard (~4.7M rows at B=1024 ≈ 14k steps) — the parent's 3 s/run anchor, extrapolated through ~3× per-step cost and ~7× steps, had become ~10-minute arm-A trials and 20–90-minute arm-B trials depending on rank. Fold jobs projected at 4 h from EU-1's 500-step measurements ran 2–3× longer in practice: the projection missed the converged step count (EU-1 measured s/step, not steps-to-convergence), the NFS→/ephemeral staging of a ~14 GB shard, and queue variance on the shared free partition. Fold 3 — the largest shard — hit its `--time` limit with the arm-B sweep 9 trials in and no per-trial persistence; Slurm killed it and the in-job state evaporated. EU-5 went to rerunning fold 3 with a trimmed grid, which tripped the criteria's own clause: "not informative if the rank sweep was truncated below the matched budget". A second interruption (shared-node contention; another user's job OOMing the node) forced a fold-2 partial rerun there was no unit left to pay for. The final state: a clean fold 1, a clean fold 2 on the second attempt, a budget-truncated fold 3 — a verdict resting on 2 usable folds with an asterisk the Step 10 audit correctly flagged, in a design whose decision rule needed agreement on 2 of 3.

The remote-orchestration layer amplified this: every fold job spans multiple orchestration sessions (11 h of 60 s SSH polls), and the profile's own warning — record job IDs in state.md or risk duplicate submissions on session recovery — is exactly the discipline that degrades under a long-running, multiply-interrupted run.

**Root cause**: Single-job-per-fold atomicity with no intra-job checkpointing, sized by a cost model extrapolated ~500× from one 3-second anchor and validated only by short-schedule measurements — while the largest multiplier (converged step count, which the brief mandates re-tuning and therefore cannot be known in advance) is the one thing EU-1 does not measure.

**Early warning signs**:
- EU-1 extended with one *full-length* arm-A convergence run (also serves Scenario 1's proxy): converged steps × measured s/step gives a real fold-wall projection before EU-2 is sized.
- Fold-1 realized wall > 2× its projection — folds 2 and 3 are strictly larger, so the overrun compounds.

**Mitigation**:
- Make fold jobs resumable before submitting any: persist each trial's result to NFS as it completes (the parent's exp-003 sweep harness appends results — verify this survives, and add refit checkpoints), so a `--time` kill costs the incomplete trial, not the fold. This converts the scenario from "unit-consuming rerun" to "resubmit and continue".
- Set `--time` at 2–3× the projection (the 24 h cap leaves room; a generous limit on the free partition costs queue priority, not money).
- Before submitting folds 2–3 in parallel, re-project from fold-1 realized numbers, not EU-1 numbers.

**Pivot indicator**: Fold-1 realized wall > 2× projection, or any remaining fold projecting > ~16 h → split that fold into sweep-job + refit/eval-job sub-submissions (accepting the unit-count pressure explicitly, with the seeds-before-folds cut order) rather than betting a bigger atomic job on a shared partition.

**Load-bearing assumption**: That EU-1's short-schedule throughput measurements plus the parent's 3 s anchor project fold-job wall time accurately enough for single-submission-per-fold packing to be safe on a shared, capped partition.

---

### 5. Arm C matched the trajectory but not the treatment — mechanism attribution inverted [Likelihood: Medium | Severity: Medium]

**What happened**: As prior finding H5 predicted, arm B's VALID-selected config was adaptive (`effrank`), so arm C replayed the logged k(t) and update-norm-ratio trajectory in a random parameter-space basis, per the port plan. The comparison ran cleanly and (B−C) came out large and positive; the write-up's strongest sentence — "spectral selection specifically, not generic low-rank projection, is the active ingredient" — rested on it. A reviewer then dismantled the control: B's basis is a *streaming* factorization that continuously rotates to track the gradient covariance, while C's random basis was drawn once and frozen. Confining 14k steps of learning to a fixed random 128-dimensional slice of a 600k-dimensional space is a categorically harsher constraint than projecting onto an adaptively tracking subspace of the same dimension — any slowly-adapting basis, spectral or not, would beat the frozen one. Separately, replaying B's norm-ratio trajectory onto C at a learning rate tuned for C's own dynamics made C effectively a mis-tuned shrinkage arm. C was a straw control; (B−C) measured "adaptive basis vs frozen basis", not "spectral selection vs random selection". The mirror failure was equally available: a subtle RNG-consumption difference between arms breaking the seed pairing, inflating (B−C) variance, and reading mechanism-null when the mechanism was real.

This was hard to anticipate because the parent's C4 — the run's best artifact, explicitly ordered ported — earned its trust in sample-space (B×B), where the eigenbasis is recomputed per batch and "random subspace of the same dimension" is a like-for-like comparison. The p-space port changes which invariants matter, and "norm/k-matched" underdetermines the one that matters most: the basis-adaptation schedule.

**Root cause**: The control's matching invariants were inherited from a different space instead of being re-derived for this one. Whether the random basis should be re-drawn/rotated at the filter's effective basis-rotation rate is a load-bearing design decision the plan never makes.

**Early warning signs**:
- Component #4's local quick test, one addition: compare C's training-loss curve to A's and B's on the synthetic task at matched k. If C's loss is dramatically worse than both, C is constraining, not controlling.
- A logged principal-angle / overlap diagnostic between B's realized basis at step t and its basis at step t−Δ — measuring how fast the "spectral" basis actually rotates, which is the number the control must match.

**Mitigation**:
- Specify the control's invariants before porting: match k(t), the norm ratio, **and the basis-refresh schedule** (re-draw or randomly rotate C's basis at B's measured rotation rate; equivalently, apply a random orthogonal rotation to B's own basis, which matches everything except the spectral identity of the directions — the cleanest version of the control).
- Report C's loss curves and TEST scores in absolute terms, not only as differences — a straw control is visible in its own row of the table.
- Apply the mechanism-honesty clause symmetrically: just as B≈C forbids claiming the spectral mechanism, a B≫C claim requires evidence C was a fair control (the rotation-rate match), or the claim downgrades to "adaptive low-rank projection helps".

**Pivot indicator**: In the #4 quick test or EU-1, C's synthetic loss curve > ~2× worse than A's where B's is not → redesign the control (rotated-basis version) before any fold job; the criteria's fixed-k fallback is documented as weaker matching, not silently used.

**Load-bearing assumption**: That "norm- and k-matched" transfers from sample-space to parameter-space as a sufficient definition of a fair mechanism control.

## Cross-Cutting Themes

1. **The kill criterion is this run's sharpest new edge.** Scenarios 2, 3, and (via truncation) 4 all end with a terminal STOP fired on evidence weaker than the criterion presumes. Its preconditions — gate passed, sweep genuinely ran — verify *execution*, not *evidential sufficiency*: they cannot distinguish an adequately-powered NULL from a power-limited one, or a hypothesis-covering sweep from a feasibility-truncated one. This is the parent's central lesson ("diagnostics validate execution, not interpretation" — which its own pre-mortem named and which materialized anyway) reappearing one level up, attached to a rule that by design forbids the follow-up run that would catch it.
2. **Pre-authorized flexibility + hard terminal rules is the dangerous combination.** Grid capping, seed cuts, and "proceed with the limitation stated" are each individually sane. Each becomes dangerous only when its output flows into the gate or the kill criterion, which do not know the flexibility was exercised. Every mitigation above with teeth is the same move: decide the reinterpretation rule *now*, in `protocol.json`, before the evidence exists — because the parent demonstrates empirically that this team's failure mode is reinterpreting a tripped safeguard after the fact.
3. **One extrapolated cost anchor under everything.** Scenarios 3 and 4 (and the packing component #1) all price off the parent's 3-second runs stretched ~500× through per-step cost, feature width, and a step count that the brief mandates re-tuning and that therefore cannot be known before EU-1 — yet EU-1 as planned measures s/step, not steps-to-convergence. One full-length arm-A run in EU-1 collapses most of this uncertainty and simultaneously serves Scenario 1's early warning.
4. **The gate concentrates roughly half the failure probability and its recovery path is the least-specified part of the plan.** "Fix the baseline" spans six repair dimensions funded by one reserve unit. The difference between a wasted EU-5 and an informative one is whether the fix ladder and stopping rule exist before fold 1 returns its ratio.

## Summary Risk Profile

| Scenario | Likelihood | Severity | Priority | Mitigable? |
|----------|-----------|----------|----------|------------|
| 1. Gate unreachable — second consecutive non-answer | High | High | Critical | Partially (denominator check + fix ladder are free; the intrinsic gap is not fixable in-budget) |
| 2. NULL by power, killed by rule | Medium | High | High | Yes (power-qualified kill wording, pre-freeze; MDE reporting) |
| 3. Feasible rank grid stopped covering the hypothesis | Medium | High | High | Yes (kill-scope floor pre-registered; EU-1 variant timing) |
| 4. Atomic fold jobs died of logistics | Medium | Medium | Medium | Yes (per-trial persistence; 2–3× --time; re-projection from fold 1) |
| 5. Arm C matched the wrong invariants | Medium | Medium | Medium | Yes (basis-rotation matching; absolute C reporting; symmetric honesty clause) |

## Top Recommendations

1. **Amend two clauses before `protocol.json` freezes, at zero compute**: (a) the kill criterion fires on NULL only if the realized per-fold MDE ≤ 0.005 (else "NULL, power-limited — line not declared dead"); (b) the kill verdict is auto-scoped to the realized rank grid, with a pre-registered floor (K_max ≥ 512) below which NULL/HURTS cannot kill the line. Both are wording changes that are legitimate now and illegitimate the moment TEST is touched — and both close the exact reinterpretation channel the parent run fell through.
2. **Buy Scenario 1's bit earlier and cheaper than the plan currently does**: compute the example model's per-era corr on the three prospective TEST blocks locally today (the gate's denominator, free); add one full-schedule arm-A convergence run to EU-1 (proxy gate ratio + real steps-to-convergence, which also de-risks Scenario 4's cost model); and write the baseline-fix ladder with per-rung budgets and a stopping rule before fold 1 is submitted.
3. **Harden the two fragile artifacts before any fold job**: per-trial result persistence + generous `--time` in the sweep harness (a Slurm kill must cost a trial, not a fold), and arm C's matching invariants re-derived for parameter space — match the basis-rotation rate (or use a randomly rotated copy of B's own basis), report C in absolute terms, and make the mechanism-honesty clause symmetric.

## Residual Risk

After all mitigations, roughly half the failure probability remains concentrated where no mitigation reaches: if the MLP-vs-example-model gap is intrinsic rather than protocol-induced, no verdict on the motivating question is reachable under this budget, and the honest deliverable is a baseline-construction negative plus a data-scaling diagnosis — the second consecutive run on this question to end without the answer. That outcome is defined and priced by the criteria, but it should be entered knowingly: P ≈ 0.45 on the gate per the decomposition's own estimate, and the reserve unit cannot reliably buy it back. Beyond the gate, the most likely clean outcome is a NULL whose per-fold MDE (~0.004+ at 110 autocorrelated eras) bounds the effect without resolving its sign — informative as a bound, terminal for the line only if the power-qualified kill wording from Recommendation 1 is adopted; without that amendment, the residual risk is that the run's most probable "success" is a kill verdict the design had substantially predetermined. Neither residual is removable within this compute profile; both are acceptable only because the criteria explicitly price them as findings, and both should appear in the write-up as what they are.
