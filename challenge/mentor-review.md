# Mentor Review (Round 2)

> Saved by the orchestrator from the mentor-review agent's returned text
> (the agent returns text and writes no files). Round-1 review is at
> challenge/round1-mentor-review.md.

## Overall Assessment

This is now a well-designed plan: the round-2 documents genuinely integrate the round-1 amendments rather than gesturing at them, the kill criterion can no longer fire on power, scope, or tuning-budget artifacts, and the highest-uncertainty bit (the baseline gate, P=0.55) is bought as early and cheaply as the design allows. Its biggest strength is that every failure shape now converts to a pre-registered, interpretable qualified finding. Its biggest weakness is sheer specification mass: an 11-item freeze checklist, a four-condition kill rule, a five-rung banded fix ladder, and a three-candidate control design all have to be executed faithfully by an autonomous agent inside 5 experiment units, and the plan's own honest arithmetic (P ≈ 0.06 unqualified verdict) says the modal outcome is a qualified one.

## What a Senior Researcher Would Do Differently

**Round-1 defect verification first** — I checked each claimed fix against the actual text, not the change-log claims:

- *Unresolvable 12-trial sweep (my round-1 headline)*: fixed. The A4 staged allocation (7–8 stage-1 trials, one per rank/adaptive point at transferred LR; 4–5 stage-2 refinements) over a documented pruned space is in the criteria verbatim, with step count and refit stopping rule explicitly moved outside the 12 trials. This is a resolvable design.
- *Missing under-exploration downgrade (F12)*: fixed and operationalized — three concrete signatures (grid boundary; non-monotone with range < ~2× across-seed sd; ≥ 4× LR shift) with a frozen HURTS→"no evidence under affordable budget" downgrade. This is better than the parent's original guard because the signatures are computable from artifacts.
- *Kill criterion unqualified*: fixed — the four-condition rule (execution + A1 MDE ≤ 0.005 + A2 K_max ≥ 512 scope + A3 signatures) is in both the criteria and the decomposition's verdict node.
- *Arm C semantics (assumption-analysis critical #1)*: fixed in design — invariants re-derived (k(t), norm ratio, basis-rotation rate), the round-1 fixed-subspace fallback explicitly retired as unacceptable, and the required pre-port CPU sim is component #2's quick test.
- *Gate denominator as unexamined random variable*: fixed — per-fold denominators computed locally before freeze, 0.010 floor, low-signal flag, coverage assertion.
- *Atomic fold jobs (pre-mortem scenario 4)*: fixed — component #7 with a kill-test, which correctly treats the parent harness's crash behaviour as unknown rather than assumed.

Remaining things I would do differently, none load-bearing:

1. **Pre-designate arm-C candidate (b) as co-primary.** The decomposition itself makes the argument that kills candidate (a): a Haar rotation of B's realized basis is a uniformly random k-subspace at each instant, so it should capture ~k/p ≈ 0.3% energy — the exact pathology A7 exists to avoid. The sim will almost certainly route to (b) (random rotation *within the span* of the tracked covariance history). Run the sim on all three, but write (b) into the plan as the expected design now, so a predictable sim outcome doesn't cost a decision cycle mid-run.
2. **Derive, don't just assert, the 0.005 MDE threshold.** It is a sensible number (≈ the parent's B×B effect magnitude, ≈ 21% of the example-model yardstick), but the criteria never say why 0.005 is the decision-relevant effect size. One sentence in `protocol.json` anchoring it ("smallest effect that would change practice, ≈ parent effect / ≈ X% of yardstick") makes the A1 qualification auditable rather than arbitrary.
3. **Watch EU-1's wall clock.** EU-1 now carries the shard build, two-architecture rank-grid timing, eigh-variant timing, stability checks, identity re-assert, *and* the full-length convergence run — the job that measures the unknown cost is itself sized assuming that cost is small. Pre-authorize a plateau-detection stop on the convergence run and a fallback to a normal (non-debug) submission if it threatens the 2 h `--qos=debug` window, so the pre-flight doesn't become the first casualty of the unknown it exists to measure.

## What Hasn't Been Examined Yet

- **Specification-fidelity risk.** The criteria document is now the most complex artifact of the run, and the Step 10 audit judges against it clause by clause. Every clause is a potential audit finding if the executor drops it under time pressure. The protocol-freeze checklist is the right mitigation — treat `protocol.json` as the single operational source of truth and have each fold job print which checklist items it is satisfying, so fidelity is checkable from logs rather than reconstructed.
- **Asymmetric qualification.** A1 qualifies NULL and A3 qualifies HURTS, but an under-explored sweep can also manufacture a NULL (VALID selects a near-identity config, B ≈ A by construction). The criteria do handle this — the distance-from-identity diagnostic and the mandate to say *which* NULL occurred — but note it is handled narratively, not by a downgrade rule. Acceptable, since both NULLs are legitimate; just make sure the write-up actually uses the diagnostic.
- **The decay probe eats stage 2.** The decay=0.999 probe consumes one of the 4–5 stage-2 trials, leaving 3–4 for LR/alpha refinement around the winner. That is thin. Defensible under the declared limitation, but if stage 1 shows a sharp LR sensitivity, the probe should be the trial that gets dropped.
- **P ≈ 0.23 for any informative outcome is worth staring at.** Roughly three-quarters of the probability mass is engineering failure (#1, #5–#9), not scientific outcome. The number is honest and the components are individually mitigated; I flag it only so nobody is surprised when the run's main battle turns out to be logistics.

## Simpler Alternatives

The obvious simplification — drop arm C and folds 2–3, run a single-fold B-vs-A comparison — would answer a weaker question (no mechanism attribution, no consistency evidence) and was effectively what the parent did wrong. The gate-first fold-1 structure already captures most of the simplification benefit: if the gate fails, folds 2–3 and arm C are never spent. The staged sweep is the simple version of the sweep. I see no materially simpler path to the same verdict; the complexity here is load-bearing, not decorative.

## Construct Validity / Information Value

Sound. The headline result is not statable on paper: the p×p `SpectralGradientFilter` has never been run on this data, the parent's target-independence theorem is correctly scoped to the row-normalized B×B variant and does not transfer, and the literature genuinely predicts both directions (Coherent Gradients/GAF/H5 for; Feldman long-tail and Adam's intrinsic heavy-tail robustness against). The object under test is the real filter via its documented API, verified present, with the alpha=0 identity and planted-subspace checks guarding against testing a broken integration. The plan answers the motivating question directly — a committed WINS/HURTS/NULL under a pre-registered decision rule, with the qualified forms pre-registered as outcomes rather than escape hatches — and the baseline gate ensures the comparison is against a baseline that demonstrably works, which is precisely the construct repair this follow-up exists to make. Arm C's redesign closes the one route by which the mechanism claim could have been vacuous by construction. No known-outcome flaw.

## Key Recommendations

1. Pre-designate arm-C candidate (b) (rotation within the tracked-history span) as the expected primary design, with the sim as confirmation rather than open-ended selection — the decomposition's own energy argument already predicts (a) fails.
2. Anchor the 0.005 MDE threshold in `protocol.json` with a one-sentence derivation tying it to a decision-relevant effect size, so the A1 power qualification is auditable.
3. Protect EU-1: plateau-stop the convergence run and pre-authorize escalation from `--qos=debug` to a normal submission if the pre-flight outgrows the 2 h window; and treat `protocol.json` as the single operational spec each fold job logs against, to contain specification-fidelity risk.

## Verdict

MINOR_REVISIONS

Every round-1 defect I raised is fixed in the documents themselves, not merely claimed — the staged sweep, the reinstated and operationalized under-exploration guard, the four-condition kill criterion, the re-derived arm C, and the gate-denominator semantics are all present in binding form. The amendments added real complexity, but the decomposition sizes it (packing arithmetic, cut order, resumability, reserve) and the plan's honest P estimates show the cost was paid in probability-of-unqualified-verdict, not in validity. Nothing remaining would invalidate the result: the three recommendations are in-flight improvements (a predictable sim outcome, a one-sentence derivation, a job-sizing precaution), none of which is a missing control, an unfair baseline, or an underpowered comparison. Ready to run with those noted.
