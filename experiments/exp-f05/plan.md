# exp-f05 — Component #2: Arm C re-derived for parameter space — B3 design sim (LOCAL CPU, zero GPU) [B3/A7/D2/D11]

**Component**: #2 in `<run-dir>/decomposition.md` (lambda 0.57, P=0.65) — READ ITS FULL SECTION ("Component 2") plus amendments **D2** and **D11** in the "Round-2 Challenge Addendum". Also read the A7 arm-C section and the symmetric mechanism-honesty clause of `<run-dir>/success-criteria.md`.

**Run dir**: `/media/titus/big/researcher-output/2026-08-03-followup-follow-up-spectral-optimizer-redo-pxp-filter-walk-`
**Filter**: use the run's canonical copy at `<run-dir>/experiments/exp-f02/src/spectral_filter.py` if it exists; otherwise copy `/home/titus/pyg/optimizers/spectral_filter.py` into your own src/ (READ-ONLY original). exp-f02 may be running concurrently — do NOT write into exp-f02/.

## Purpose

This sim **determines the final arm-C (mechanism control) design, in writing, before any cluster control code is ported**. Arm C must separate "spectral selection matters" from "any low-rank projection does this". The round-1 design (fixed random k-subspace, norm-matched) is RETIRED: at p ≈ 600k a fixed random k-subspace captures ~k/p ≈ 0.3% of gradient energy, so norm-matching means 30–250× amplification — noise injection, not a control.

**A7 matching invariants**: k(t), the update-norm-ratio trajectory, AND the basis-rotation rate (principal angles between the realized basis at t and t−Δ).

## Task

Local CPU only, ~20–30 min sim runtime target. All work inside `<run-dir>/experiments/exp-f05/` (src/, out/, run.log).

Build a synthetic gradient stream at p = 600k with planted low-rank + noise structure (or harvest gradients from a small synthetic Numerai-shaped MLP task with the filter running — either is acceptable; document the choice). Run arm B (the real `SpectralGradientFilter`, hard top-k, `normalize="none"`) to obtain realized k(t), kept-norm trajectory, and basis-rotation rate. Then evaluate the three candidate C designs at k ∈ {8, 512, 2048}:

- **(a)** random orthogonal rotation of B's own realized basis — C's basis = R·U_B(t), R a fixed random orthogonal transform (e.g. random signed permutation composed with a subsampled randomized Hadamard transform, O(p log p), exactly orthogonal).
- **(b) [D2: expected primary]** random rotation **within the span** of B's tracked covariance factorization / recent-gradient history, with a **strictly larger history span** (rotate within an r > k ambient space, truncate to k) — selects non-spectrally inside the space where gradient energy actually lives.
- **(c)** re-drawn random basis at B's measured rotation rate — weakest candidate.

For each candidate at each k, measure:
1. **Captured-energy fraction** of the true gradient stream.
2. **Realized norm-amplification factor** needed to match the logged arm-B kept-norm trajectory.
3. **Principal-angle rotation-rate match** to arm B.
4. **Descent check**: does the norm-matched control still descend on the synthetic task, with a loss curve within ~2× of a plain-AdamW arm A? (The straw-control pivot indicator.)

**D2 distinctness criteria (the control must differ from B, not clone it)**:
5. **Principal-angle floor** between C's realized basis and B's (C is not B in disguise).
6. **Update-cosine ceiling** between C's and B's filtered updates.
7. **Positive-control-for-the-control**: an acceptable C must FAIL the planted-subspace task where B succeeds (build the same planted-dominant-direction task as exp-f02's sub-check (c); C must NOT recover the planted direction, cosine well below B's).

**D11**: specify (in the design decision) the post-hoc invariant-match report the fold jobs will emit — realized principal-angle and norm-ratio between B and C over the full schedule, per fold — as the fair-control evidence the symmetric honesty clause consumes.

## Output

- `out/arm-c-sim.json`: per candidate × k — captured energy, amplification, rotation-rate match, descent ratio, principal-angle floor, update-cosine, planted-task result.
- **`out/arm-c-design.md`: the binding design decision** — which candidate is arm C (D2 expects (b) unless the sim says otherwise), its exact construction, its invariant-match evidence plan (D11), and the pre-registered **empty-middle fallback** (if NO candidate sits between the noise degeneracy and the clone degeneracy: drop C, scope the run to (B−A), redirect C's refit-seed budget toward 5 seeds).
- `results.md`: PASS/FAIL, the candidate table, the design decision summary.
- `src/`: the sim code + the chosen candidate's implementation sketch (the fold jobs will port it).

## Pass criterion
At least one candidate matches all three A7 invariants, passes the D2 distinctness criteria (including failing the planted task where B succeeds), and descends with a loss curve within ~2× of arm A; design decision documented in out/arm-c-design.md.

## Fail criterion
Every candidate either requires noise-injection-scale amplification with no descent, or cannot match the invariants / distinctness within ~2h of surgery. On FAIL: invoke the D2 empty-middle fallback in writing (drop C, scope to (B−A), redirect seed budget) — this is a pre-registered outcome, not a run-stopper; the minimum viable verdict survives without C.

## Constraints
- NEVER modify files outside `<run-dir>`. Parent run dir and `~/pyg/` are READ-ONLY.
- LOCAL CPU only; no GPU, no cluster. Vectorize; avoid materializing p×p anything (p=600k — use implicit/factored operators).
- Do NOT create Python virtual environments. Use the system Python.
- Write `results.md` with PASS/FAIL, the full candidate × k table, and the design decision. Full stdout to `run.log`.
