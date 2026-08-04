# exp-f04 — Component #3: Per-fold inference power + MDE machinery (LOCAL CPU, zero GPU) [B4/A1/A6]

**Component**: #3 in `<run-dir>/decomposition.md` (lambda 0.48, P=0.70) — READ ITS FULL SECTION ("Component 3") plus amendments D3 and D8 in the "Round-2 Challenge Addendum". Also read the A1 (power-qualified kill) and A6 (inference spec) sections of `<run-dir>/success-criteria.md`.

**Run dir**: `/media/titus/big/researcher-output/2026-08-03-followup-follow-up-spectral-optimizer-redo-pxp-filter-walk-`
**Fold boundaries (input)**: `<run-dir>/experiments/exp-f01/out/protocol-draft.json` — the prospective TEST windows. Read exp-f01/results.md too.
**Parent per-era vectors (READ-ONLY inputs)**:
- `/media/titus/big/researcher-output/2026-07-31-spectral-optimizer-for-noise-reduction-on-financial-timeseri/experiments/exp-001/out/example_per_era_corr.csv` (652 eras: era, numerai_corr, spearman, n)
- `/media/titus/big/researcher-output/2026-07-31-spectral-optimizer-for-noise-reduction-on-financial-timeseri/experiments/exp-004/out/per_era_numerai_corr.csv` and `per_era_spearman.csv` (parent per-era arm series — use for realistic arm-level noise/ACF texture)
- Parent audit's independent bootstrap: `/media/titus/big/researcher-output/2026-07-31-spectral-optimizer-for-noise-reduction-on-financial-timeseri/audit/rerun-exp-004/` (reference implementation to adapt, not to trust blindly)

## Task

Local CPU only. All work inside `<run-dir>/experiments/exp-f04/` (src/, out/, run.log).

Build and validate the inference machinery that the verdict will use, and measure the design's realized power BEFORE any TEST touch:

1. **Hierarchical era-block × seed bootstrap (A6)**: moving-block bootstrap over era blocks with seed-level resampling nested inside; block length from each fold window's own lag-1 ACF (recomputed per window — never inherited); bootstrap RNG fixed and stability checked across >= 2 RNG seeds. This code is a deliverable — the verdict analysis will import it.

2. **Power simulation**: restrict the parent per-era vectors to each prospective TEST window from protocol-draft.json; recompute lag-1 ACF and block length per window; simulate paired (B−A) per-era difference series calibrated to the parent's observed per-era difference noise (era-correlated, with a seed-level variance component estimated from the parent's 3-seed arm series) under (i) null and (ii) ±0.005 injected true effect, at BOTH 3 and 5 paired seeds; compute P(95% CI excludes 0) per scenario per fold.

3. **Per-fold MDE table (A1/D3)**: the standing table — per fold, the 95% CI half-width of the seed-averaged fold-mean difference; report as "CI half-width (≈50%-power MDE)" (D3's honest naming). Derive the 0.005 anchor in one sentence (relative to the gated baseline's expected magnitude ~0.014, i.e. 0.005 ≈ 35% of the gate-level baseline score) for protocol.json.

4. **D3 CI coverage check**: empirical coverage of the 95% CI at 3 and 5 seeds under the simulated data-generating process; if 3-seed coverage is sub-nominal, flag that 5 seeds is a VALIDITY lever (not just power) for the packing decision in Wave 3.

5. **D8 pooled secondary estimand**: implement and pre-register the pooled cross-fold estimand (concatenated era-level paired (B−A) under one hierarchical MBB); state in output that it cannot overturn the per-fold rule.

**B4 pass threshold (wired, not narrative)**: PASS iff P(CI excludes 0 | true effect ±0.005) >= 0.6 on >= 2 of 3 prospective folds AND median MDE <= ~0.005 AND conclusions stable across 2 bootstrap RNG seeds.

On FAIL: this is NOT a proceed-with-caveat. Report FAIL clearly; the A1 power-qualified kill wording becomes load-bearing (it goes into protocol.json either way); the run's most probable clean outcome is then a power-limited NULL and 5 seeds becomes mandatory-if-packable. Also emit the D8 decisiveness-checkpoint inputs (your per-fold power numbers) so the freeze step can act on them.

## Outputs
- `out/power-sim.json`: per fold — ACF, block length, MDE at 3 and 5 seeds, P(detect ±0.005) at 3 and 5 seeds, coverage at 3 and 5 seeds, RNG-stability check.
- `src/`: the hierarchical bootstrap module (deliverable) + sim code.
- `results.md`: PASS/FAIL against the B4 threshold, the MDE table, the 0.005 derivation sentence, and explicit implications (5-seed validity flag, D8 checkpoint inputs).

## Constraints
- NEVER modify files outside `<run-dir>`. Parent run dir READ-ONLY.
- LOCAL CPU only; no GPU, no cluster. Target < 30 min runtime (vectorize the bootstrap).
- No venvs; system python3 (numpy/pandas/scipy available).
