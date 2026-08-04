# exp-f01 — Component #9: Walk-forward fold arithmetic + protocol.json draft + assertions + gate denominators (LOCAL CPU, zero GPU)

**Component**: #9 in `<run-dir>/decomposition.md` (lambda 0.16, P=0.85) — READ ITS FULL SECTION ("Component 9") plus amendments D5, D7, D10 in the "Round-2 Challenge Addendum" before writing code. Also read the relevant criteria sections of `<run-dir>/success-criteria.md` (walk-forward protocol, A5 gate semantics, A8 assertions).

**Run dir**: `/media/titus/big/researcher-output/2026-08-03-followup-follow-up-spectral-optimizer-redo-pxp-filter-walk-`
**Parent run (READ-ONLY)**: `/media/titus/big/researcher-output/2026-07-31-spectral-optimizer-for-noise-reduction-on-financial-timeseri`

## Task

Local CPU only, no cluster, no GPU. All work and outputs inside `<run-dir>/experiments/exp-f01/` (src/, out/, run.log).

1. **Realized usable-era list**: Derive it the same way the parent did — read parent `experiments/exp-001/src/data_prep.py` and parent `experiments/exp-001/out/protocol.json` for the era-purge/embargo arithmetic (target horizon 20 days at 5-day era spacing → embargo E = 4) and parent-recorded usable-era facts (643 usable eras). Data is at `<run-dir>/data/` (symlinks): `v5.0_train.parquet`, `v5.0_validation.parquet` (read era column only — do NOT load full feature columns; use pyarrow with column selection), `v5.0_validation_example_preds.parquet`.

2. **3-fold expanding-window boundaries**: TRAIN (all usable eras up to T) / embargo 4 / VALID (~96 eras, hp selection only) / embargo 4 / TEST (~110 eras). REFIT trains on train+embargo+valid (everything up to one embargo before TEST). TEST blocks contiguous, non-overlapping, covering the most recent data, coverage >= 0.95 of the intended span. Per fold, hard-assert and PRINT:
   - **THE assertion**: `min(test_eras) - max(refit_train_eras) == E + 1 == 5` (on the usable-era index sequence; if raw era numbering has gaps, adjust the boundary definition, not the assertion, and document the adjustment)
   - disjointness, contiguity, non-overlap, coverage
   - **Coverage assertion (A8)**: every TEST era of every fold is present in `data/v5.0_validation_example_preds.parquet`.

3. **A5 gate denominators**: from parent `experiments/exp-001/out/example_per_era_corr.csv` (652 rows: era, numerai_corr, spearman, n), compute the example model's mean per-era numerai_corr on each prospective TEST block. Apply the 0.010 absolute floor (denominator = max(raw, 0.010); if floored, set a low-signal flag). **D10**: if all three raw denominators are comfortably above the floor, record that; otherwise pre-register the flagged fold's role in the decision rule and gate-failure tally NOW, in the protocol draft.

4. **D7 proxy calibration**: compute the example model's own VALID-slice mean vs TEST-block mean per fold (local pandas on the same CSV) — this offset calibrates EU-1's proxy gate ratio. Record per-fold offset and the calibrated proxy→TEST mapping rule.

5. **D5**: write the deterministic refit-extension rule (loss slope over last 5% of steps > threshold → extend once by a stated factor, logged) and the cross-fold trial-step semantics one-line rule into the protocol draft.

6. **Output** `out/protocol-draft.json`: fold boundaries (era labels AND usable-index), raw-era↔usable-index mapping, embargo E=4, coverage numbers, per-fold gate denominators (raw + floored + flags), D7 offsets, D5 rules, assertion results. This is the DRAFT — the freeze happens later (Wave 3) after EU-1; structure it so later components append (converged step count, realized rank grid, etc. get added then).

## Pass criterion
All asserts pass on all 3 folds; protocol-draft.json written with boundaries, mapping, denominators, D7 offsets; V≈96/S≈110 targets met within coverage >= 0.95.

## Fail criterion
The ==5 condition cannot be satisfied exactly for some fold given era-list gaps → per the decomposition, fix the boundary definition so the semantic condition holds and document; this is arithmetic, it gets fixed, not worked around. Report FAIL only if genuinely unresolvable.

## Constraints
- NEVER modify files outside `<run-dir>`. Parent run dir and `~/pyg/` are READ-ONLY.
- Do not re-download any data. Do not load full-width parquet into memory (2.4GB/3.8GB files; era + id columns only).
- Write `results.md` with PASS/FAIL, the printed assertion output, the fold table (train/valid/test era ranges, counts), denominators table, and D7 offsets. Full stdout to `run.log`.
