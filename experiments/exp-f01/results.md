# Experiment f01 Results

## Component Tested
Component #9: Walk-forward fold arithmetic + protocol.json draft + assertions +
A5 gate denominators (+ amendments D5, D7, D10) — LOCAL CPU, zero GPU.

## Verdict: PASS

## Setup
- Environment: system Python 3 (local desktop CPU), numpy, pandas, pyarrow.
  No packages installed; no venv; no cluster or GPU touched.
- Duration: 1.9 s script runtime (single run, no retries needed).
- Resources used: local CPU only; peak data read was narrow column slices
  (`era`/`data_type`/`target`/`id`) of the parquets — never full feature width.
- Code: `experiments/exp-f01/src/fold_arithmetic.py`; full stdout in
  `experiments/exp-f01/run.log`; outputs in `experiments/exp-f01/out/`
  (`protocol-draft.json` 9.1 KB, `fold_boundaries.csv`).

## What Was Tested
The realized usable-era list was derived exactly the way the parent run did
(parent `exp-001/src/data_prep.py`): era usable iff ≥ 95% of rows have a
non-null target; embargo E = ceil(20-day horizon / 5-day era spacing) = 4.
Result: 574 train-parquet eras (0001–0574) + 651 usable validation eras
(0575–1225, all 651 usable) = **1225 usable eras with zero raw-numbering
gaps**, cross-checked against the parent-recorded facts (574 / 651 / E=4 all
match). Because the raw era numbering is gap-free, the raw-era ↔ usable-index
mapping is the identity shift `usable_index = int(era) − 1` (recorded in the
draft with spot checks); no boundary-definition adjustment was needed and THE
assertion holds identically on both index and raw-era arithmetic.

On this list the 3-fold expanding-window walk-forward boundaries were computed
(TRAIN / embargo 4 / VALID 96 / embargo 4 / TEST 110; REFIT = train + embargo
+ valid), all pre-registered assertions were hard-asserted and printed per
fold, the A8 example-preds coverage assertion was checked by an id-join
against `v5.0_validation_example_preds.parquet` (which has no era column —
membership resolved via validation-parquet ids), the A5 gate denominators
were computed from the parent's `example_per_era_corr.csv` with the 0.010
floor (D10), the D7 VALID-vs-TEST proxy-calibration offsets were computed
per fold, and the D5 refit-extension and cross-fold trial-step rules were
written into `out/protocol-draft.json` (structured for Wave-3 appends before
freeze).

## Results

### Raw Output
```
usable-era list: 1225 eras (0001..1225), raw-numbering gaps: NONE

ASSERT [PASS] fold 1 THE-assertion (usable-index): min(test) - max(refit_train) == E+1 == 5  min(test_idx)=895, max(refit_idx)=890, gap=5
ASSERT [PASS] fold 1 THE-assertion (raw eras):  min(test)=0896, max(refit)=0891, gap=5
ASSERT [PASS] fold 2 THE-assertion (usable-index):  min(test_idx)=1005, max(refit_idx)=1000, gap=5
ASSERT [PASS] fold 2 THE-assertion (raw eras):  min(test)=1006, max(refit)=1001, gap=5
ASSERT [PASS] fold 3 THE-assertion (usable-index):  min(test_idx)=1115, max(refit_idx)=1110, gap=5
ASSERT [PASS] fold 3 THE-assertion (raw eras):  min(test)=1116, max(refit)=1111, gap=5
ASSERT [PASS] TEST blocks non-overlapping across folds
ASSERT [PASS] TEST blocks jointly contiguous
ASSERT [PASS] TEST union covers the most recent data
ASSERT [PASS] coverage of intended most-recent span >= 0.95  coverage=1.0000 (330/330)
ASSERT [PASS] fold 1 A8: all 110 TEST eras in example preds  missing=[]
ASSERT [PASS] fold 2 A8: all 110 TEST eras in example preds  missing=[]
ASSERT [PASS] fold 3 A8: all 110 TEST eras in example preds  missing=[]
SUMMARY: ALL ASSERTIONS PASSED
```
(Plus per-fold disjointness, contiguity, block-size, embargo-width, and
expanding-window assertions — 40 assertions total, all PASS; full list in
`run.log`.)

### Fold table
| Fold | TRAIN (n) | VALID (n) | TEST (n) | REFIT-TRAIN (n) |
|------|-----------|-----------|----------|-----------------|
| 1 | 0001..0791 (791) | 0796..0891 (96) | 0896..1005 (110) | 0001..0891 (891) |
| 2 | 0001..0901 (901) | 0906..1001 (96) | 1006..1115 (110) | 0001..1001 (1001) |
| 3 | 0001..1011 (1011) | 1016..1111 (96) | 1116..1225 (110) | 0001..1111 (1111) |

### A5 gate denominators (D10) and D7 offsets
| Fold | Example TEST mean (raw) | Floored denom | Low-signal flag | Gate = 0.60× denom | Example VALID mean | D7 offset (T−V) | D7 ratio T/V |
|------|------------------------|---------------|-----------------|--------------------|--------------------|-----------------|--------------|
| 1 | +0.03957 | 0.03957 | No | +0.02374 | +0.03394 | +0.00563 | 1.166 |
| 2 | +0.02756 | 0.02756 | No | +0.01654 | +0.04064 | −0.01307 | 0.678 |
| 3 | +0.01639 | 0.01639 | No | +0.00983 | +0.02867 | −0.01228 | 0.572 |

**D10 outcome**: all three raw denominators are above the 0.010 floor
(margins 3.96×, 2.76×, 1.64×); no fold is low-signal-flagged; no special
decision-rule role needed pre-registration. The all-above-floor fact is
recorded in the draft.

### Metrics
| Metric | Target | Actual | Pass? |
|--------|--------|--------|-------|
| THE assertion (==5) fires on every fold | 3/3 folds | 3/3 (gap=5 on index AND raw eras) | Y |
| All other assertions (disjoint/contiguous/non-overlap/embargo/A8) | all pass | 40/40 PASS | Y |
| VALID size | ~96 | 96 exactly, all folds | Y |
| TEST size | ~110 | 110 exactly, all folds | Y |
| TEST coverage of intended most-recent span | ≥ 0.95 | 1.0000 (330/330) | Y |
| A8: TEST eras in example preds | all 330 | all 330 (min per-era pred rows 6007) | Y |
| protocol-draft.json written with boundaries, mapping, denominators, D7, D5 | yes | yes (9.1 KB, valid JSON, append-ready) | Y |

### Analysis
The fold arithmetic is clean: the raw era numbering across the train and
validation parquets is perfectly contiguous (0001–1225) and every validation
era clears the 95% target-coverage bar, so the usable-index and raw-era
formulations of THE assertion coincide and the fail-mode the plan anticipated
(era-list gaps breaking the exact ==5 condition) did not arise. The V=96 /
S=110 targets are met exactly with coverage 1.0 — the three TEST blocks tile
the most recent 330 eras (0896–1225) with no slack, and each fold's refit
training data ends exactly 5 eras (one embargo + 1) before its TEST block
begins.

The gate denominators carry real information for downstream sizing. The
example model's edge decays monotonically into the most recent data:
+0.03957 → +0.02756 → +0.01639 across the three TEST blocks. All clear the
0.010 floor, so the gate is well-posed on every fold, but fold 3's margin is
only 1.64× — the pre-mortem's "denominator lands near the floor" scenario is
closest on the most recent fold. Correspondingly, the absolute score arm A
needs on fold 3 (+0.00983) is much lower than on fold 1 (+0.02374): the gate
is easiest exactly where the yardstick is weakest, which is what the
low-signal-flag machinery exists to monitor even though it did not trigger.

The D7 calibration shows why the parent-style uncalibrated proxy would have
been misleading: the example model's own TEST/VALID ratio is 1.166 on fold 1
but 0.678 and 0.572 on folds 2–3, i.e. a VALID-slice proxy overstates
attainable TEST performance by ~1.5–1.75× on the later folds. The
pre-registered mapping rule (carry arm A's VALID mean to TEST by the fold's
example T/V ratio, divide by the floored denominator; sizing/scheduling only)
is now in the draft, alongside the D5 deterministic refit-extension rule
(extend once by 1.5× iff the final 5% of steps contributes > 2% of the total
EMA-loss drop) and the one-line cross-fold trial-step rule (steps scale with
rows at fixed batch from the EU-1 converged count N*).

## Unexpected Observations
- The example-preds parquet has no `era` column (only `prediction`, `id`), so
  the A8 assertion was implemented as an id-join against the validation
  parquet — the join found 656 eras with predictions: the 651 validation eras
  plus 5 `data_type="test"` eras (1226–1230, unresolved targets). All fold
  TEST eras are ≤ 1225, so this does not affect the assertion, but any later
  code that reads example preds should filter `data_type == "validation"`.
- The parent plan described the CSV as "652 rows"; that count includes the
  header — it has 651 data rows, one per usable validation era, exactly
  covering every VALID and TEST era of every fold.
- The example model's per-block mean decays ~2.4× from fold-1 TEST to fold-3
  TEST (+0.0396 → +0.0164). The parent's single verdict-block figure
  (+0.0235 over 0971–1225) masks this drift; per-fold denominators were the
  right call.
- Fold-2 and fold-3 D7 offsets are negative and large relative to the
  denominators (−0.013 vs 0.016–0.028): regime drift between VALID and TEST
  slices is first-order here, not a correction term.

## Implications
What this tells us: the criterion was met — all 40 assertions (including THE
==5 assertion and the A8 coverage assertion on all 3 folds) passed, exact
V=96/S=110 blocks tile the most recent 330 eras at coverage 1.0, all three
gate denominators are above the floor with no flags, and
`out/protocol-draft.json` contains the boundaries, the raw-era↔usable-index
mapping, the floored denominators, the D7 offsets + mapping rule, the D5
rules, and an explicit list of Wave-3 append slots. Next steps: Component #3
(power sim) can consume the prospective boundaries immediately; EU-1 (#8)
uses the fold-1 boundaries for shard sizing and the D7 rule for its proxy
gate ratio; Wave 3 appends the EU-1-derived quantities (converged step count,
realized rank grid, arm-C design, A1/A2/A3 wording) and freezes the draft as
`protocol.json` before any TEST touch.
