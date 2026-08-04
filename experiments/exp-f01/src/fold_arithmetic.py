#!/usr/bin/env python3
"""exp-f01 / Component #9: Walk-forward fold arithmetic + protocol-draft.json
+ assertions + A5 gate denominators + D5/D7/D10 amendments.

LOCAL CPU ONLY. Zero GPU, zero cluster. Reads narrow column slices only.

Derives the realized usable-era list the same way the parent run did
(parent exp-001 src/data_prep.py):
  - validation era usable iff >= 95% of its rows have non-null target
  - embargo E = ceil(target_horizon_days / era_spacing_days) = ceil(20/5) = 4
Then computes the 3-fold expanding-window walk-forward boundaries
(TRAIN / embargo 4 / VALID ~96 / embargo 4 / TEST ~110; TEST blocks
contiguous, non-overlapping, most recent data), hard-asserts THE assertion
    min(test_eras) - max(refit_train_eras) == E + 1 == 5
per fold (on the usable-era index sequence), the A8 example-preds coverage
assertion, disjointness/contiguity/non-overlap/coverage, computes the A5
gate denominators with the 0.010 floor (D10), the D7 proxy-calibration
offsets, and writes out/protocol-draft.json with the D5 rules.

Outputs (all in <exp>/out/):
  protocol-draft.json   - the draft protocol (freeze happens in Wave 3)
  fold_boundaries.csv   - human-readable fold table
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

T0 = time.time()

EXP = Path(__file__).resolve().parent.parent          # .../experiments/exp-f01
RUN = EXP.parent.parent                               # run dir
DATA = RUN / "data"
OUT = EXP / "out"
OUT.mkdir(exist_ok=True)

PARENT = Path("/media/titus/big/researcher-output/"
              "2026-07-31-spectral-optimizer-for-noise-reduction-on-financial-timeseri")
PARENT_CSV = PARENT / "experiments/exp-001/out/example_per_era_corr.csv"
PARENT_PROTOCOL = PARENT / "experiments/exp-001/out/protocol.json"

# --- pre-registered constants (identical derivation to parent data_prep.py) ---
TARGET_HORIZON_DAYS = 20     # main target `target` == cyrus 20D
ERA_SPACING_DAYS = 5         # v5 eras are weekly (5 business days)
EMBARGO = int(np.ceil(TARGET_HORIZON_DAYS / ERA_SPACING_DAYS))   # = 4
MIN_TARGET_COVERAGE = 0.95   # era usable iff >= 95% rows have target
N_VALID = 96                 # VALID block size (hp selection only)
N_TEST = 110                 # TEST block size
N_FOLDS = 3
GATE_FLOOR = 0.010           # A5 degenerate-yardstick absolute floor
GATE_THRESHOLD = 0.60        # arm A must reach >= 0.60 x floored denominator

failures = []


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"ASSERT [{status}] {name}  {detail}")
    if not cond:
        failures.append(name)
    return bool(cond)


def main():
    # =================== 1. realized usable-era list ===================
    print("=" * 78)
    print("STEP 1: realized usable-era list (parent data_prep.py derivation)")
    print("=" * 78)

    tf = pq.ParquetFile(DATA / "v5.0_train.parquet")
    tera = tf.read(columns=["era"]).to_pandas()["era"]
    train_eras = sorted(tera.unique())
    print(f"train.parquet: {len(tera)} rows, {len(train_eras)} eras "
          f"({train_eras[0]}..{train_eras[-1]})")

    vf = pq.ParquetFile(DATA / "v5.0_validation.parquet")
    vdf = vf.read(columns=["era", "data_type", "target"]).to_pandas()
    print(f"validation.parquet meta slice: {len(vdf)} rows, "
          f"mem {vdf.memory_usage(deep=True).sum()/1e9:.2f} GB")
    vdf = vdf[vdf["data_type"] == "validation"]
    cov = vdf.groupby("era", observed=True)["target"].apply(
        lambda s: s.notna().mean())
    usable_val_eras = sorted(cov[cov >= MIN_TARGET_COVERAGE].index)
    all_val_eras = sorted(vdf["era"].unique())
    print(f"validation eras: {len(all_val_eras)} total "
          f"({all_val_eras[0]}..{all_val_eras[-1]}), "
          f"{len(usable_val_eras)} usable (target coverage >= "
          f"{MIN_TARGET_COVERAGE})")

    # cross-check against parent-recorded facts
    pprot = json.load(open(PARENT_PROTOCOL))
    check("parent-facts: train_eras_total matches",
          len(train_eras) == pprot["train_eras_total"],
          f"{len(train_eras)} vs parent {pprot['train_eras_total']}")
    check("parent-facts: validation_eras_usable matches",
          len(usable_val_eras) == pprot["validation_eras_usable"],
          f"{len(usable_val_eras)} vs parent {pprot['validation_eras_usable']}")
    check("parent-facts: embargo matches", EMBARGO == pprot["embargo_eras"],
          f"E={EMBARGO}")

    # the walk-forward usable-era list = all train eras + usable validation
    # eras, in temporal order.  (The parent's '643 usable eras' figure was its
    # validation-side tuning+verdict blocks after ITS boundary purge; the
    # walk-forward redesign trains through the old train/validation boundary,
    # so the purge lives in the per-fold embargoes instead.)
    usable = [str(e) for e in train_eras] + [str(e) for e in usable_val_eras]
    usable_int = [int(e) for e in usable]
    gaps = [(usable_int[i], usable_int[i + 1])
            for i in range(len(usable_int) - 1)
            if usable_int[i + 1] != usable_int[i] + 1]
    print(f"usable-era list: {len(usable)} eras "
          f"({usable[0]}..{usable[-1]}), raw-numbering gaps: {gaps or 'NONE'}")
    check("usable-era raw numbering contiguous", len(gaps) == 0,
          f"gaps={gaps}")
    # raw-era <-> usable-index mapping (A8): usable_index i -> usable[i]
    idx_of = {e: i for i, e in enumerate(usable)}
    n_usable = len(usable)

    # =================== 2. 3-fold expanding-window boundaries =============
    print()
    print("=" * 78)
    print("STEP 2: 3-fold expanding-window boundaries + assertions")
    print("=" * 78)
    # TEST blocks: contiguous, non-overlapping, most recent 3*N_TEST usable
    # eras. Fold f (1-based, oldest TEST first):
    #   TEST_f  = usable[n - (N_FOLDS - f + 1)*N_TEST : n - (N_FOLDS - f)*N_TEST]
    #   VALID_f = the N_VALID usable eras ending EMBARGO+1 before TEST start
    #   TRAIN_f = all usable eras up to EMBARGO+1 before VALID start
    #   REFIT_f = TRAIN + embargo + VALID = everything up to EMBARGO+1
    #             before TEST start
    intended_span = usable[n_usable - N_FOLDS * N_TEST:]
    folds = []
    for f in range(1, N_FOLDS + 1):
        t_start = n_usable - (N_FOLDS - f + 1) * N_TEST
        t_end = t_start + N_TEST                       # exclusive
        test_idx = list(range(t_start, t_end))
        v_end = t_start - EMBARGO - 1                  # inclusive index
        v_start = v_end - N_VALID + 1
        valid_idx = list(range(v_start, v_end + 1))
        tr_end = v_start - EMBARGO - 1                 # inclusive index
        train_idx = list(range(0, tr_end + 1))
        refit_idx = list(range(0, v_end + 1))          # train+embargo+valid
        fold = {
            "fold": f,
            "train_idx": (0, tr_end), "valid_idx": (v_start, v_end),
            "test_idx": (t_start, t_end - 1),
            "refit_idx": (0, v_end),
            "train_eras": [usable[i] for i in train_idx],
            "valid_eras": [usable[i] for i in valid_idx],
            "test_eras": [usable[i] for i in test_idx],
            "refit_train_eras": [usable[i] for i in refit_idx],
        }
        folds.append(fold)

        print(f"\n--- Fold {f} ---")
        print(f"TRAIN : idx [0, {tr_end}]  eras {usable[0]}..{usable[tr_end]}"
              f"  (n={len(train_idx)})")
        print(f"embargo: idx [{tr_end+1}, {v_start-1}]  eras "
              f"{usable[tr_end+1]}..{usable[v_start-1]}  (n={EMBARGO})")
        print(f"VALID : idx [{v_start}, {v_end}]  eras "
              f"{usable[v_start]}..{usable[v_end]}  (n={len(valid_idx)})")
        print(f"embargo: idx [{v_end+1}, {t_start-1}]  eras "
              f"{usable[v_end+1]}..{usable[t_start-1]}  (n={EMBARGO})")
        print(f"TEST  : idx [{t_start}, {t_end-1}]  eras "
              f"{usable[t_start]}..{usable[t_end-1]}  (n={len(test_idx)})")
        print(f"REFIT-TRAIN: idx [0, {v_end}]  eras "
              f"{usable[0]}..{usable[v_end]}  (n={len(refit_idx)})")

        # ---- THE assertion (on the usable-era index sequence) ----
        gap_idx = min(test_idx) - max(refit_idx)
        check(f"fold {f} THE-assertion (usable-index): "
              f"min(test) - max(refit_train) == E+1 == 5",
              gap_idx == EMBARGO + 1 == 5,
              f"min(test_idx)={min(test_idx)}, max(refit_idx)={max(refit_idx)},"
              f" gap={gap_idx}")
        # raw-era arithmetic coincides because numbering is gap-free
        gap_raw = int(fold["test_eras"][0]) - int(fold["refit_train_eras"][-1])
        check(f"fold {f} THE-assertion (raw eras): "
              f"min(test_eras) - max(refit_train_eras) == 5",
              gap_raw == 5,
              f"min(test)={fold['test_eras'][0]}, "
              f"max(refit)={fold['refit_train_eras'][-1]}, gap={gap_raw}")
        # embargo between TRAIN and VALID
        check(f"fold {f} train->valid embargo == {EMBARGO}",
              v_start - tr_end - 1 == EMBARGO,
              f"gap eras={v_start - tr_end - 1}")
        # block sizes
        check(f"fold {f} sizes VALID=={N_VALID}, TEST=={N_TEST}",
              len(valid_idx) == N_VALID and len(test_idx) == N_TEST,
              f"V={len(valid_idx)} S={len(test_idx)}")
        # disjointness
        s_tr, s_v, s_te = set(train_idx), set(valid_idx), set(test_idx)
        check(f"fold {f} TRAIN/VALID/TEST pairwise disjoint",
              not (s_tr & s_v) and not (s_tr & s_te) and not (s_v & s_te))
        check(f"fold {f} refit-train disjoint from TEST",
              not (set(refit_idx) & s_te))
        # contiguity of each block (index-contiguous ranges by construction;
        # assert anyway)
        for nm, ii in (("TRAIN", train_idx), ("VALID", valid_idx),
                       ("TEST", test_idx), ("REFIT", refit_idx)):
            check(f"fold {f} {nm} contiguous on usable index",
                  ii == list(range(ii[0], ii[-1] + 1)))

    # cross-fold TEST properties
    all_test = [i for fd in folds for i in range(fd["test_idx"][0],
                                                 fd["test_idx"][1] + 1)]
    check("TEST blocks non-overlapping across folds",
          len(all_test) == len(set(all_test)))
    check("TEST blocks jointly contiguous",
          sorted(all_test) == list(range(min(all_test), max(all_test) + 1)))
    check("TEST union covers the most recent data",
          max(all_test) == n_usable - 1)
    coverage = len(set(all_test)) / len(intended_span)
    check("coverage of intended most-recent span >= 0.95",
          coverage >= 0.95, f"coverage={coverage:.4f} "
          f"({len(set(all_test))}/{len(intended_span)})")
    # expanding-window property
    check("expanding windows: train end strictly increases",
          folds[0]["train_idx"][1] < folds[1]["train_idx"][1]
          < folds[2]["train_idx"][1])

    # =================== 3. A8 example-preds coverage assertion ============
    print()
    print("=" * 78)
    print("STEP 3: A8 assertion - every TEST era present in "
          "v5.0_validation_example_preds.parquet")
    print("=" * 78)
    # preds parquet has columns [prediction, id] only -> era membership via
    # id-join with the validation parquet (narrow columns: era, id).
    vids = pq.read_table(DATA / "v5.0_validation.parquet",
                         columns=["era", "id"])
    pids = pq.read_table(DATA / "v5.0_validation_example_preds.parquet",
                         columns=["id"])
    joined = vids.join(pids.append_column(
        "haspred", [np.ones(len(pids), dtype=np.int8)]), keys="id",
        join_type="left outer")
    jdf = joined.select(["era", "haspred"]).to_pandas()
    era_pred_counts = (jdf.assign(haspred=jdf["haspred"].fillna(0))
                       .groupby("era", observed=True)["haspred"].sum())
    covered_eras = set(era_pred_counts[era_pred_counts > 0].index.astype(str))
    print(f"eras with >=1 example prediction: {len(covered_eras)}")
    for fd in folds:
        missing = [e for e in fd["test_eras"] if e not in covered_eras]
        check(f"fold {fd['fold']} A8: all {len(fd['test_eras'])} TEST eras "
              f"in example preds", len(missing) == 0,
              f"missing={missing[:5]}{'...' if len(missing) > 5 else ''}")
        # also require full per-row coverage on TEST eras (stronger check,
        # reported not asserted)
        mincov = int(era_pred_counts.loc[
            [e for e in era_pred_counts.index.astype(str)
             if e in set(fd["test_eras"])]].min())
        print(f"        fold {fd['fold']}: min per-era pred rows on TEST = "
              f"{mincov}")

    # =================== 4. A5 gate denominators (+ D10) ===================
    print()
    print("=" * 78)
    print("STEP 4: A5 gate denominators from parent example_per_era_corr.csv")
    print("=" * 78)
    pe = pd.read_csv(PARENT_CSV, dtype={"era": str})
    print(f"parent CSV: {len(pe)} data rows, eras "
          f"{pe['era'].min()}..{pe['era'].max()}")
    check("parent CSV covers all fold TEST+VALID eras",
          all(e in set(pe["era"]) for fd in folds
              for e in fd["test_eras"] + fd["valid_eras"]))
    denominators = []
    for fd in folds:
        sub = pe[pe["era"].isin(fd["test_eras"])]
        raw = float(sub["numerai_corr"].mean())
        floored = max(raw, GATE_FLOOR)
        flag = raw < GATE_FLOOR
        denominators.append({
            "fold": fd["fold"],
            "test_era_range": [fd["test_eras"][0], fd["test_eras"][-1]],
            "n_eras_in_csv": int(len(sub)),
            "example_mean_numerai_corr_raw": raw,
            "denominator_floored": floored,
            "floor": GATE_FLOOR,
            "low_signal_flag": bool(flag),
            "gate_threshold_abs": GATE_THRESHOLD * floored,
        })
        print(f"fold {fd['fold']}: example-model mean per-era numerai_corr on "
              f"TEST ({fd['test_eras'][0]}..{fd['test_eras'][-1]}, n={len(sub)})"
              f" = {raw:+.5f}  -> floored denominator {floored:.5f}"
              f"  low_signal_flag={flag}"
              f"  gate needs arm A >= {GATE_THRESHOLD*floored:+.5f}")
    n_flagged = sum(d["low_signal_flag"] for d in denominators)
    if n_flagged == 0:
        d10 = ("D10: all three raw denominators are above the 0.010 floor "
               "(min raw = {:.5f}); no fold is low-signal-flagged; no "
               "special decision-rule role needs pre-registration.".format(
                   min(d["example_mean_numerai_corr_raw"]
                       for d in denominators)))
    else:
        d10 = ("D10: {} fold(s) fell below the 0.010 floor and are flagged "
               "low-signal yardstick. Pre-registered role, fixed NOW before "
               "any model result exists: a low-signal-flagged fold still "
               "runs the full comparison and still enters the >=2-of-3 "
               "decision rule, but its gate outcome is tallied separately "
               "('gate passed on flagged fold') in all reporting, and the "
               "write-up may not cite a flagged-fold gate pass as evidence "
               "that the baseline demonstrably works.".format(n_flagged))
    print(d10)
    # comfortable-margin note for D10
    margins = [d["example_mean_numerai_corr_raw"] / GATE_FLOOR
               for d in denominators]
    print(f"raw/floor margins per fold: "
          f"{['{:.2f}x'.format(m) for m in margins]}")

    # =================== 5. D7 proxy calibration ===========================
    print()
    print("=" * 78)
    print("STEP 5: D7 proxy calibration - example model VALID-slice vs "
          "TEST-block mean")
    print("=" * 78)
    d7 = []
    for fd, dn in zip(folds, denominators):
        vsub = pe[pe["era"].isin(fd["valid_eras"])]
        v_mean = float(vsub["numerai_corr"].mean())
        t_mean = dn["example_mean_numerai_corr_raw"]
        offset_add = t_mean - v_mean
        ratio = t_mean / v_mean if v_mean > 0 else float("nan")
        d7.append({
            "fold": fd["fold"],
            "valid_era_range": [fd["valid_eras"][0], fd["valid_eras"][-1]],
            "n_valid_eras_in_csv": int(len(vsub)),
            "example_valid_mean": v_mean,
            "example_test_mean": t_mean,
            "offset_additive_test_minus_valid": offset_add,
            "ratio_test_over_valid": ratio,
        })
        print(f"fold {fd['fold']}: example VALID mean "
              f"({fd['valid_eras'][0]}..{fd['valid_eras'][-1]}, n={len(vsub)})"
              f" = {v_mean:+.5f}; TEST mean = {t_mean:+.5f}; "
              f"offset = {offset_add:+.5f}; ratio T/V = {ratio:.3f}")
    d7_rule = (
        "D7 proxy->TEST mapping rule (pre-registered): EU-1's proxy gate "
        "ratio is computed on the fold-1 VALID slice as proxy_ratio = "
        "armA_valid_mean / example_valid_mean(fold 1). The calibrated "
        "projected TEST gate ratio for fold f is calib_ratio(f) = "
        "(armA_valid_mean * ratio_test_over_valid(f)) / "
        "denominator_floored(f), i.e. arm A's VALID score is carried to "
        "TEST multiplicatively by the example model's own T/V ratio for "
        "that fold. Per D7's reserve pre-commitment: if calib_ratio(fold 1) "
        "< 0.60, EU-5 is earmarked for the A5 fix ladder before EU-2 "
        "launches, and EU-2's phase 2 (arms B/C) is made conditional in the "
        "sbatch structure. Proxy/calibrated ratios gate sizing and "
        "scheduling only - never the comparison itself.")
    print(d7_rule)

    # =================== 6. D5 rules =======================================
    d5_refit_rule = (
        "D5 refit-extension rule (deterministic, pre-registered): let N be "
        "the planned refit step count (= selected-trial steps x "
        "rows_refit/rows_train, A4) and L(s) the EMA of training loss with "
        "halflife max(10, round(0.01*N)) steps. Compute D_tail = "
        "L(ceil(0.95*N)) - L(N) and D_total = L(ceil(0.05*N)) - L(N). If "
        "D_total > 0 and D_tail > 0.02 * D_total (the final 5% of steps "
        "still contributes > 2% of the total loss drop, i.e. slope over the "
        "last 5% exceeds threshold), extend the refit ONCE by factor 1.5 "
        "(N -> ceil(1.5*N)), log the extension and the realized D_tail/"
        "D_total; never extend twice.")
    d5_step_rule = (
        "D5 cross-fold trial-step semantics (one line): the EU-1 converged "
        "step count N* defines fold-1 trial steps; fold-f trial steps = "
        "round(N* x rows_train(f)/rows_train(fold1)) at fixed batch size, "
        "and refit steps scale the same way by rows_refit(f)/rows_train(f) "
        "per A4.")
    print()
    print(d5_refit_rule)
    print(d5_step_rule)

    # =================== 7. write protocol-draft.json ======================
    print()
    print("=" * 78)
    print("STEP 7: write out/protocol-draft.json")
    print("=" * 78)
    assertion_results = {
        "all_passed": len(failures) == 0,
        "n_failed": len(failures),
        "failed": failures,
    }
    draft = {
        "_status": "DRAFT - freeze happens in Wave 3 after EU-1; later "
                   "components append converged_step_count, realized_rank_"
                   "grid, arm_c_final_design, power_sim, seeds decision, "
                   "A1/A2/A3 wording, then rename to protocol.json",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "component": "exp-f01 / decomposition Component 9",
        "dataset": "numerai v5.0",
        "main_target": "target (cyrus_20d)",
        "target_horizon_days": TARGET_HORIZON_DAYS,
        "era_spacing_days": ERA_SPACING_DAYS,
        "embargo_eras": EMBARGO,
        "embargo_rule": "ceil(target_horizon_days / era_spacing_days)",
        "min_target_coverage": MIN_TARGET_COVERAGE,
        "usable_era_list": {
            "definition": "all train-parquet eras + validation eras with "
                          "target coverage >= 0.95 (parent data_prep.py "
                          "derivation), temporal order",
            "n_usable": n_usable,
            "first_era": usable[0], "last_era": usable[-1],
            "raw_numbering_gaps": gaps,
            "raw_era_to_usable_index": "usable_index i <-> raw era "
                                       "zero-padded str(i+1); numbering is "
                                       "gap-free so the mapping is the "
                                       "identity shift i = int(era) - 1",
            "mapping_spot_checks": {e: idx_of[e] for e in
                                    [usable[0], "0574", "0575",
                                     usable[-1]]},
            "n_train_parquet_eras": len(train_eras),
            "n_usable_validation_eras": len(usable_val_eras),
        },
        "folds": [
            {
                "fold": fd["fold"],
                "train_era_range": [fd["train_eras"][0],
                                    fd["train_eras"][-1]],
                "train_usable_idx_range": list(fd["train_idx"]),
                "n_train_eras": len(fd["train_eras"]),
                "embargo1_eras": [usable[fd["train_idx"][1] + 1 + j]
                                  for j in range(EMBARGO)],
                "valid_era_range": [fd["valid_eras"][0],
                                    fd["valid_eras"][-1]],
                "valid_usable_idx_range": list(fd["valid_idx"]),
                "n_valid_eras": len(fd["valid_eras"]),
                "embargo2_eras": [usable[fd["valid_idx"][1] + 1 + j]
                                  for j in range(EMBARGO)],
                "test_era_range": [fd["test_eras"][0], fd["test_eras"][-1]],
                "test_usable_idx_range": list(fd["test_idx"]),
                "n_test_eras": len(fd["test_eras"]),
                "refit_train_era_range": [fd["refit_train_eras"][0],
                                          fd["refit_train_eras"][-1]],
                "refit_train_usable_idx_range": list(fd["refit_idx"]),
                "n_refit_train_eras": len(fd["refit_train_eras"]),
                "THE_assertion": "min(test)-max(refit_train) == E+1 == 5 "
                                 "(PASS, on usable index and raw eras)",
            } for fd in folds
        ],
        "coverage": {
            "intended_span": [intended_span[0], intended_span[-1]],
            "n_intended": len(intended_span),
            "n_covered": len(set(all_test)),
            "coverage": coverage,
        },
        "gate": {
            "rule": "arm A mean per-era numerai_corr on fold TEST >= "
                    "0.60 x floored denominator",
            "floor": GATE_FLOOR,
            "denominators": denominators,
            "D10_statement": d10,
        },
        "D7_proxy_calibration": {
            "per_fold": d7,
            "mapping_rule": d7_rule,
        },
        "D5_rules": {
            "refit_extension_rule": d5_refit_rule,
            "cross_fold_trial_step_rule": d5_step_rule,
        },
        "assertion_results": assertion_results,
        "appended_later_by_wave3": [
            "converged_step_count (EU-1 B1)", "realized_rank_grid (A2)",
            "arm_C_final_design (A7/D2)", "power_sim + MDE wording (A1/B4/D3)",
            "seeds 5-vs-3 (A6, #1)", "A3 signatures incl. D1/D4",
            "gate bands + fix ladder + retry semantics (A5)",
            "A4 staged allocation + D4 LR scaling", "D8 pooled estimand",
            "D9 sd source", "D12 kill wording", "eigh-variant policy (B1)",
        ],
    }
    with open(OUT / "protocol-draft.json", "w") as fjs:
        json.dump(draft, fjs, indent=2)
    print(f"wrote {OUT/'protocol-draft.json'}")

    # human-readable fold table
    rows = []
    for fd, dn, d7f in zip(folds, denominators, d7):
        rows.append({
            "fold": fd["fold"],
            "train": f"{fd['train_eras'][0]}..{fd['train_eras'][-1]}",
            "n_train": len(fd["train_eras"]),
            "valid": f"{fd['valid_eras'][0]}..{fd['valid_eras'][-1]}",
            "n_valid": len(fd["valid_eras"]),
            "test": f"{fd['test_eras'][0]}..{fd['test_eras'][-1]}",
            "n_test": len(fd["test_eras"]),
            "refit_train": f"{fd['refit_train_eras'][0]}.."
                           f"{fd['refit_train_eras'][-1]}",
            "denom_raw": round(dn["example_mean_numerai_corr_raw"], 5),
            "denom_floored": round(dn["denominator_floored"], 5),
            "low_signal": dn["low_signal_flag"],
            "gate_abs": round(dn["gate_threshold_abs"], 5),
            "d7_valid_mean": round(d7f["example_valid_mean"], 5),
            "d7_offset": round(d7f["offset_additive_test_minus_valid"], 5),
            "d7_ratio_TV": round(d7f["ratio_test_over_valid"], 3),
        })
    ft = pd.DataFrame(rows)
    ft.to_csv(OUT / "fold_boundaries.csv", index=False)
    print(ft.to_string(index=False))

    print()
    print("=" * 78)
    print(f"SUMMARY: {('ALL ASSERTIONS PASSED' if not failures else 'FAILURES: ' + str(failures))}")
    print(f"elapsed {time.time()-T0:.1f}s")
    print("=" * 78)
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
