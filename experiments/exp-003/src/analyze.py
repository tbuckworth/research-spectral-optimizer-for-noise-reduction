#!/usr/bin/env python3
"""exp-003 local analysis: F5 block length, F3 frozen threshold, #3 power
simulation (moving-block bootstrap), #4 budget arithmetic, F4 go/no-go gate.

Runs locally on CPU from the sweep outputs pulled back from the cluster.
F1: uses only the VERDICT-BLOCK ERA COUNT from protocol.json — never loads
verdict-block data.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path(__file__).parent
EXP = SRC.parent
OUT = EXP / "out"
REF = EXP / "ref"
EXP001_OUT = EXP.parent / "exp-001" / "out"

rng = np.random.default_rng(2026)

N_OUTER = 1000   # MC replications per scenario
N_INNER = 500    # inner bootstrap resamples for the CI
SEEDS_PER_ARM = 3
CI_LEVEL = 95.0


def acf(x, max_lag=20):
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    out = []
    for lag in range(1, max_lag + 1):
        a, b = x[:-lag], x[lag:]
        out.append(float(np.corrcoef(a, b)[0, 1]))
    return out


def block_boot_means(x, L, n_boot, rg):
    """Circular moving-block bootstrap means of series x."""
    n = len(x)
    nb = int(np.ceil(n / L))
    starts = rg.integers(0, n, size=(n_boot, nb))
    idx = (starts[..., None] + np.arange(L)) % n
    return x[idx].reshape(n_boot, nb * L)[:, :n].mean(axis=1)


def block_resample(x, L, n_out_len, rg):
    """One circular moving-block resample of length n_out_len from x."""
    n = len(x)
    nb = int(np.ceil(n_out_len / L))
    starts = rg.integers(0, n, size=nb)
    idx = (starts[:, None] + np.arange(L)) % n
    return x[idx].reshape(-1)[:n_out_len]


def simulate_scenario(noise_pool, L, n_eras, delta, thr, n_outer, iid=False):
    """noise_pool: list of demeaned difference series (per-era, 3-seed-mean
    scale). Returns dict of probabilities and CI half-width stats."""
    res = {"helps_star": 0, "hurts_star": 0, "doesnt": 0, "any_cat": 0,
           "helps_strict": 0, "hurts_strict": 0}
    hw = []
    for _ in range(n_outer):
        src = noise_pool[rng.integers(0, len(noise_pool))]
        if iid:
            sim = rng.choice(src, size=n_eras, replace=True) + delta
        else:
            sim = block_resample(src, L, n_eras, rng) + delta
        means = (rng.choice(sim, size=(N_INNER, n_eras), replace=True)
                 .mean(axis=1) if iid else
                 block_boot_means(sim, L, N_INNER, rng))
        lo, hi = np.percentile(means, [(100 - CI_LEVEL) / 2,
                                       100 - (100 - CI_LEVEL) / 2])
        m = sim.mean()
        hw.append((hi - lo) / 2)
        helps = lo > 0
        hurts = hi < 0
        doesnt = (lo > -thr) and (hi < thr)
        res["helps_star"] += helps
        res["hurts_star"] += hurts
        res["doesnt"] += doesnt
        res["any_cat"] += (helps or hurts or doesnt)
        res["helps_strict"] += (helps and m >= thr)
        res["hurts_strict"] += (hurts and m <= -thr)
    out = {k: v / n_outer for k, v in res.items()}
    out["ci_halfwidth_median"] = float(np.median(hw))
    out["ci_halfwidth_p90"] = float(np.percentile(hw, 90))
    return out


def main():
    sweep = json.load(open(OUT / "sweep_results.json"))
    proto = json.load(open(REF / "protocol.json"))
    f9 = json.load(open(REF / "f9_recalibration.json"))
    bs = pd.read_csv(OUT / "per_era_best_seeds.csv")
    n_verdict = len(proto["verdict_block"])  # era COUNT only (F1)
    print(f"verdict-block era count (from protocol.json): {n_verdict}")

    best = sweep["best"]
    report = {"best": best, "zero_predictor": sweep["zero_predictor"],
              "leakage_gate_fired": sweep["leakage_gate_fired"],
              "n_verdict_eras": n_verdict}

    # ---------------- F12 under-exploration signatures ----------------
    tdf = pd.DataFrame(sweep["trials"])
    grid = sweep["grid"]
    axes = {"lr": grid["lr"], "wd": grid["wd"], "dropout": grid["dropout"]}
    sig = {}
    for ax, vals in axes.items():
        others = [a for a in axes if a != ax]
        # slice through the best config along this axis
        sl = tdf[(tdf[others[0]] == best[others[0]])
                 & (tdf[others[1]] == best[others[1]])].sort_values(ax)
        curve = list(zip(sl[ax].tolist(), sl["mean_spearman"].tolist()))
        vs = sorted(vals)
        at_min = best[ax] == vs[0]
        at_max = best[ax] == vs[-1]
        s = sl["mean_spearman"].to_numpy()
        # improving toward the boundary the best sits on?
        improving = None
        if at_max and len(s) >= 2:
            improving = bool(s[-1] > s[-2])
        elif at_min and len(s) >= 2:
            improving = bool(s[0] > s[1])
        sig[ax] = {"best_value": best[ax], "grid": vs,
                   "on_boundary": bool(at_min or at_max),
                   "trend_still_improving_at_boundary": improving,
                   "slice_through_best": curve}
        print(f"F12 axis {ax}: best={best[ax]} grid={vs} "
              f"boundary={sig[ax]['on_boundary']} "
              f"improving_at_boundary={improving} slice={curve}")
    # non-monotone shape along lr for each (wd, dropout) slice
    lr_shapes = []
    for (wd, do), sl in tdf.groupby(["wd", "dropout"]):
        s = sl.sort_values("lr")["mean_spearman"].to_numpy()
        d = np.diff(s)
        if all(x <= 0 for x in d):
            shape = "monotone decreasing in lr"
        elif all(x >= 0 for x in d):
            shape = "monotone increasing in lr (boundary risk)"
        elif d[0] > 0 and d[-1] < 0:
            shape = "interior peak (well-behaved)"
        else:
            shape = "valley / irregular (under-exploration signature)"
        lr_shapes.append({"wd": wd, "dropout": do, "shape": shape,
                          "values": [float(x) for x in s]})
    n_bad = sum(1 for r in lr_shapes if "signature" in r["shape"]
                or "risk" in r["shape"])
    report["f12_signatures"] = {
        "per_axis": sig,
        "lr_slice_shapes": lr_shapes,
        "boundary_axes": [a for a in sig if sig[a]["on_boundary"]],
        "summary": ("best config sits on the grid boundary for "
                    + ", ".join(a for a in sig if sig[a]["on_boundary"])
                    + "; these signatures carry into exp-004's F12 "
                      "verdict-downgrade rule")}
    print(f"F12: boundary axes = {report['f12_signatures']['boundary_axes']}; "
          f"irregular lr slices: {n_bad}/{len(lr_shapes)}")

    # ---------------- F5: autocorrelation + block length ----------------
    sp0 = bs["spearman_seed0"].to_numpy()
    nc0 = bs["numerai_corr_seed0"].to_numpy()
    acf_nc = acf(nc0)
    acf_sp = acf(sp0)
    overlap_min = proto["embargo_eras"]  # ceil(20d/5d) = 4
    first_small = next((i + 1 for i, a in enumerate(acf_nc) if abs(a) < 0.1),
                       len(acf_nc) + 1)
    L = max(overlap_min, first_small)
    report["f5"] = {"lag1_autocorr_numerai_corr_best": acf_nc[0],
                    "lag1_autocorr_spearman_best": acf_sp[0],
                    "acf_numerai_corr_lags1to10": acf_nc[:10],
                    "overlap_min_block": overlap_min,
                    "first_lag_below_0.1": first_small,
                    "block_length_rule": "L = max(embargo overlap, first lag "
                                         "with |acf|<0.1)",
                    "block_length": int(L)}
    print(f"F5: lag-1 acf (numerai_corr, best seed0) = {acf_nc[0]:+.3f}; "
          f"acf[1..5] = {[f'{a:+.2f}' for a in acf_nc[:5]]}; "
          f"block length L = {L}")

    # ---------------- F3: FROZEN threshold ----------------
    # "Realized tuned-baseline mean per-era corr": two candidate readings.
    #  (a) sweep-selection value (seed 0 of the best trial) -- inflated by
    #      selection over 12 trials (winner's curse);
    #  (b) mean over the 3 available seeds of the best config -- the honest
    #      estimate of what the tuned baseline arm will actually deliver in
    #      exp-004 (a 3-seed-mean arm), partially de-biasing the selection.
    # We freeze (b): it is the lower and therefore the more conservative
    # choice for the power/F4 computation (a smaller threshold is harder to
    # resolve; if power holds at (b) it holds at (a)), and it is the better
    # predictor of the baseline level the verdict comparison will see.
    seed_mean_nc = float(np.mean([bs[f"numerai_corr_seed{s}"].to_numpy()
                                  .mean() for s in (0, 1, 2)]))
    seed_mean_sp = float(np.mean([bs[f"spearman_seed{s}"].to_numpy()
                                  .mean() for s in (0, 1, 2)]))
    thr_nc = min(0.005, 0.25 * seed_mean_nc)
    thr_sp = min(0.005, 0.25 * seed_mean_sp)
    report["f3_frozen_threshold"] = {
        "rule": "min(0.005, 0.25 x realized tuned-baseline mean per-era corr)",
        "baseline_definition": "mean over 3 seeds of the best config "
                               "(winner's-curse-adjusted), NOT the "
                               "sweep-selection value",
        "baseline_numerai_corr_3seed_mean": seed_mean_nc,
        "baseline_spearman_3seed_mean": seed_mean_sp,
        "alternative_sweep_selection_value_nc": best["mean_numerai_corr"],
        "alternative_threshold_if_selection_value_used":
            min(0.005, 0.25 * best["mean_numerai_corr"]),
        "primary_metric": "numerai_corr",
        "threshold_numerai_corr": thr_nc,
        "threshold_spearman": thr_sp,
        "frozen_before_any_arm_comparison": True}
    print(f"F3 realized baseline (3-seed mean): numerai_corr "
          f"{seed_mean_nc:+.5f}, spearman {seed_mean_sp:+.5f} "
          f"(sweep-selection value was {best['mean_numerai_corr']:+.5f})")
    print(f"F3 FROZEN threshold: numerai_corr {thr_nc:.5f} "
          f"(spearman variant {thr_sp:.5f}; selection-value alternative "
          f"would have been "
          f"{min(0.005, 0.25 * best['mean_numerai_corr']):.5f})")

    # ---------------- seed-noise model for paired differences ----------
    # Pairwise per-era differences between seeds of the SAME config estimate
    # the per-era noise of a paired arm-difference from seed variation.
    # diff of 3-seed means has var 2*sig^2/3; a single seed pair has 2*sig^2
    # -> scale pairwise series by 1/sqrt(3).
    pools = {}
    for met in ("numerai_corr", "spearman"):
        cols = [f"{met}_seed{s}" for s in (0, 1, 2)]
        M = bs[cols].to_numpy()
        pairs = [(0, 1), (0, 2), (1, 2)]
        D = [(M[:, i] - M[:, j]) for i, j in pairs]
        D = [(d - d.mean()) / np.sqrt(SEEDS_PER_ARM) for d in D]
        pools[met] = D
        sig_seed = float(np.sqrt(np.mean(np.var(M, axis=1, ddof=1))))
        sd_diff3 = float(np.mean([d.std(ddof=1) for d in D]))
        acf_d = acf(D[0], max_lag=10)
        if met == "numerai_corr":
            report["seed_noise"] = {
                "per_era_seed_sd": sig_seed,
                "per_era_diff_of_3seed_means_sd": sd_diff3,
                "diff_series_lag1_acf": acf_d[0]}
            print(f"seed noise ({met}): per-era seed sd {sig_seed:.5f}; "
                  f"3-seed-mean diff sd {sd_diff3:.5f}; "
                  f"diff-series lag1 acf {acf_d[0]:+.3f}")

    # ---------------- #3 power simulation ----------------
    scen = {"null": 0.0, "plus_thr": thr_nc, "minus_thr": -thr_nc,
            "plus_2thr": 2 * thr_nc}
    power = {}
    for noise_tag, mult in (("seed_noise", 1.0), ("2x_noise", 2.0)):
        pool = [d * mult for d in pools["numerai_corr"]]
        power[noise_tag] = {}
        for name, delta in scen.items():
            power[noise_tag][name] = simulate_scenario(
                pool, L, n_verdict, delta, thr_nc, N_OUTER)
        # naive iid comparison (NOT headline, F5)
        power[noise_tag]["null_iid_comparison"] = simulate_scenario(
            pool, L, n_verdict, 0.0, thr_nc, N_OUTER // 2, iid=True)
    report["power_sim"] = {"n_outer": N_OUTER, "n_inner": N_INNER,
                          "block_length": int(L),
                          "n_eras_simulated": n_verdict,
                          "seeds_per_arm": SEEDS_PER_ARM,
                          "threshold_used": thr_nc,
                          "results": power}
    for tag in power:
        for name in ("null", "plus_thr", "minus_thr"):
            r = power[tag][name]
            print(f"power[{tag}][{name}]: hw_med {r['ci_halfwidth_median']:.5f} "
                  f"any_cat {r['any_cat']:.3f} helps* {r['helps_star']:.3f} "
                  f"doesnt {r['doesnt']:.3f} helps_strict {r['helps_strict']:.3f}")

    # #3 pass/fail numbers
    hw = power["seed_noise"]["null"]["ci_halfwidth_median"]
    report["c3_verdict_inputs"] = {
        "ci_halfwidth_median_seed_noise": hw,
        "pass_rule": "half-width <= ~F3 threshold", "thr": thr_nc,
        "hard_fail_rule": "half-width > 0.008"}

    # ---------------- F4 joint gate ----------------
    gate_scen = ("null", "plus_thr", "minus_thr")
    for tag in ("seed_noise", "2x_noise"):
        vals = {s: power[tag][s]["any_cat"] for s in gate_scen}
        report.setdefault("f4_gate", {})[tag] = {
            "p_any_category_by_scenario": vals,
            "min_p": min(vals.values())}
    gate = report["f4_gate"]["seed_noise"]["min_p"]
    report["f4_gate"]["decision_rule"] = "GO if min P(any category) >= 0.6"
    report["f4_gate"]["gate_value"] = gate
    report["f4_gate"]["decision"] = "GO" if gate >= 0.6 else "NO-GO"
    print(f"\nF4 gate: min P(any verdict category) = {gate:.3f} -> "
          f"{report['f4_gate']['decision']}  "
          f"(2x-noise sensitivity: {report['f4_gate']['2x_noise']['min_p']:.3f})")

    # ---------------- #4 budget arithmetic ----------------
    t_adamw = float(np.median(tdf["train_seconds"] + tdf["eval_seconds"]))
    t_eval = float(np.median(tdf["eval_seconds"]))
    timing_file = EXP001_OUT.parent / "cluster" / "out" / "timing.json"
    if not timing_file.exists():
        timing_file = EXP001_OUT / "timing.json"
    if timing_file.exists():
        timing = json.load(open(timing_file))
        fon = [t for t in timing if t.get("filter_on") and t.get("B") == 1024]
        t_spectral = (fon[0]["ms_per_step"] * 2000 / 1000 + t_eval
                      if fon else None)
        t_sp_src = (f"exp-001 {timing_file.name} (measured "
                    f"{fon[0]['ms_per_step']:.1f} ms/step, B=1024 filter-on, "
                    f"2000 steps) + measured eval")
    else:
        t_spectral = None
    if t_spectral is None:
        t_spectral = 3.9 * 60  # exp-002 conservative L40 projection
        t_sp_src = "exp-002 projection (3.9 min/2000-step run, 10x CPU->L40)"
    # F12: tuning match defined in COMPUTE. AdamW tuning spent 12 trials.
    adamw_tuning_compute = 12 * t_adamw
    n_sp_trials = max(1, int(round(adamw_tuning_compute / t_spectral)))
    job_cap = 25 * 60
    overhead = 120.0  # data load, imports, diagnostics logging
    # exp-004 single job: spectral tuning (F12-matched) + 2 arms x 3 seeds
    # (filter-on spectral, filter-off = tuned AdamW at identical base config)
    # + C4 random-subspace norm-matched control x 3 seeds + GAF-style
    # simple-agreement ablation x 3 seeds (both co-scheduled, F7-protected
    # order) + diagnostics.
    t_gaf = t_spectral  # dominated by per-sample grads; eigh was ~3% (exp-002)
    t_rs = t_spectral   # same per-sample-grad cost, random basis instead
    exp004 = (n_sp_trials * t_spectral      # spectral tuning (matched compute)
              + 3 * t_spectral              # filter-on arm, 3 seeds
              + 3 * t_adamw                 # filter-off arm, 3 seeds
              + 3 * t_rs                    # C4 random-subspace, 3 seeds
              + 3 * t_gaf                   # GAF-style ablation, 3 seeds
              + overhead)
    exp004_min_viable = (n_sp_trials * t_spectral + 3 * t_spectral
                         + 3 * t_adamw + 3 * t_rs + overhead)
    # exp-005 (conditional, exp-002 PASSED): sequence arm, 2 arms x 3 seeds,
    # split as a 2-task array (one arm per task) so each task stays well
    # under the cap even at the conservative 3.9-min projection.
    t_seq = 3.9 * 60  # exp-002 conservative projection (filter-on)
    exp005_task = 3 * t_seq + overhead     # one arm x 3 seeds per array task
    exp005_total = 6 * t_seq + 2 * overhead
    budget = {
        "t_adamw_run_seconds_measured": t_adamw,
        "t_spectral_run_seconds": t_spectral, "t_spectral_source": t_sp_src,
        "adamw_tuning_compute_seconds": adamw_tuning_compute,
        "f12_matched_spectral_trials": n_sp_trials,
        "job_cap_seconds": job_cap,
        "exp004_single_job_seconds": exp004,
        "exp004_fits": exp004 <= job_cap,
        "exp004_min_viable_seconds": exp004_min_viable,
        "exp005_seq_array_task_seconds": exp005_task,
        "exp005_seq_task_fits": exp005_task <= job_cap,
        "exp005_seq_total_seconds": exp005_total,
        "f7_cuts_needed": exp004 > job_cap,
        "notes": ["Muon and GBT: pre-designated cuts (F7 order, already "
                  "future work W3) -- not scheduled",
                  "sequence arm conditional on exp-002 PASS (it passed) and "
                  "on room here (there is room); cut before GAF/random-"
                  "subspace under pressure (F7) -- no pressure exists",
                  "C4 random-subspace control protected (F7), co-scheduled "
                  "inside the exp-004 job",
                  "F12: matched-compute tuning affords only ~1 spectral "
                  "trial; spectral arm runs at the transferred prior-repo "
                  "search center -- an under-exploration signature for the "
                  "SPECTRAL arm, feeding exp-004's F12 downgrade rule"],
    }
    report["budget"] = budget
    print(f"\n#4 budget: t_adamw {t_adamw:.1f}s/run, t_spectral "
          f"{t_spectral:.1f}s/run ({t_sp_src})")
    print(f"  AdamW tuning compute {adamw_tuning_compute:.0f}s -> matched "
          f"spectral trials: {n_sp_trials}")
    print(f"  exp-004 single job (tune + 2x3 arms + C4 + GAF + diag): "
          f"{exp004/60:.1f} min (cap 25) fits={budget['exp004_fits']}")
    print(f"  exp-004 minimum viable (drop GAF): "
          f"{exp004_min_viable/60:.1f} min")
    print(f"  exp-005 sequence arm: 2-task array, {exp005_task/60:.1f} "
          f"min/task (cap 25) fits={budget['exp005_seq_task_fits']}; "
          f"total {exp005_total/60:.1f} min")
    print(f"  F7 cut order invoked: {budget['f7_cuts_needed']}")

    with open(OUT / "analysis.json", "w") as f:
        json.dump(report, f, indent=2, default=float)
    print("\nwrote out/analysis.json")


if __name__ == "__main__":
    main()
