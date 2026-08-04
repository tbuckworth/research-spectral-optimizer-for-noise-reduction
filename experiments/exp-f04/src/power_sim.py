"""exp-f04: realized-power measurement of the verdict machinery BEFORE any
TEST touch (B4/A1/A6/D3/D8).

Calibration inputs (READ-ONLY, parent run):
  - exp-001/out/example_per_era_corr.csv  (651 eras 0575-1225; arm-level ACF texture)
  - exp-004/out/per_era_numerai_corr.csv  (255 eras 0971-1225; 3-seed paired arms
    -> per-era paired difference panel D[era, seed] = filter_on - filter_off)

Fold windows: exp-f01/out/protocol-draft.json (prospective TEST blocks,
eras 0896-1005 / 1006-1115 / 1116-1225; 110 eras each).

DGP (hierarchical, calibrated to the parent's observed difference noise):
  D[e, s] = delta + b_s + eps_e + u_{e,s}
    b_s   ~ N(0, sigma_b^2)   seed main effect      (parent panel estimate)
    eps_e ~ AR(1)(rho_f) with stationary sd sigma_e (era-common component)
    u_es  ~ N(0, sigma_u^2)   era x seed residual
  sigma_b, sigma_e, sigma_u estimated once from the parent 255-era 3-seed
  panel by two-way ANOVA decomposition. rho_f is per fold window:
  the seed-averaged parent diff series' lag-1 ACF on the window overlap when
  the overlap has >= 60 eras (folds 2, 3), else the full-panel diff ACF
  (fold 1, overlap only 35 eras - flagged). A HIGH-ACF STRESS variant sets
  rho_f to the example model's own within-window arm-level ACF (~0.71), the
  pre-mortem's "recent-window ACF near 0.763" scenario.

Per Monte-Carlo replicate the FULL pipeline runs exactly as the verdict
analysis will: seed-average -> lag-1 ACF -> block length -> hierarchical
era-block x seed MBB -> 95% CI. Power = P(CI excludes 0); coverage =
P(CI contains delta); MDE = median CI half-width (D3 honest name:
"CI half-width (~50%-power MDE)").

B4 gate (wired): PASS iff at the 3-seed design floor
  P(CI excludes 0 | |delta| = 0.005) >= 0.6 on >= 2 of 3 folds
  AND median MDE <= 0.005
  AND conclusions stable across 2 bootstrap RNG seeds.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hier_bootstrap import block_length, hier_mbb, lag1_acf  # noqa: E402

PARENT = Path("/media/titus/big/researcher-output/"
              "2026-07-31-spectral-optimizer-for-noise-reduction-on-financial-timeseri")
RUN = Path("/media/titus/big/researcher-output/"
           "2026-08-03-followup-follow-up-spectral-optimizer-redo-pxp-filter-walk-")
OUT = RUN / "experiments/exp-f04/out"

N_MC = 500          # Monte-Carlo replicates per scenario cell
N_BOOT = 2000       # bootstrap replicates per CI
EFFECTS = [0.0, +0.005, -0.005]
SEED_COUNTS = [3, 5]
BOOT_RNG_SEEDS = [20260804, 987654321]  # stability check across 2 bootstrap RNGs
DGP_SEED_BASE = 555000
MIN_OVERLAP_FOR_WINDOW_ACF = 60
PARENT_SEEDS = [0, 1, 2]


def anova_components(D: np.ndarray) -> dict:
    """Two-way (era x seed) ANOVA decomposition of the paired-diff panel."""
    n_e, n_s = D.shape
    mu = D.mean()
    era_means = D.mean(axis=1)
    seed_means = D.mean(axis=0)
    resid = D - era_means[:, None] - seed_means[None, :] + mu
    sig_u2 = float((resid ** 2).sum() / ((n_e - 1) * (n_s - 1)))
    sig_b2 = float(max(seed_means.var(ddof=1) - sig_u2 / n_e, 0.0))
    sig_e2 = float(max(era_means.var(ddof=1) - sig_u2 / n_s, 0.0))
    return {"mu": float(mu), "sigma_u": np.sqrt(sig_u2),
            "sigma_b": np.sqrt(sig_b2), "sigma_e": np.sqrt(sig_e2)}


def simulate_panel(rng: np.random.Generator, n: int, S: int, delta: float,
                   rho: float, sigma_e: float, sigma_u: float,
                   sigma_b: float) -> np.ndarray:
    """One simulated (n, S) paired-difference panel under the DGP."""
    b = rng.normal(0.0, sigma_b, size=S)
    z = rng.normal(0.0, 1.0, size=n)
    eps = np.empty(n)
    eps[0] = z[0] * sigma_e
    c = np.sqrt(max(1.0 - rho * rho, 0.0)) * sigma_e
    for t in range(1, n):
        eps[t] = rho * eps[t - 1] + c * z[t]
    u = rng.normal(0.0, sigma_u, size=(n, S))
    return delta + b[None, :] + eps[:, None] + u


def run_cell(n: int, S: int, delta: float, rho: float, comp: dict,
             dgp_seed: int, boot_seed: int) -> dict:
    """One scenario cell: N_MC replicates of the full analysis pipeline."""
    dgp_rng = np.random.default_rng(dgp_seed)
    boot_rng = np.random.default_rng(boot_seed)
    half_widths = np.empty(N_MC)
    excl0 = np.zeros(N_MC, dtype=bool)
    cover = np.zeros(N_MC, dtype=bool)
    hw_era = np.empty(N_MC)
    excl0_era = np.zeros(N_MC, dtype=bool)
    cover_era = np.zeros(N_MC, dtype=bool)
    Ls = np.empty(N_MC, dtype=int)
    rhos = np.empty(N_MC)
    for i in range(N_MC):
        D = simulate_panel(dgp_rng, n, S, delta, rho,
                           comp["sigma_e"], comp["sigma_u"], comp["sigma_b"])
        r = hier_mbb(D, n_boot=N_BOOT, rng=boot_rng)
        half_widths[i] = r["half_width"]
        excl0[i] = r["excludes_zero"]
        cover[i] = r["lo"] <= delta <= r["hi"]
        Ls[i] = r["block_length"]
        rhos[i] = r["rho_lag1"]
        # A6 fallback variant: era-block-only MBB on the seed-averaged series
        # (headline only if seed variance is demonstrably negligible; measured
        # here so Wave 3 knows what that justification would buy).
        r2 = hier_mbb(D.mean(axis=1, keepdims=True), n_boot=N_BOOT, rng=boot_rng)
        hw_era[i] = r2["half_width"]
        excl0_era[i] = r2["excludes_zero"]
        cover_era[i] = r2["lo"] <= delta <= r2["hi"]
    return {
        "n_eras": n, "n_seeds": S, "delta": delta, "dgp_rho": rho,
        "n_mc": N_MC, "n_boot": N_BOOT,
        "power_excl0": float(excl0.mean()),
        "coverage": float(cover.mean()),
        "median_half_width": float(np.median(half_widths)),
        "power_excl0_era_only": float(excl0_era.mean()),
        "coverage_era_only": float(cover_era.mean()),
        "median_half_width_era_only": float(np.median(hw_era)),
        "mean_realized_rho": float(rhos.mean()),
        "median_block_length": float(np.median(Ls)),
    }


def main() -> None:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)

    # ---- calibration from parent per-era vectors -------------------------
    ex = pd.read_csv(PARENT / "experiments/exp-001/out/example_per_era_corr.csv",
                     dtype={"era": str})
    d4 = pd.read_csv(PARENT / "experiments/exp-004/out/per_era_numerai_corr.csv",
                     dtype={"era": str})
    D_parent = np.stack(
        [(d4[f"filter_on_s{s}"] - d4[f"filter_off_s{s}"]).to_numpy()
         for s in PARENT_SEEDS], axis=1)
    comp = anova_components(D_parent)
    global_diff_acf = lag1_acf(D_parent.mean(axis=1))
    print("== calibration (parent 255-era, 3-seed paired-diff panel) ==")
    print(f"  sigma_e (era-common)   = {comp['sigma_e']:.5f}")
    print(f"  sigma_u (era x seed)   = {comp['sigma_u']:.5f}")
    print(f"  sigma_b (seed main)    = {comp['sigma_b']:.5f}")
    print(f"  panel grand mean       = {comp['mu']:+.5f} (parent B-A; sim uses injected delta)")
    print(f"  full-panel diff lag-1 ACF = {global_diff_acf:+.4f}")

    proto = json.load(open(RUN / "experiments/exp-f01/out/protocol-draft.json"))
    folds = []
    for f in proto["folds"]:
        a, b = f["test_era_range"]
        w_ex = ex[(ex.era >= a) & (ex.era <= b)]
        ov = d4[(d4.era >= a) & (d4.era <= b)]
        n = len(w_ex)
        assert n == f["n_test_eras"] == 110
        rho_example = lag1_acf(w_ex.numerai_corr.to_numpy())
        D_ov = np.stack(
            [(ov[f"filter_on_s{s}"] - ov[f"filter_off_s{s}"]).to_numpy()
             for s in PARENT_SEEDS], axis=1) if len(ov) > 2 else None
        rho_ov = lag1_acf(D_ov.mean(axis=1)) if D_ov is not None else None
        use_ov = D_ov is not None and len(ov) >= MIN_OVERLAP_FOR_WINDOW_ACF
        rho_d = rho_ov if use_ov else global_diff_acf
        folds.append({
            "fold": f["fold"], "test_era_range": [a, b], "n": n,
            "rho_example_window": rho_example,
            "diff_overlap_n": int(len(ov)),
            "rho_diff_overlap": rho_ov,
            "rho_dgp_primary": rho_d,
            "rho_dgp_source": ("window-overlap diff ACF" if use_ov
                               else f"full-panel diff ACF (overlap {len(ov)} < "
                                    f"{MIN_OVERLAP_FOR_WINDOW_ACF} eras - flagged)"),
            "diff_overlap_seedavg_sd": (float(D_ov.mean(axis=1).std(ddof=1))
                                        if D_ov is not None else None),
            "example_window_sd": float(w_ex.numerai_corr.std(ddof=1)),
            "block_length_primary": block_length(rho_d, n),
            "block_length_stress": block_length(rho_example, n),
        })
        print(f"fold {f['fold']} [{a}-{b}]: rho_example={rho_example:+.4f}  "
              f"diff overlap n={len(ov)} rho_diff={rho_ov:+.4f}  "
              f"-> primary DGP rho={rho_d:+.4f} (L={block_length(rho_d, n)}), "
              f"stress rho={rho_example:+.4f} (L={block_length(rho_example, n)})")

    # ---- simulation grid -------------------------------------------------
    results = {"primary": [], "stress": [], "pooled": []}
    cell_id = 0
    for variant, rho_key, effects in [
            ("primary", "rho_dgp_primary", EFFECTS),
            ("stress", "rho_example_window", [0.0, +0.005])]:
        for fd in folds:
            for S in SEED_COUNTS:
                for delta in effects:
                    cell_id += 1
                    for j, bs in enumerate(BOOT_RNG_SEEDS):
                        r = run_cell(fd["n"], S, delta, fd[rho_key], comp,
                                     dgp_seed=DGP_SEED_BASE + cell_id,
                                     boot_seed=bs)
                        r.update(fold=fd["fold"], variant=variant,
                                 boot_rng_seed=bs, boot_rng_index=j)
                        results[variant].append(r)
                        print(f"[{variant} f{fd['fold']} S={S} delta={delta:+.3f} "
                              f"bootRNG#{j}] power={r['power_excl0']:.3f} "
                              f"cover={r['coverage']:.3f} "
                              f"medHW={r['median_half_width']:.5f} "
                              f"medL={r['median_block_length']:.0f} "
                              f"({time.time()-t0:.0f}s)")

    # ---- D8 pooled secondary estimand (330 concatenated eras) -----------
    # AR(1) continues across fold boundaries with per-segment rho (the three
    # TEST blocks tile contiguous eras 0896-1225).
    def simulate_pooled(rng, S, delta):
        segs = []
        prev = None
        for fd in folds:
            n, rho = fd["n"], fd["rho_dgp_primary"]
            z = rng.normal(0.0, 1.0, size=n)
            eps = np.empty(n)
            c = np.sqrt(max(1.0 - rho * rho, 0.0)) * comp["sigma_e"]
            eps[0] = (rho * prev + c * z[0]) if prev is not None else z[0] * comp["sigma_e"]
            for t in range(1, n):
                eps[t] = rho * eps[t - 1] + c * z[t]
            prev = eps[-1]
            b = rng.normal(0.0, comp["sigma_b"], size=S)  # per-fold refit seeds
            u = rng.normal(0.0, comp["sigma_u"], size=(n, S))
            segs.append(delta + b[None, :] + eps[:, None] + u)
        return np.concatenate(segs, axis=0)

    for S in SEED_COUNTS:
        for delta in EFFECTS:
            cell_id += 1
            for j, bs in enumerate(BOOT_RNG_SEEDS):
                dgp_rng = np.random.default_rng(DGP_SEED_BASE + cell_id)
                boot_rng = np.random.default_rng(bs)
                hw = np.empty(N_MC); e0 = np.zeros(N_MC, bool); cv = np.zeros(N_MC, bool)
                for i in range(N_MC):
                    Dp = simulate_pooled(dgp_rng, S, delta)
                    r = hier_mbb(Dp, n_boot=N_BOOT, rng=boot_rng)
                    hw[i] = r["half_width"]; e0[i] = r["excludes_zero"]
                    cv[i] = r["lo"] <= delta <= r["hi"]
                rec = {"variant": "pooled", "n_eras": 330, "n_seeds": S,
                       "delta": delta, "n_mc": N_MC, "n_boot": N_BOOT,
                       "power_excl0": float(e0.mean()), "coverage": float(cv.mean()),
                       "median_half_width": float(np.median(hw)),
                       "boot_rng_seed": bs, "boot_rng_index": j}
                results["pooled"].append(rec)
                print(f"[pooled S={S} delta={delta:+.3f} bootRNG#{j}] "
                      f"power={rec['power_excl0']:.3f} cover={rec['coverage']:.3f} "
                      f"medHW={rec['median_half_width']:.5f} ({time.time()-t0:.0f}s)")

    # ---- B4 verdict wiring (evaluated at the 3-seed design floor) --------
    def detection_power(variant, fold, S, rng_index):
        """Pooled +/-0.005 detection power for a fold (mean of the +- cells;
        symmetric DGP)."""
        cells = [r for r in results[variant]
                 if r["fold"] == fold and r["n_seeds"] == S
                 and r["boot_rng_index"] == rng_index and abs(r["delta"]) == 0.005]
        return float(np.mean([r["power_excl0"] for r in cells]))

    def mde(variant, fold, S, rng_index):
        cells = [r for r in results[variant]
                 if r["fold"] == fold and r["n_seeds"] == S
                 and r["boot_rng_index"] == rng_index and r["delta"] == 0.0]
        return float(np.mean([r["median_half_width"] for r in cells]))

    verdicts = {}
    for j in range(len(BOOT_RNG_SEEDS)):
        pw = {f["fold"]: detection_power("primary", f["fold"], 3, j) for f in folds}
        md = {f["fold"]: mde("primary", f["fold"], 3, j) for f in folds}
        n_pass = sum(1 for v in pw.values() if v >= 0.6)
        med_mde = float(np.median(list(md.values())))
        verdicts[f"boot_rng_{j}"] = {
            "seed_floor": 3,
            "per_fold_power_pm0.005": pw,
            "per_fold_mde": md,
            "n_folds_power_ge_0.6": n_pass,
            "median_mde": med_mde,
            "b4_power_condition": n_pass >= 2,
            "b4_mde_condition": med_mde <= 0.005,
            "b4_pass": (n_pass >= 2) and (med_mde <= 0.005),
        }
    v0, v1 = verdicts["boot_rng_0"], verdicts["boot_rng_1"]
    max_dp = max(abs(v0["per_fold_power_pm0.005"][f] - v1["per_fold_power_pm0.005"][f])
                 for f in v0["per_fold_power_pm0.005"])
    max_dm = max(abs(v0["per_fold_mde"][f] - v1["per_fold_mde"][f])
                 for f in v0["per_fold_mde"])
    stable = (v0["b4_pass"] == v1["b4_pass"]) and max_dp <= 0.05
    b4_pass = v0["b4_pass"] and stable

    # 5-seed lever numbers (validity + power lever for Wave-3 packing)
    lever5 = {}
    for j in range(len(BOOT_RNG_SEEDS)):
        pw5 = {f["fold"]: detection_power("primary", f["fold"], 5, j) for f in folds}
        md5 = {f["fold"]: mde("primary", f["fold"], 5, j) for f in folds}
        lever5[f"boot_rng_{j}"] = {
            "per_fold_power_pm0.005": pw5, "per_fold_mde": md5,
            "n_folds_power_ge_0.6": sum(1 for v in pw5.values() if v >= 0.6),
            "median_mde": float(np.median(list(md5.values()))),
        }

    # D3 coverage summary (per fold x seeds, pooled over all effect cells, RNG#0)
    coverage = {}
    for S in SEED_COUNTS:
        for f in folds:
            cells = [r for r in results["primary"]
                     if r["fold"] == f["fold"] and r["n_seeds"] == S
                     and r["boot_rng_index"] == 0]
            cov = float(np.mean([r["coverage"] for r in cells]))
            coverage[f"fold{f['fold']}_S{S}"] = cov
    cov3 = float(np.mean([coverage[f"fold{f['fold']}_S3"] for f in folds]))
    cov5 = float(np.mean([coverage[f"fold{f['fold']}_S5"] for f in folds]))
    five_seed_validity_flag = cov3 < 0.925

    # A6 era-only fallback summary (informational; headline only if the
    # realized seed variance is demonstrably negligible, per A6)
    def era_only_summary(S, j):
        pw = {f["fold"]: float(np.mean(
            [r["power_excl0_era_only"] for r in results["primary"]
             if r["fold"] == f["fold"] and r["n_seeds"] == S
             and r["boot_rng_index"] == j and abs(r["delta"]) == 0.005]))
            for f in folds}
        md = {f["fold"]: float(np.mean(
            [r["median_half_width_era_only"] for r in results["primary"]
             if r["fold"] == f["fold"] and r["n_seeds"] == S
             and r["boot_rng_index"] == j and r["delta"] == 0.0]))
            for f in folds}
        cov = {f["fold"]: float(np.mean(
            [r["coverage_era_only"] for r in results["primary"]
             if r["fold"] == f["fold"] and r["n_seeds"] == S
             and r["boot_rng_index"] == j])) for f in folds}
        return {"per_fold_power_pm0.005": pw, "per_fold_mde": md,
                "per_fold_coverage": cov,
                "median_mde": float(np.median(list(md.values()))),
                "n_folds_power_ge_0.6": sum(1 for v in pw.values() if v >= 0.6)}

    era_only = {f"S{S}_rng{j}": era_only_summary(S, j)
                for S in SEED_COUNTS for j in range(len(BOOT_RNG_SEEDS))}

    anchor = ("MDE anchor derivation (D3, one sentence): 0.005 is ~35% of the "
              "gate-level baseline score - the gate requires arm A to reach "
              "0.60 x the example model's per-fold TEST mean (~0.0235 on the "
              "parent's test block, i.e. ~0.014 absolute), and 0.005/0.014 "
              "~= 0.35, so the design is asked to resolve differences of about "
              "one third of the smallest performance the baseline is allowed "
              "to have.")

    out = {
        "component": "exp-f04 / decomposition Component 3 (B4/A1/A6/D3/D8)",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "runtime_seconds": round(time.time() - t0, 1),
        "config": {"n_mc": N_MC, "n_boot": N_BOOT, "effects": EFFECTS,
                   "seed_counts": SEED_COUNTS, "boot_rng_seeds": BOOT_RNG_SEEDS,
                   "dgp_seed_base": DGP_SEED_BASE,
                   "block_length_rule": "AR(1) plug-in, n^(1/3) rate, clipped [2, n//3]"},
        "calibration": {
            "source_panel": "parent exp-004 per_era_numerai_corr.csv, "
                            "filter_on - filter_off, eras 0971-1225, 3 seeds",
            "sigma_e": comp["sigma_e"], "sigma_u": comp["sigma_u"],
            "sigma_b": comp["sigma_b"], "parent_panel_mean": comp["mu"],
            "full_panel_diff_lag1_acf": global_diff_acf,
        },
        "folds": folds,
        "cells": results,
        "b4_verdict": {
            "threshold": "P(CI excl 0 | |delta|=0.005) >= 0.6 on >= 2/3 folds "
                         "AND median MDE <= 0.005 AND stable across 2 boot RNGs; "
                         "evaluated at the 3-seed design floor",
            "per_rng": verdicts,
            "rng_stability": {"max_power_delta": max_dp, "max_mde_delta": max_dm,
                              "stable": stable},
            "b4_pass": bool(b4_pass),
        },
        "five_seed_lever": lever5,
        "a6_era_only_fallback": {
            "note": "era-block-only MBB on the seed-averaged series; may serve "
                    "as headline ONLY if realized seed variance is demonstrably "
                    "negligible (< ~10% of era variance) per A6; measured here "
                    "so the freeze step knows what that justification buys",
            "summaries": era_only},
        "d3_coverage": {"per_fold_seed": coverage,
                        "mean_coverage_3seed": cov3,
                        "mean_coverage_5seed": cov5,
                        "nominal": 0.95,
                        "five_seed_validity_flag": bool(five_seed_validity_flag)},
        "mde_anchor_sentence": anchor,
        "d8_pooled_statement": ("D8 pooled secondary estimand pre-registered: "
                                "concatenated era-level paired (B-A) across the "
                                "three TEST blocks (330 eras) under ONE "
                                "hierarchical era-block x seed MBB, block length "
                                "from the pooled series' own lag-1 ACF. It is "
                                "SECONDARY and cannot overturn the per-fold "
                                "decision rule."),
        "d8_decisiveness_checkpoint_inputs": {
            "per_fold_power_3seed_rng0": v0["per_fold_power_pm0.005"],
            "per_fold_power_5seed_rng0": lever5["boot_rng_0"]["per_fold_power_pm0.005"],
            "pooled_power_3seed": [r["power_excl0"] for r in results["pooled"]
                                   if r["n_seeds"] == 3 and abs(r["delta"]) == 0.005
                                   and r["boot_rng_index"] == 0],
            "pooled_power_5seed": [r["power_excl0"] for r in results["pooled"]
                                   if r["n_seeds"] == 5 and abs(r["delta"]) == 0.005
                                   and r["boot_rng_index"] == 0],
        },
    }
    with open(OUT / "power-sim.json", "w") as fh:
        json.dump(out, fh, indent=1)

    print("\n== B4 VERDICT (3-seed design floor, primary DGP) ==")
    for j in range(2):
        v = verdicts[f"boot_rng_{j}"]
        print(f" bootRNG#{j}: per-fold power {v['per_fold_power_pm0.005']}  "
              f"folds>=0.6: {v['n_folds_power_ge_0.6']}  "
              f"median MDE {v['median_mde']:.5f}  b4_pass={v['b4_pass']}")
    print(f" RNG stability: max power delta {max_dp:.3f}, max MDE delta "
          f"{max_dm:.5f}, stable={stable}")
    print(f" 5-seed lever: power {lever5['boot_rng_0']['per_fold_power_pm0.005']} "
          f" median MDE {lever5['boot_rng_0']['median_mde']:.5f}")
    print(f" coverage: 3-seed {cov3:.3f}, 5-seed {cov5:.3f} (nominal 0.95); "
          f"5-seed validity flag: {five_seed_validity_flag}")
    print(f" A6 era-only fallback (S=3, rng0): "
          f"power {era_only['S3_rng0']['per_fold_power_pm0.005']} "
          f"median MDE {era_only['S3_rng0']['median_mde']:.5f} "
          f"coverage {era_only['S3_rng0']['per_fold_coverage']}")
    print(f"\nB4 OVERALL: {'PASS' if b4_pass else 'FAIL'}")
    print(f"total runtime {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
