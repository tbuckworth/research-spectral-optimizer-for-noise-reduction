#!/usr/bin/env python3
"""exp-005 local architecture-consistency analysis (CPU).

Adapted from exp-004's analyze_verdict.py with the SAME frozen machinery;
the ONLY changes: GAF arm dropped (cut per plan), and an explicit
comparison table against exp-004's MLP numbers is added.

Inputs: out/per_era_numerai_corr.csv, out/per_era_spearman.csv,
out/diagnostics.json (pulled back from the cluster job), plus
<run-dir>/data/shard_verdict.parquet (per-era target dispersion, F8) and
../exp-004/out/verdict_analysis.json (MLP comparison).

Frozen machinery (binding, verbatim exp-004):
  F3: verdict threshold 0.00398 mean per-era numerai_corr (spearman 0.00391).
  F5: circular moving-block bootstrap, L=4 eras, paired per-era design,
      95% percentile CIs, 10,000 resamples, rng seed 2026.
Verdict rule:
  helps  = CI excludes 0 (lo>0) AND mean diff >= thr
  hurts  = CI excludes 0 (hi<0) AND mean diff <= -thr  (then F12 downgrade)
  doesnt = CI within (-thr, +thr)  (equivalence)
  else   -> no-verdict: report effect estimate with CI, honestly scoped.
F8: tail-era breakdown (descriptive, iid bootstrap within buckets, with the
    recorded regression-to-the-mean caveat on the baseline-corr split).
F10: corr-Sharpe diff via the same joint block-bootstrap machinery.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path(__file__).parent
EXP = SRC.parent
OUT = EXP / "out"
DATA = EXP.parent.parent / "data"
EXP004 = EXP.parent / "exp-004" / "out" / "verdict_analysis.json"

# FROZEN (exp-003)
THR_NC = 0.00398
THR_SP = 0.00391
L = 4
N_BOOT = 10000
CI = 95.0
rng = np.random.default_rng(2026)

ARMS = ["filter_off", "filter_on", "c4_random"]
SEEDS = [0, 1, 2]


def block_idx(n, L, n_boot, rg):
    """Circular moving-block bootstrap index matrix (n_boot, n)."""
    nb = int(np.ceil(n / L))
    starts = rg.integers(0, n, size=(n_boot, nb))
    idx = (starts[..., None] + np.arange(L)) % n
    return idx.reshape(n_boot, nb * L)[:, :n]


def boot_ci_mean(x, idx):
    means = x[idx].mean(axis=1)
    lo, hi = np.percentile(means, [(100 - CI) / 2, 100 - (100 - CI) / 2])
    return float(x.mean()), float(lo), float(hi)


def verdict_of(mean, lo, hi, thr):
    if lo > 0 and mean >= thr:
        return "helps"
    if hi < 0 and mean <= -thr:
        return "hurts"
    if lo > -thr and hi < thr:
        return "doesnt_help"
    return "no_verdict"


def sharpe(x):
    s = x.std(ddof=1)
    return float(x.mean() / s) if s > 0 else float("nan")


def main():
    nc = pd.read_csv(OUT / "per_era_numerai_corr.csv")
    sp = pd.read_csv(OUT / "per_era_spearman.csv")
    diag = json.load(open(OUT / "diagnostics.json"))
    n = len(nc)
    print(f"verdict-block eras in eval: {n}")
    report = {"n_eras": n, "frozen": {"thr_nc": THR_NC, "thr_sp": THR_SP,
                                      "block_L": L, "n_boot": N_BOOT,
                                      "ci_level": CI},
              "arch": diag["frozen"].get("arch", "GRU"),
              "config_provenance": diag["frozen"].get("config_provenance")}

    # 3-seed-mean series per arm
    arm_nc = {a: nc[[f"{a}_s{s}" for s in SEEDS]].mean(axis=1).to_numpy()
              for a in ARMS}
    arm_sp = {a: sp[[f"{a}_s{s}" for s in SEEDS]].mean(axis=1).to_numpy()
              for a in ARMS}

    print("\n--- arm levels (mean per-era corr over verdict block) ---")
    levels = {}
    for a in ARMS:
        levels[a] = {"numerai_corr_3seed_mean": float(arm_nc[a].mean()),
                     "spearman_3seed_mean": float(arm_sp[a].mean()),
                     "numerai_corr_per_seed": [
                         float(nc[f"{a}_s{s}"].mean()) for s in SEEDS],
                     "corr_sharpe_3seed_mean_series": sharpe(arm_nc[a])}
        print(f"  {a:11s}: nc {arm_nc[a].mean():+.5f} "
              f"(seeds {['%+.5f' % nc[f'{a}_s{s}'].mean() for s in SEEDS]}), "
              f"sp {arm_sp[a].mean():+.5f}, Sharpe {sharpe(arm_nc[a]):.3f}")
    levels["zero_pred"] = {"numerai_corr": float(nc["zero_pred"].mean()),
                           "spearman": float(sp["zero_pred"].mean())}
    print(f"  zero_pred  : nc {nc['zero_pred'].mean():+.5f}, "
          f"sp {sp['zero_pred'].mean():+.5f}")
    report["levels"] = levels

    # F9 sanity band on realized baseline (GRU)
    f9 = json.load(open(EXP / "ref" / "f9_recalibration.json"))
    band = f9["sanity_band"]
    base = levels["filter_off"]["numerai_corr_3seed_mean"]
    report["f9_check"] = {
        "band": band, "baseline_nc": base,
        "inside": bool(band[0] <= base <= band[1]),
        "leakage_gate_fired": bool(
            base > f9["leakage_gate_numerai_corr"]
            or levels["filter_off"]["spearman_3seed_mean"]
            > f9["leakage_gate_spearman"])}
    print(f"\nF9: GRU baseline nc {base:+.5f} vs band [{band[0]:.4f}, "
          f"{band[1]:.4f}] -> "
          f"{'INSIDE' if report['f9_check']['inside'] else 'OUTSIDE'}; "
          f"leakage gate fired: {report['f9_check']['leakage_gate_fired']}")

    # shared bootstrap index matrix (same resamples for all paired stats)
    idx = block_idx(n, L, N_BOOT, rng)

    # --- headline + attribution paired comparisons ---
    pairs = [("filter_on", "filter_off"), ("c4_random", "filter_off"),
             ("filter_on", "c4_random")]
    print("\n--- paired per-era diffs, moving-block bootstrap "
          f"(L={L}, {N_BOOT} resamples, {CI:.0f}% CI) ---")
    comps = {}
    for a, b in pairs:
        d_nc = arm_nc[a] - arm_nc[b]
        d_sp = arm_sp[a] - arm_sp[b]
        m, lo, hi = boot_ci_mean(d_nc, idx)
        ms, los, his = boot_ci_mean(d_sp, idx)
        v = verdict_of(m, lo, hi, THR_NC)
        comps[f"{a}_minus_{b}"] = {
            "nc": {"mean": m, "ci": [lo, hi], "category_vs_F3": v},
            "spearman": {"mean": ms, "ci": [los, his],
                         "category_vs_F3": verdict_of(ms, los, his, THR_SP)}}
        print(f"  {a} - {b}: nc {m:+.5f} CI [{lo:+.5f}, {hi:+.5f}] "
              f"-> {v}; sp {ms:+.5f} CI [{los:+.5f}, {his:+.5f}]")
    report["comparisons"] = comps

    # per-seed headline diffs (same seed on vs off)
    per_seed = {}
    for s in SEEDS:
        d = (nc[f"filter_on_s{s}"] - nc[f"filter_off_s{s}"]).to_numpy()
        m, lo, hi = boot_ci_mean(d, idx)
        per_seed[f"seed{s}"] = {"mean": m, "ci": [lo, hi]}
        print(f"  headline per-seed {s}: nc diff {m:+.5f} "
              f"CI [{lo:+.5f}, {hi:+.5f}]")
    report["headline_per_seed"] = per_seed

    # --- THE VERDICT (GRU arm) ---
    h = comps["filter_on_minus_filter_off"]["nc"]
    verdict = h["category_vs_F3"]
    f12_applied = False
    wording = verdict
    if verdict == "hurts":
        f12_applied = True
        wording = ("no evidence of benefit under the affordable tuning "
                   "budget (F12 downgrade of 'hurts')")
    report["verdict"] = {"category": verdict, "f12_downgrade_applied":
                         f12_applied, "wording": wording,
                         "mean_diff_nc": h["mean"], "ci_nc": h["ci"],
                         "threshold_nc": THR_NC}
    print(f"\n=== GRU-ARM VERDICT (filter_on vs filter_off, F3={THR_NC}): "
          f"{verdict}{' -> F12 DOWNGRADE: ' + wording if f12_applied else ''}"
          f" ===")
    print(f"    mean paired diff {h['mean']:+.5f}, "
          f"CI [{h['ci'][0]:+.5f}, {h['ci'][1]:+.5f}]")

    # --- F10 corr-Sharpe via same machinery (joint block resampling) ---
    on, off = arm_nc["filter_on"], arm_nc["filter_off"]
    on_r, off_r = on[idx], off[idx]  # joint resamples: same era blocks
    sh_diff = (on_r.mean(axis=1) / on_r.std(axis=1, ddof=1)
               - off_r.mean(axis=1) / off_r.std(axis=1, ddof=1))
    lo, hi = np.percentile(sh_diff, [(100 - CI) / 2, 100 - (100 - CI) / 2])
    sh_point = sharpe(on) - sharpe(off)
    report["f10_corr_sharpe"] = {
        "sharpe_on": sharpe(on), "sharpe_off": sharpe(off),
        "diff_point": float(sh_point), "diff_ci": [float(lo), float(hi)],
        "ci_excludes_zero": bool(lo > 0 or hi < 0)}
    print(f"\nF10 corr-Sharpe: on {sharpe(on):.3f} off {sharpe(off):.3f} "
          f"diff {sh_point:+.3f} CI [{lo:+.3f}, {hi:+.3f}] "
          f"(excludes 0: {report['f10_corr_sharpe']['ci_excludes_zero']})")

    # --- F8 tail-era breakdown ---
    print("\n--- F8 tail-era breakdown (quartiles; iid bootstrap within "
          "bucket, descriptive) ---")
    disp = (pd.read_parquet(DATA / "shard_verdict.parquet",
                            columns=["era", "target"])
            .assign(era=lambda d: d["era"].astype(str))
            .groupby("era")["target"].std())
    disp = disp.reindex(nc["era"].astype(str).str.zfill(4)).to_numpy()
    d_nc = arm_nc["filter_on"] - arm_nc["filter_off"]
    f8 = {}
    for split_name, key in (("baseline_corr", arm_nc["filter_off"]),
                            ("target_dispersion", disp)):
        qs = np.quantile(key, [0.25, 0.5, 0.75])
        bucket = np.digitize(key, qs)  # 0..3, Q1=lowest
        rows = []
        for q in range(4):
            m = bucket == q
            dd = d_nc[m]
            bs = rng.choice(dd, size=(2000, len(dd)), replace=True).mean(axis=1)
            lo_, hi_ = np.percentile(bs, [2.5, 97.5])
            rows.append({"quartile": f"Q{q+1}", "n_eras": int(m.sum()),
                         "split_var_mean": float(np.mean(key[m])),
                         "mean_diff_nc": float(dd.mean()),
                         "iid_ci": [float(lo_), float(hi_)]})
            print(f"  {split_name} Q{q+1} (n={m.sum()}): "
                  f"diff {dd.mean():+.5f} CI~[{lo_:+.5f}, {hi_:+.5f}]")
        f8[split_name] = rows
    report["f8_tail_breakdown"] = f8
    report["f8_note"] = ("bucketing by realized baseline corr induces "
                         "regression-to-the-mean in the on-off diff; "
                         "descriptive only. Within-bucket eras are "
                         "non-contiguous -> iid bootstrap, labeled.")

    # --- diagnostics engagement check ---
    print("\n--- engagement diagnostics (filtered arms, verdict runs) ---")
    eng = {}
    for tag, d in diag["runs"].items():
        log = d.get("log", [])
        if not log:
            continue
        ks = np.array([e["k"] for e in log])
        ratios = np.array([e["ratio"] for e in log])
        B = diag["frozen"]["batch"]
        entry = {"n_logged": len(log),
                 "frac_steps_0<k<B": float(np.mean((ks > 0) & (ks < B))),
                 "k_median": float(np.median(ks)),
                 "ratio_median": float(np.median(ratios)),
                 "ratio_min": float(ratios.min()),
                 "ratio_max": float(ratios.max())}
        if "kept_energy_frac" in log[0]:
            kef = np.array([e["kept_energy_frac"] for e in log])
            entry["kept_energy_frac_median"] = float(np.median(kef))
            entry["kept_energy_frac_range"] = [float(kef.min()),
                                               float(kef.max())]
        cosv = np.array([e["cos_vs_mean"] for e in log])
        entry["cos_vs_mean_median"] = float(np.median(cosv))
        eng[tag] = entry
    report["engagement"] = eng
    fon = [v for t, v in eng.items() if t.startswith("filter_on")]
    engaged = all(v["frac_steps_0<k<B"] == 1.0 and
                  v["kept_energy_frac_median"] < 0.995 and
                  v["ratio_median"] > 0.01 for v in fon)
    report["engagement_ok"] = bool(engaged)
    for t, v in sorted(eng.items()):
        print(f"  {t}: k_med {v['k_median']:.0f}, ratio_med "
              f"{v['ratio_median']:.3f}, cos_med "
              f"{v['cos_vs_mean_median']:.3f}"
              + (f", kept_energy_med {v['kept_energy_frac_median']:.3f}"
                 if "kept_energy_frac_median" in v else ""))
    print(f"engagement_ok (filter engaged, selective, non-degenerate "
          f"throughout): {engaged}")

    # --- architecture-consistency comparison vs exp-004 MLP ---
    print("\n--- architecture consistency: GRU (this exp) vs MLP (exp-004) ---")
    consistency = None
    if EXP004.exists():
        mlp = json.load(open(EXP004))
        mh = mlp["comparisons"]["filter_on_minus_filter_off"]["nc"]
        mc4 = mlp["comparisons"]["filter_on_minus_c4_random"]["nc"]
        gc4 = comps["filter_on_minus_c4_random"]["nc"]
        rows = [
            ("baseline nc (filter_off)",
             mlp["levels"]["filter_off"]["numerai_corr_3seed_mean"],
             levels["filter_off"]["numerai_corr_3seed_mean"]),
            ("filter_on nc",
             mlp["levels"]["filter_on"]["numerai_corr_3seed_mean"],
             levels["filter_on"]["numerai_corr_3seed_mean"]),
            ("c4_random nc",
             mlp["levels"]["c4_random"]["numerai_corr_3seed_mean"],
             levels["c4_random"]["numerai_corr_3seed_mean"]),
            ("headline diff (on-off)", mh["mean"], h["mean"]),
            ("on-c4 diff", mc4["mean"], gc4["mean"]),
        ]
        print(f"  {'quantity':26s} {'MLP (exp-004)':>14s} {'GRU (exp-005)':>14s}")
        for name_, mv, gv in rows:
            print(f"  {name_:26s} {mv:+14.5f} {gv:+14.5f}")
        print(f"  MLP headline: {mh['mean']:+.5f} CI [{mh['ci'][0]:+.5f}, "
              f"{mh['ci'][1]:+.5f}] -> {mh['category_vs_F3']}")
        print(f"  GRU headline: {h['mean']:+.5f} CI [{h['ci'][0]:+.5f}, "
              f"{h['ci'][1]:+.5f}] -> {h['category_vs_F3']}")
        same_cat = mh["category_vs_F3"] == h["category_vs_F3"]
        same_dir = (mh["mean"] < 0) == (h["mean"] < 0)
        print(f"  same category: {same_cat}; same direction: {same_dir}")
        consistency = {
            "mlp_headline": mh, "gru_headline": h,
            "mlp_on_minus_c4": mc4, "gru_on_minus_c4": gc4,
            "mlp_levels": {a: mlp["levels"][a]["numerai_corr_3seed_mean"]
                           for a in ARMS},
            "gru_levels": {a: levels[a]["numerai_corr_3seed_mean"]
                           for a in ARMS},
            "same_category": bool(same_cat),
            "same_direction": bool(same_dir)}
    else:
        print("  exp-004 verdict_analysis.json NOT FOUND — no comparison")
    report["exp004_consistency"] = consistency

    report["timings_s"] = diag["timings_s"]
    with open(OUT / "seq_analysis.json", "w") as f:
        json.dump(report, f, indent=2, default=float)
    print("\nwrote out/seq_analysis.json")


if __name__ == "__main__":
    main()
