#!/usr/bin/env python3
"""Local CPU analysis of the exp-001 debug-job outputs.

Establishes, per (mode, B, composition) config:
  - sustained-selectivity stats for the headline #1 criterion,
  - C3 cosine distributions,
  - C2 real-vs-permuted comparison (hard mode),
  - C5 era-identity probe summary,
and prints everything as a machine-readable JSON + human-readable tables.

Interpretation notes (from src/spectral_optimizer.py):
  - k: number of eigenvalues of the row-normalized similarity matrix
    S = Gn Gn^T above the MP-style threshold 2 * (trace/B) ~= 2.0
    (hard/soft modes; clamped to >=1 in hard mode), or the number of
    directions holding 90% cumulative variance (variance mode).
  - consensus_ratio: ||filtered_grad|| / ||mean_grad||. NOT bounded by 1:
    the filter projects the uniform weight vector onto the kept eigen-
    subspace and applies it to the UNnormalized G, so removing anti-
    aligned (cancelling) components can RAISE the norm. The bounded
    "fraction of gradient energy kept" is computed here from the saved
    full eigenspectra instead: sum(eigs > thr) / sum(eigs).
"""
import json
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent.parent
OUT = HERE / "cluster" / "out"

runs = json.load(open(OUT / "engagement.json"))
spectra = np.load(OUT / "spectra.npz")

LATE = 100  # steps >= LATE are "post-transient"


def arr(log, key):
    return np.array([e[key] for e in log if key in e], dtype=float)


def steps_of(log, key):
    return np.array([e["step"] for e in log if key in e])


def frac(x, lo, hi):
    return float(np.mean((x >= lo) & (x <= hi)))


def summarize(r):
    log = r["log"]
    B = r["B"]
    k = arr(log, "k")
    ratio = arr(log, "consensus_ratio")
    cosF = arr(log, "cos_grad_filtered_vs_mean")
    cosU = arr(log, "cos_update_filtered_vs_unfiltered")
    steps = arr(log, "step")
    late = steps >= LATE
    top1 = np.array([e["top_eigs"][0] for e in log])

    s = {
        "tag": r["tag"], "mode": r["mode"], "B": B,
        "comp": r["composition"], "permuted": r["permuted"],
        "n_logged": len(log),
        # selectivity
        "frac_steps_0<k<B": frac(k, 1, B - 1) if r["mode"] != "soft"
                            else float(np.mean((k > 0) & (k < B))),
        "frac_steps_k==0": float(np.mean(k == 0)),
        "frac_steps_k==B": float(np.mean(k == B)),
        "k_median_all": float(np.median(k)),
        "k_median_late": float(np.median(k[late])),
        "k_min": int(k.min()), "k_max": int(k.max()),
        # norm ratio (unbounded)
        "ratio_median_all": float(np.median(ratio)),
        "ratio_median_late": float(np.median(ratio[late])),
        "ratio_iqr_late": [float(np.percentile(ratio[late], 25)),
                           float(np.percentile(ratio[late], 75))],
        "ratio_frac_in_[0.1,0.9]": frac(ratio, 0.1, 0.9),
        "ratio_frac_<0.02": float(np.mean(ratio < 0.02)),
        "ratio_frac_>0.98": float(np.mean(ratio > 0.98)),
        "ratio_frac_>1": float(np.mean(ratio > 1.0)),
        # C3
        "cosF_median": float(np.median(cosF)),
        "cosF_mean": float(np.mean(cosF)),
        "cosF_frac_>0.95": float(np.mean(cosF > 0.95)),
        "cosF_frac_>0.95_late": float(np.mean(cosF[late] > 0.95)),
        "cosF_max": float(cosF.max()),
        "cosU_median": float(np.median(cosU)),
        "cosU_frac_>0.95": float(np.mean(cosU > 0.95)),
        # top eigenvalue vs threshold 2.0
        "top1_median_all": float(np.median(top1)),
        "top1_median_late": float(np.median(top1[late])),
        "top1_min": float(top1.min()),
        "loss_first5_mean": float(np.mean(arr(log, "loss")[:5])),
        "loss_last5_mean": float(np.mean(arr(log, "loss")[-5:])),
    }
    # C5 probes
    if "c5_cos_filtered_sameera" in log[0]:
        same = arr(log, "c5_cos_filtered_sameera")
        other = arr(log, "c5_cos_filtered_othererra")
        msame = arr(log, "c5_cos_mean_sameera")
        mother = arr(log, "c5_cos_mean_otherera")
        s["c5"] = {
            "n_probes": len(same),
            "filtered_sameera_mean": float(np.mean(same)),
            "filtered_otherera_mean": float(np.mean(other)),
            "filtered_gap_mean": float(np.mean(same - other)),
            "filtered_gap_per_probe": [round(float(v), 3)
                                       for v in (same - other)],
            "mean_sameera_mean": float(np.mean(msame)),
            "mean_otherera_mean": float(np.mean(mother)),
            "mean_gap_mean": float(np.mean(msame - mother)),
            "wilcoxon_p_filtered_same_vs_other":
                float(stats.wilcoxon(same, other).pvalue)
                if len(same) >= 6 else None,
        }
    return s


summaries = [summarize(r) for r in runs]

# ---------- spectra-derived kept-energy fraction ----------
spec_stats = {}
for key in spectra.files:
    ev = spectra[key]
    B = len(ev)
    thr = 2.0 * ev.sum() / B
    kept = ev[ev > thr]
    spec_stats[key] = {
        "B": B,
        "top1": float(ev[0]), "top2": float(ev[1]), "top5": float(ev[4]),
        "thr": float(thr),
        "n_above_thr": int(len(kept)),
        "energy_frac_above_thr": float(kept.sum() / ev.sum()),
        "top1_margin_over_thr": float(ev[0] / thr),
    }

# ---------- C2: real vs permuted (hard mode) ----------
c2 = {}
for B in (256, 1024):
    for comp in ("within", "mixed"):
        real = next(r for r in runs if r["mode"] == "hard" and r["B"] == B
                    and r["composition"] == comp and not r["permuted"])
        perm = next(r for r in runs if r["mode"] == "hard" and r["B"] == B
                    and r["composition"] == comp and r["permuted"])

        def late_vals(r, key):
            return np.array([e[key] for e in r["log"] if e["step"] >= LATE])

        def late_top1(r):
            return np.array([e["top_eigs"][0] for e in r["log"]
                             if e["step"] >= LATE])

        pair = {}
        for name, f in (("k", lambda r: late_vals(r, "k")),
                        ("ratio", lambda r: late_vals(r, "consensus_ratio")),
                        ("top1", late_top1)):
            a, b = f(real), f(perm)
            u = stats.mannwhitneyu(a, b, alternative="two-sided")
            pair[name] = {
                "real_median": float(np.median(a)),
                "perm_median": float(np.median(b)),
                "real_iqr": [float(np.percentile(a, 25)),
                             float(np.percentile(a, 75))],
                "perm_iqr": [float(np.percentile(b, 25)),
                             float(np.percentile(b, 75))],
                "mannwhitney_p": float(u.pvalue),
            }
        # spectra energy comparison across the 4 snapshots
        for label, r in (("real", real), ("perm", perm)):
            keys = [k for k in spectra.files if k.startswith(r["tag"] + "__")]
            pair[f"{label}_spectra"] = {
                k.split("__")[1]: spec_stats[k] for k in sorted(keys)}
        # loss trajectories
        pair["loss_last5_real"] = float(np.mean(
            [e["loss"] for e in real["log"][-5:]]))
        pair["loss_last5_perm"] = float(np.mean(
            [e["loss"] for e in perm["log"][-5:]]))
        c2[f"B={B}_comp={comp}"] = pair

out = {"per_config": summaries, "c2_hard_real_vs_permuted": c2,
       "spectra_stats": spec_stats}
with open(HERE / "out" / "analysis.json", "w") as f:
    json.dump(out, f, indent=2)

# ---------- human-readable ----------
hdr = (f"{'tag':44s} {'0<k<B':>6s} {'k_med(late)':>11s} {'rat_med':>8s} "
       f"{'rat[.1,.9]':>10s} {'rat>1':>6s} {'cosF_med':>8s} {'cosF>.95':>8s} "
       f"{'cosU>.95':>8s}")
print(hdr)
for s in summaries:
    print(f"{s['tag']:44s} {s['frac_steps_0<k<B']:6.2f} "
          f"{s['k_median_late']:11.1f} {s['ratio_median_late']:8.3f} "
          f"{s['ratio_frac_in_[0.1,0.9]']:10.2f} {s['ratio_frac_>1']:6.2f} "
          f"{s['cosF_median']:8.3f} {s['cosF_frac_>0.95']:8.2f} "
          f"{s['cosU_frac_>0.95']:8.2f}")

print("\n--- C5 era-identity probes (within-era configs) ---")
for s in summaries:
    if "c5" in s:
        c = s["c5"]
        print(f"{s['tag']:44s} filt same {c['filtered_sameera_mean']:+.3f} "
              f"other {c['filtered_otherera_mean']:+.3f} "
              f"gap {c['filtered_gap_mean']:+.3f} "
              f"(mean-grad gap {c['mean_gap_mean']:+.3f}) "
              f"wilcoxon p={c['wilcoxon_p_filtered_same_vs_other']}")

print("\n--- C2 hard mode: real vs permuted (late steps >= 100) ---")
for key, pair in c2.items():
    print(f"\n[{key}]")
    for name in ("k", "ratio", "top1"):
        d = pair[name]
        print(f"  {name:6s} real med {d['real_median']:8.3f} "
              f"IQR {d['real_iqr'][0]:.3f}-{d['real_iqr'][1]:.3f} | "
              f"perm med {d['perm_median']:8.3f} "
              f"IQR {d['perm_iqr'][0]:.3f}-{d['perm_iqr'][1]:.3f} | "
              f"MWU p={d['mannwhitney_p']:.4f}")
    print(f"  loss last5: real {pair['loss_last5_real']:.5f} "
          f"perm {pair['loss_last5_perm']:.5f}")
    for label in ("real", "perm"):
        for st, d in pair[f"{label}_spectra"].items():
            print(f"  {label} {st}: top1 {d['top1']:7.2f} thr {d['thr']:.2f} "
                  f"n>thr {d['n_above_thr']:3d} "
                  f"energyfrac {d['energy_frac_above_thr']:.3f} "
                  f"top1/thr {d['top1_margin_over_thr']:.2f}")
