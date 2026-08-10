#!/usr/bin/env python3
"""Create the final exp-006 report and figures."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT, FIG = ROOT/"out", ROOT/"figures"
FIG.mkdir(exist_ok=True)
sweep = json.load(open(OUT/"rank-sweep-summary.json"))
comp = json.load(open(OUT/"comparison.json"))
mech = json.load(open(OUT/"numerai-mechanism-diagnostic.json"))
adamw_valid = 0.013102348087525554

plt.style.use("seaborn-v0_8-whitegrid")
ranks = sweep["ranks"]
means = [sweep["by_rank"][str(r)]["mean_valid"] for r in ranks]
stds = [sweep["by_rank"][str(r)]["std_valid"] for r in ranks]
fig, ax = plt.subplots(figsize=(10, 6))
ax.errorbar(ranks, means, yerr=stds, marker="o", capsize=5, lw=2.5, label="Stable spectral (3 seeds)")
ax.axhline(adamw_valid, color="black", ls="--", lw=2, label="Selected AdamW VALID")
ax.axvspan(15.6, 16.4, color="#ef8a62", alpha=.25, label="Selected rank")
ax.set(xlabel="Maximum retained eigenvectors", ylabel="Full-VALID mean per-era correlation",
       title="Corrected p×p spectral-filter rank response")
ax.set_xticks(ranks); ax.legend(); fig.tight_layout(); fig.savefig(FIG/"rank_response_fixed.png", dpi=180); plt.close(fig)

per = comp["per_seed"]
fig, ax = plt.subplots(figsize=(9, 6))
for row in per:
    ax.plot([0, 1], [row["adamw"], row["spectral"]], color="#9b9b9b", lw=2)
ax.scatter(np.zeros(len(per)), [x["adamw"] for x in per], s=90, color="#2878b5", zorder=3)
ax.scatter(np.ones(len(per)), [x["spectral"] for x in per], s=90, color="#e36a22", zorder=3)
ax.set_xticks([0,1], ["AdamW", "Corrected spectral"]); ax.set_ylabel("TEST mean per-era correlation")
ax.set_title("Paired held-out performance: every seed worsened")
fig.tight_layout(); fig.savefig(FIG/"paired_test_fixed.png", dpi=180); plt.close(fig)

era = comp["per_era"]
x = np.array([r["era"] for r in era]); d = np.array([r["mean_difference"] for r in era])
cum = np.cumsum(d)/np.arange(1, len(d)+1)
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(x, cum, color="#159570", lw=3); ax.axhline(0, color="black", lw=1.5)
ax.axhline(comp["B_minus_A"], color="#d55e00", ls="--", lw=2, label="Final mean")
ax.set(xlabel="TEST era", ylabel="Cumulative mean spectral − AdamW",
       title="Out-of-sample effect of the corrected spectral filter")
ax.legend(); fig.tight_layout(); fig.savefig(FIG/"cumulative_test_fixed.png", dpi=180); plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
cfgs = [f"rank={r},tol=1e-08" for r in (8,32,128,512,2048)]
ks = [mech[c]["k"]["median"] for c in cfgs]
orth = [mech[c]["orthogonality_error"]["max"] for c in cfgs]
axes[0].plot([8,32,128,512,2048], ks, marker="o", lw=2.5)
axes[0].set(xlabel="Nominal rank", ylabel="Median realised rank", title="Relative cutoff limits numerical rank")
axes[0].set_xscale("log", base=2)
axes[1].plot([8,32,128,512,2048], orth, marker="o", lw=2.5, color="#7b3294")
axes[1].axhline(5e-3, color="#b42318", ls="--", label="repair threshold")
axes[1].set(xlabel="Nominal rank", ylabel="Maximum ||VᵀV − I||₂", title="Corrected basis remains orthonormal")
axes[1].set_xscale("log", base=2); axes[1].set_yscale("log"); axes[1].legend()
fig.tight_layout(); fig.savefig(FIG/"numerical_diagnostics_fixed.png", dpi=180); plt.close(fig)

ci = comp["intervals"][0]
lines = ["# exp-006 — stable spectral filter", "", "## Conclusions", "",
    "1. **The numerical correction works.** Exact covariance, scale-invariance, resume, fp32 stress, and actual-gradient orthogonality gates all passed.",
    "2. **The corrected hard spectral filter does not help this Numerai MLP.** Rank 16 was selected on VALID, but all spectral ranks remained below AdamW.",
    f"3. On held-out TEST, AdamW scored **{comp['arm_A_mean']:+.6f}** and corrected spectral scored **{comp['arm_B_mean']:+.6f}**; the paired difference was **{comp['B_minus_A']:+.6f}**, 95% block-bootstrap CI **[{ci['lo']:+.6f}, {ci['hi']:+.6f}]**.",
    "", "## VALID rank sweep", "", "| Rank | Mean | Seed SD | Realised k | Retained norm |", "|---:|---:|---:|---:|---:|"]
for r in ranks:
    z=sweep["by_rank"][str(r)]; lines.append(f"| {r} | {z['mean_valid']:+.6f} | {z['std_valid']:.6f} | {z['realized_k_median']:.1f} | {z['median_kept_norm']:.6f} |")
lines += ["", f"AdamW selected VALID score: **{adamw_valid:+.6f}**.", "", "## Paired TEST", "", "| Seed | AdamW | Spectral | Difference |", "|---:|---:|---:|---:|"]
for z in per: lines.append(f"| {z['seed']} | {z['adamw']:+.6f} | {z['spectral']:+.6f} | {z['difference']:+.6f} |")
lines += ["", "## Scope", "", "This establishes that the original implementation had a serious numerical error and that the local replacement fixes it. It also shows that the intended hard top-k filter, with this MLP and fold, is inferior to AdamW. It does not test soft spectral weighting, alternative decay, or other datasets.", "", "## Figures", "", "- `figures/rank_response_fixed.png`", "- `figures/paired_test_fixed.png`", "- `figures/cumulative_test_fixed.png`", "- `figures/numerical_diagnostics_fixed.png`", ""]
(ROOT/"results.md").write_text("\n".join(lines))
print("\n".join(lines[:12]))
