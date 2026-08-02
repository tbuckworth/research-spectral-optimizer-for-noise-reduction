#!/usr/bin/env python3
"""Headline forest plot for the paper.

Point estimates and moving-block-bootstrap 95% CIs are taken from the run's
primary analysis outputs (exp-004/out/verdict_analysis.json,
exp-005/out/seq_analysis.json) and the audit re-execution
(audit/rerun-exp-004/, results-audit.md re-execution table).
All values are mean per-era numerai_corr differences over the 255
verdict-block eras (3-seed means unless noted).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

F3 = 0.00398

headline = [  # (label, mean, lo, hi)
    ("MLP, seeds 0-2 (exp-004)",             -0.00527, -0.00886, -0.00181),
    ("GRU, seeds 0-2 (exp-005)",             -0.00484, -0.00758, -0.00211),
    ("MLP audit re-run, seeds 10-12\n(producer shard)", -0.00546, -0.00892, -0.00204),
    ("MLP audit re-run, seeds 10-12\n(audit-built shard)", -0.00562, -0.00911, -0.00201),
    ("MLP 6-seed pool (0-2 + 10-12)",        -0.00537, -0.00853, -0.00222),
]

c4 = [
    ("MLP (exp-004)",                        -0.00116, -0.00366,  0.00134),
    ("GRU (exp-005)",                        -0.00011, -0.00233,  0.00215),
    ("MLP re-run (producer shard)",           0.00054, -0.00161,  0.00275),
    ("MLP re-run (audit-built shard)",       -0.00114, -0.00334,  0.00108),
]

fig, axes = plt.subplots(
    2, 1, figsize=(7.2, 5.4), sharex=True,
    gridspec_kw={"height_ratios": [len(headline), len(c4)]},
)

for ax, rows, title, color in [
    (axes[0], headline,
     "Headline: filter-on $-$ filter-off (spectral filter vs. tuned AdamW)",
     "#b2182b"),
    (axes[1], c4,
     "Mechanism control: filter-on $-$ C4 random subspace (norm/$k$-matched)",
     "#2166ac"),
]:
    ys = list(range(len(rows)))[::-1]
    for y, (label, m, lo, hi) in zip(ys, rows):
        ax.plot([lo, hi], [y, y], color=color, lw=2)
        ax.plot([lo, lo], [y - 0.14, y + 0.14], color=color, lw=2)
        ax.plot([hi, hi], [y - 0.14, y + 0.14], color=color, lw=2)
        ax.plot(m, y, "o", color=color, ms=6)
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows], fontsize=8.5)
    ax.axvline(0.0, color="black", lw=0.8)
    ax.axvline(-F3, color="gray", lw=0.8, ls="--")
    ax.axvline(F3, color="gray", lw=0.8, ls="--")
    ax.set_title(title, fontsize=9.5, loc="left")
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.grid(axis="x", color="0.9", lw=0.6)
    ax.set_axisbelow(True)

axes[1].set_xlabel(
    "Difference in mean per-era numerai_corr (255 verdict eras, MBB 95% CI);"
    "\ndashed lines: pre-registered practical-significance threshold "
    "$\\pm F3 = \\pm 0.00398$",
    fontsize=8.5,
)
fig.tight_layout()
fig.savefig("headline_forest.pdf")
print("wrote headline_forest.pdf")
