#!/usr/bin/env python3
"""Generate the compact evidence figures used by the project presentation."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
FIG = ROOT / "figures"
DATA = ROOT / "data"
EXP = ROOT.parent.parent

BLUE = "#2563eb"
ORANGE = "#f97316"
GREEN = "#16a34a"
RED = "#dc2626"
INK = "#172033"
MUTED = "#64748b"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 13,
    "axes.labelsize": 10,
    "axes.edgecolor": "#cbd5e1",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "text.color": INK,
    "axes.labelcolor": INK,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})


def save(fig: plt.Figure, name: str) -> None:
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / name, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


outer = pd.read_csv(DATA / "outer-results.csv")
summary = outer.groupby("arm").agg(
    corr=("corr_mean", "mean"), sharpe=("corr_sharpe", "mean"), bmc=("bmc_mean", "mean")
)

# Audited headline: absolute levels and per-seed differences.
fig, ax = plt.subplots(figsize=(7.2, 3.7))
vals = [summary.loc["adamw", "corr"], summary.loc["spectral", "corr"]]
bars = ax.bar(["AdamW", "Spectral"], vals, color=[BLUE, ORANGE], width=.58)
ax.set_ylabel("Mean per-era Numerai CORR")
ax.set_ylim(0, .027)
ax.set_title("Untouched outer block: spectral pipeline wins all three seeds")
for bar, value in zip(bars, vals):
    ax.text(bar.get_x() + bar.get_width()/2, value + .00045, f"{value:.5f}", ha="center", weight="bold")
ax.text(.5, .003, "+0.00307 CORR\n+15.5% relative", ha="center", color=GREEN, weight="bold")
ax.grid(axis="y", alpha=.22)
save(fig, "outer-headline.png")

pivot = outer.pivot(index="seed", columns="arm", values="corr_mean")
fig, ax = plt.subplots(figsize=(7.2, 3.7))
for seed, row in pivot.iterrows():
    ax.plot([0, 1], [row.adamw, row.spectral], marker="o", linewidth=2.4, alpha=.85)
    ax.text(-.04, row.adamw, f"s{seed}  {row.adamw:.4f}", ha="right", va="center", fontsize=9)
    ax.text(1.04, row.spectral, f"{row.spectral:.4f}  s{seed}", ha="left", va="center", fontsize=9)
ax.set_xticks([0, 1], ["AdamW config 38", "Spectral config 39"])
ax.set_xlim(-.32, 1.32)
ax.set_ylim(.0175, .0252)
ax.set_ylabel("Mean per-era CORR")
ax.set_title("Paired random seeds: 3 / 3 favour the selected spectral pipeline")
ax.grid(axis="y", alpha=.22)
save(fig, "outer-seeds.png")

# Metric tradeoff.
fig, axes = plt.subplots(1, 3, figsize=(9, 3.2))
for ax, metric, title, fmt in zip(
    axes, ["corr", "sharpe", "bmc"], ["CORR ↑", "CORR Sharpe ↑", "BMC ↑"], [".5f", ".3f", ".5f"]
):
    values = [summary.loc["adamw", metric], summary.loc["spectral", metric]]
    ax.bar(["AdamW", "Spectral"], values, color=[BLUE, ORANGE], width=.62)
    ax.set_title(title, weight="bold")
    ax.grid(axis="y", alpha=.2)
    top = max(values)
    ax.set_ylim(0, top * 1.22)
    for i, value in enumerate(values):
        ax.text(i, value + top*.035, format(value, fmt), ha="center", fontsize=9, weight="bold")
fig.suptitle("The gain is predictive correlation and stability—not benchmark uniqueness", y=1.02, weight="bold")
save(fig, "metric-tradeoff.png")

# Chronological split schematic.
fig, ax = plt.subplots(figsize=(10, 3.2))
segments = [
    (0, 296, "Train\neras 0001–0296", BLUE),
    (296, 16, "16-era purge", "#cbd5e1"),
    (312, 78, "Untouched outer test\neras 0313–0390", ORANGE),
]
for start, width, label, color in segments:
    ax.barh(0, width, left=start, height=.58, color=color, edgecolor="white")
    ax.text(start + width/2, 0, label, ha="center", va="center", fontsize=10,
            color="white" if color in (BLUE, ORANGE) else INK, weight="bold")
ax.annotate("Hyperparameters selected only inside the blue region", xy=(190, .34), xytext=(95, .9),
            arrowprops={"arrowstyle":"->", "color":MUTED}, ha="center", color=MUTED)
ax.annotate("Three frozen seeds; no reselection", xy=(350, .34), xytext=(350, .9),
            arrowprops={"arrowstyle":"->", "color":MUTED}, ha="center", color=MUTED)
ax.set_xlim(0, 390); ax.set_ylim(-.55, 1.2); ax.axis("off")
ax.set_title("Numerai evaluation respects time: whole eras, purge, then later eras", weight="bold")
save(fig, "chronological-split.png")

# Search funnel.
fig, ax = plt.subplots(figsize=(9.2, 3.8))
stages = ["Paired F0 screen", "Equal-coverage confirmation", "Frozen outer test", "Audited result"]
counts = [80, 16, 6, 6]
widths = [8.4, 5.9, 3.8, 2.5]
colors = ["#dbeafe", "#bfdbfe", "#fed7aa", "#bbf7d0"]
for i, (stage, count, width, color) in enumerate(zip(stages, counts, widths, colors)):
    y = 3-i
    ax.barh(y, width, left=(9-width)/2, height=.7, color=color, edgecolor="white")
    ax.text(4.5, y, f"{stage}\n{count} optimizer/config/seed cells", ha="center", va="center", weight="bold")
    if i < 3:
        ax.annotate("", xy=(4.5, y-.55), xytext=(4.5, y-.82), arrowprops={"arrowstyle":"->", "color":MUTED})
ax.set_xlim(0,9); ax.set_ylim(-.6,3.6); ax.axis("off")
ax.set_title("Bounded one-day protocol: selection narrows before untouched outcomes", weight="bold")
save(fig, "search-funnel.png")

# Architecture / compute comparison.
arch = pd.DataFrame({
    "model": ["AdamW c38", "Spectral c39"],
    "parameters_m": [6.799, 10.194],
    "batch": [1024, 2048],
    "examples_m": [20.48, 40.96],
})
fig, axes = plt.subplots(1, 3, figsize=(9.3, 3.1))
for ax, col, title in zip(axes, ["parameters_m", "batch", "examples_m"],
                          ["Parameters (M)", "Batch size", "Examples seen (M)"]):
    vals = arch[col]
    ax.bar(["AdamW", "Spectral"], vals, color=[BLUE, ORANGE], width=.62)
    ax.set_title(title, weight="bold"); ax.grid(axis="y", alpha=.2)
    ax.set_ylim(0, max(vals)*1.25)
    for i,v in enumerate(vals): ax.text(i, v+max(vals)*.04, f"{v:g}", ha="center", weight="bold")
fig.suptitle("The outer comparison estimates best selected pipelines—not an optimizer-only effect", y=1.04, weight="bold")
save(fig, "architecture-comparison.png")

# Earlier rank-grid summary.
rank = json.loads((EXP / "exp-007/out/rank-grid-300k-summary.json").read_text())["arms"]
rows=[]
for label,d in rank.items():
    rows.append({"label":label, "rank":d["rank"], "valid":d["selected_smoothed_valid_corr"],
                 "test":d["test_corr_at_valid_selected_step"], "train":d["train_corr_at_valid_selected_step"]})
rankdf=pd.DataFrame(rows).sort_values("rank")
fig, ax = plt.subplots(figsize=(8.2, 3.7))
x=np.arange(len(rankdf)); w=.34
ax.bar(x-w/2,rankdf.valid,w,label="VALID-selected correlation",color=ORANGE)
ax.bar(x+w/2,rankdf.test,w,label="TEST at selected step",color=BLUE)
ax.set_xticks(x,["AdamW" if r==0 else f"r={r:,}" for r in rankdf["rank"]])
ax.set_ylabel("Mean per-era correlation"); ax.set_ylim(0,.027)
ax.set_title("Exploratory overparameterized study: benefit saturates above rank ≈1,024")
ax.legend(frameon=False,ncols=2); ax.grid(axis="y",alpha=.2)
save(fig,"rank-grid-summary.png")

# Research chronology with signed outcomes; scales intentionally labelled as different studies.
fig, ax = plt.subplots(figsize=(9.2, 3.8))
labels=["Original B×B filter\n(exp-004)","Corrected low-rank p×p\n(exp-006)",
        "Overparameterized high-rank\n(exp-007, exploratory)","Official-data bounded test\n(exp-008)"]
effects=[-.00527,-.01625,.00627,.00307]
colors=[RED,RED,GREEN,GREEN]
y=np.arange(4)
ax.axvline(0,color=INK,lw=1)
ax.barh(y,effects,color=colors,alpha=.88)
ax.set_yticks(y,labels); ax.invert_yaxis(); ax.set_xlabel("Reported spectral − AdamW CORR (study-specific endpoint)")
ax.set_title("Why the conclusion changed: implementation, rank, capacity and protocol all changed")
for yi,v in zip(y,effects): ax.text(v + (.0004 if v>=0 else -.0004),yi,f"{v:+.4f}",va="center",ha="left" if v>=0 else "right",weight="bold")
ax.grid(axis="x",alpha=.2)
save(fig,"research-chronology.png")

# Leaderboard context: deliberately separate metric families.
lb=json.loads((DATA/"leaderboard-summary.json").read_text())["summary"]["metrics"]["corr20V2Rep"]
fig,ax=plt.subplots(figsize=(8.4,3.8))
names=["Leaderboard\nmedian","Leaderboard\np90","Leaderboard\nmaximum","Our AdamW\nhistorical CORR","Our spectral\nhistorical CORR"]
values=[lb["median"],lb["p90"],lb["maximum"],summary.loc["adamw","corr"],summary.loc["spectral","corr"]]
bars=ax.bar(names,values,color=["#94a3b8"]*3+[BLUE,ORANGE])
for i,b in enumerate(bars):
    if i>=3: b.set_hatch("///")
    ax.text(i,values[i]+.00045,f"{values[i]:.4f}",ha="center",fontsize=9,weight="bold")
ax.axvline(2.5,color=INK,ls="--",lw=1)
ax.text(1,.002,"Live CORR20v2 reputation",ha="center",color=MUTED)
ax.text(3.5,.002,"Historical main-target CORR\nNOT rank-comparable",ha="center",color=RED,weight="bold")
ax.set_ylim(0,.027); ax.set_ylabel("Correlation-like score")
ax.set_title("Numerically promising; leaderboard position is unknown without forward submissions")
ax.grid(axis="y",alpha=.2)
save(fig,"leaderboard-context.png")

# Seed uncertainty and honest interpretation.
d=(pivot.spectral-pivot.adamw).to_numpy(); mean=d.mean(); sem=d.std(ddof=1)/np.sqrt(len(d)); half=4.30265273*sem
fig,ax=plt.subplots(figsize=(7.4,3.3))
ax.axvline(0,color=INK,lw=1)
ax.errorbar(mean,0,xerr=half,fmt="o",markersize=9,capsize=7,color=ORANGE,lw=2.4)
ax.scatter(d,[.18,.28,.38],color=GREEN,s=48,zorder=3)
for i,x in enumerate(d): ax.text(x,.45,f"seed {i}: {x:+.4f}",ha="center",fontsize=8)
ax.set_yticks([]); ax.set_ylim(-.25,.65); ax.set_xlim(-.003,.009)
ax.set_xlabel("Spectral − AdamW CORR")
ax.set_title(f"Three consistent wins, but seed-level 95% t interval crosses zero [{mean-half:+.4f}, {mean+half:+.4f}]")
ax.grid(axis="x",alpha=.2)
save(fig,"seed-uncertainty.png")

print(f"generated figures in {FIG}")
