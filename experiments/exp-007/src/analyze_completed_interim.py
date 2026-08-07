#!/usr/bin/env python3
"""Plot only the completed million-step arms for an interim email report."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out"
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

FILES = {
    "AdamW": "adamw-r0-seed20260805-million.json",
    "Spectral top 2,048": "spectral-top-r2048-seed20260805-million.json",
}
COLORS = {"AdamW": "#222222", "Spectral top 2,048": "#7440a4"}
data = {name: json.load(open(OUT / path)) for name, path in FILES.items()}
for payload in data.values():
    assert payload["curve"][-1]["step"] == 1_000_000
    assert len(payload["curve"]) == 102


def save_trajectory(metric: str, filename: str, ylabel: str) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    for name, payload in data.items():
        curve = payload["curve"]
        steps = [row["step"] for row in curve]
        for axis, split in zip(axes, ("train", "valid", "test")):
            axis.plot(steps, [row[split][metric] for row in curve],
                      color=COLORS[name], linewidth=1.8, label=name)
    labels = ("TRAIN monitor", "Full VALID", "Full exploratory TEST")
    for axis, label in zip(axes, labels):
        axis.set_ylabel(f"{label}\n{ylabel}")
        axis.grid(alpha=.2)
        if metric == "mean_per_era_corr":
            axis.axhline(0, color=".6", linewidth=.8)
    axes[0].legend(frameon=False)
    axes[-1].set_xlabel("Training step")
    fig.suptitle("Completed million-step overparameterized MLP arms")
    fig.patch.set_facecolor("#ffffff")
    fig.tight_layout(rect=(0, 0, 1, .97))
    fig.savefig(FIG / filename, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


save_trajectory("mean_per_era_corr", "completed_interim_correlation.png", "correlation")
save_trajectory("mse", "completed_interim_mse.png", "MSE")

rows = []
for name, payload in data.items():
    curve = payload["curve"]
    valid = np.asarray([row["valid"]["mean_per_era_corr"] for row in curve])
    smooth = np.convolve(valid, np.ones(5) / 5, mode="valid")
    index = int(np.argmax(smooth)) + 4
    row = curve[index]
    rows.append((name, row["step"], float(smooth[index - 4]),
                 row["test"]["mean_per_era_corr"]))

x = np.arange(len(rows)); width = .34
fig, axis = plt.subplots(figsize=(8, 4.5))
axis.bar(x - width / 2, [row[2] for row in rows], width,
         label="VALID-selected rolling VALID", color="#4c78a8")
axis.bar(x + width / 2, [row[3] for row in rows], width,
         label="TEST at VALID-selected step", color="#f58518")
axis.set_xticks(x, [row[0] for row in rows])
axis.set_ylabel("Mean per-era correlation")
axis.set_title("Completed arms at validation-selected checkpoints")
axis.grid(axis="y", alpha=.2)
axis.legend(frameon=False)
fig.patch.set_facecolor("#ffffff")
fig.tight_layout()
fig.savefig(FIG / "completed_interim_selection.png", dpi=150,
            bbox_inches="tight", facecolor="white")
plt.close(fig)

for row in rows:
    print(row)
