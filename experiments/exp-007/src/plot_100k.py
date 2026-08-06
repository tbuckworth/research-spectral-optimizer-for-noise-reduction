#!/usr/bin/env python3
"""Reproduce the exp-007 100k trajectory figures from committed JSON outputs."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out"
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

SERIES = {
    "AdamW": "adamw-r0-seed20260805-100k.json",
    "Top 256": "spectral-top-r256-seed20260805-100k.json",
    "Top 512": "spectral-top-r512-seed20260805-100k.json",
    "Top 1024": "spectral-top-r1024-seed20260805-100k.json",
    "Top 2048": "spectral-top-r2048-seed20260805-100k.json",
    "Remove 256": "spectral-remove-r256-seed20260805-100k.json",
    "Remove 1024": "spectral-remove-r1024-seed20260805-100k.json",
    "Remove+norm 256": "spectral-remove-renorm-r256-seed20260805-100k.json",
    "Remove+norm 1024": "spectral-remove-renorm-r1024-seed20260805-100k.json",
}
COLORS = {
    "AdamW": "#111111", "Top 256": "#cdb7ef", "Top 512": "#a77bd4",
    "Top 1024": "#7440a4", "Top 2048": "#3f1765",
    "Remove 256": "#8ec5ee", "Remove 1024": "#1976b6",
    "Remove+norm 256": "#f28b50", "Remove+norm 1024": "#b6401e",
}


def rolling(values, width=10):
    return np.convolve(values, np.ones(width) / width, mode="valid")


data = {name: json.load(open(OUT / file))["curve"] for name, file in SERIES.items()}
fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
for name, curve in data.items():
    steps = np.array([row["step"] for row in curve])
    valid = np.array([row["valid"]["mean_per_era_corr"] for row in curve])
    train = np.array([row["train"]["mean_per_era_corr"] for row in curve])
    axes[0].plot(steps, valid, color=COLORS[name], alpha=.13, linewidth=.8)
    axes[0].plot(steps[9:], rolling(valid), color=COLORS[name], linewidth=2, label=name)
    axes[1].plot(steps, train, color=COLORS[name], linewidth=1.6)
for axis in axes:
    axis.axhline(0, color=".5", linewidth=.8)
    axis.grid(alpha=.2)
axes[0].set_title("100k-step overparameterized MLP trajectories (5k-step rolling validation mean)")
axes[0].set_ylabel("Validation mean per-era correlation")
axes[1].set_ylabel("Train-monitor mean per-era correlation")
axes[1].set_xlabel("Training step")
axes[0].legend(ncol=3, fontsize=8, frameon=False)
fig.tight_layout()
fig.savefig(FIG / "correlation_100k.png", dpi=180)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
for name, curve in data.items():
    steps = [row["step"] for row in curve]
    axes[0].plot(steps, [row["train"]["mse"] for row in curve], color=COLORS[name], label=name)
    axes[1].plot(steps, [row["valid"]["mse"] for row in curve], color=COLORS[name], label=name)
axes[0].set_title("Train MSE")
axes[1].set_title("Validation MSE")
for axis in axes:
    axis.set_xlabel("Training step")
    axis.set_ylabel("MSE")
    axis.set_ylim(.025, .075)
    axis.grid(alpha=.2)
axes[1].legend(ncol=2, fontsize=7, frameon=False)
fig.tight_layout()
fig.savefig(FIG / "mse_100k.png", dpi=180)
