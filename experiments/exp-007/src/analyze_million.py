#!/usr/bin/env python3
"""Audit and plot the completed million-step full-split experiment."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out"
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

FILES = {
    "AdamW": "adamw-r0-seed20260805-million.json",
    "Top 2048": "spectral-top-r2048-seed20260805-million.json",
    "Top 3072": "spectral-top-r3072-seed20260805-million.json",
    "Top 4096": "spectral-top-r4096-seed20260805-million.json",
}
COLORS = {"AdamW": "#111111", "Top 2048": "#7440a4",
          "Top 3072": "#5a2685", "Top 4096": "#351052"}


def atomic_json(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as handle:
        json.dump(value, handle, indent=2)
        handle.flush(); os.fsync(handle.fileno())
    os.replace(tmp, path)


def rolling(values, width=5):
    values = np.asarray(values, dtype=float)
    return np.convolve(values, np.ones(width) / width, mode="valid")


data = {}
for name, filename in FILES.items():
    payload = json.load(open(OUT / filename))
    assert payload["steps"] == 1_000_000
    assert payload["test_touched"] is True
    assert payload["valid_scope"] == payload["test_scope"] == "full"
    curve = payload["curve"]
    assert curve[-1]["step"] == 1_000_000
    assert len({row["step"] for row in curve}) == len(curve)
    for row in curve:
        assert row["valid"]["n_rows"] == 578_430 and row["valid"]["n_eras"] == 96
        assert row["test"]["n_rows"] == 744_354 and row["test"]["n_eras"] == 110
        for split in ("train", "valid", "test"):
            assert all(math.isfinite(row[split][key]) for key in ("mse", "mean_per_era_corr"))
    data[name] = payload

summary = {"selection_rule": "maximum five-checkpoint rolling VALID correlation; TEST not used",
           "test_role": "repeated exploratory trajectory, not an untouched holdout", "arms": {}}
for name, payload in data.items():
    curve = payload["curve"]
    valid = [row["valid"]["mean_per_era_corr"] for row in curve]
    smoothed = rolling(valid, 5)
    selected_index = int(np.argmax(smoothed)) + 4
    selected = curve[selected_index]
    terminal = curve[-1]
    summary["arms"][name] = {
        "rank": payload["rank"],
        "selected_step": selected["step"],
        "selected_smoothed_valid_corr": float(smoothed[selected_index - 4]),
        "selected_raw_valid_corr": selected["valid"]["mean_per_era_corr"],
        "test_corr_at_valid_selected_step": selected["test"]["mean_per_era_corr"],
        "test_mse_at_valid_selected_step": selected["test"]["mse"],
        "exploratory_best_test_corr": max(row["test"]["mean_per_era_corr"] for row in curve),
        "terminal_valid_corr": terminal["valid"]["mean_per_era_corr"],
        "terminal_test_corr": terminal["test"]["mean_per_era_corr"],
        "terminal_train_corr": terminal["train"]["mean_per_era_corr"],
        "terminal_valid_mse": terminal["valid"]["mse"],
        "terminal_test_mse": terminal["test"]["mse"],
        "final_100k_mean_valid_corr": float(np.mean([
            row["valid"]["mean_per_era_corr"] for row in curve if row["step"] >= 900_000])),
        "final_100k_mean_test_corr": float(np.mean([
            row["test"]["mean_per_era_corr"] for row in curve if row["step"] >= 900_000])),
    }
atomic_json(OUT / "million-summary.json", summary)

fig, axes = plt.subplots(3, 1, figsize=(12, 11), sharex=True)
for name, payload in data.items():
    curve = payload["curve"]
    steps = [row["step"] for row in curve]
    for axis, split in zip(axes, ("train", "valid", "test")):
        axis.plot(steps, [row[split]["mean_per_era_corr"] for row in curve],
                  color=COLORS[name], linewidth=1.8, label=name)
for axis, label in zip(axes, ("TRAIN monitor", "Full VALID", "Full exploratory TEST")):
    axis.axhline(0, color=".5", linewidth=.8)
    axis.set_ylabel(f"{label} correlation")
    axis.grid(alpha=.2)
axes[0].set_title("Million-step overparameterized MLP correlation trajectories")
axes[0].legend(frameon=False, ncol=4)
axes[-1].set_xlabel("Training step")
fig.tight_layout()
fig.savefig(FIG / "correlation_million.png", dpi=180)

fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
for name, payload in data.items():
    curve = payload["curve"]
    steps = [row["step"] for row in curve]
    axes[0].plot(steps, [row["valid"]["mse"] for row in curve],
                 color=COLORS[name], linewidth=1.8, label=name)
    axes[1].plot(steps, [row["test"]["mse"] for row in curve],
                 color=COLORS[name], linewidth=1.8)
axes[0].set_title("Full VALID MSE")
axes[1].set_title("Full exploratory TEST MSE")
for axis in axes:
    axis.set_ylabel("MSE"); axis.grid(alpha=.2)
axes[0].legend(frameon=False, ncol=4)
axes[-1].set_xlabel("Training step")
fig.tight_layout()
fig.savefig(FIG / "mse_million.png", dpi=180)

names = list(data)
selected_valid = [summary["arms"][name]["selected_smoothed_valid_corr"] for name in names]
selected_test = [summary["arms"][name]["test_corr_at_valid_selected_step"] for name in names]
x = np.arange(len(names)); width = .36
fig, axis = plt.subplots(figsize=(9, 5))
axis.bar(x - width/2, selected_valid, width, label="VALID-selected rolling VALID")
axis.bar(x + width/2, selected_test, width, label="TEST at VALID-selected step")
axis.set_xticks(x, names)
axis.set_ylabel("Mean per-era correlation")
axis.set_title("Rank saturation at VALID-selected checkpoints")
axis.grid(axis="y", alpha=.2); axis.legend(frameon=False)
fig.tight_layout()
fig.savefig(FIG / "rank_saturation_million.png", dpi=180)

print(json.dumps(summary, indent=2))
