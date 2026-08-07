#!/usr/bin/env python3
"""Audit and plot the completed million-step full-split experiment."""
from __future__ import annotations

import hashlib
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
EXPECTED_RANKS = {"AdamW": 0, "Top 2048": 2048, "Top 3072": 3072,
                  "Top 4096": 4096}
EXPECTED_STEPS = [100, 1_000, *range(10_000, 1_000_001, 10_000)]


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
audit = {
    "status": "passed",
    "expected": {
        "architecture": [2376, 2048, 1024, 512, 1],
        "parameter_count": 7_491_585,
        "train_rows": 4_017_510,
        "seed": 20260805,
        "learning_rate": 3e-5,
        "evaluation_steps": EXPECTED_STEPS,
        "valid_rows": 578_430,
        "valid_eras": 96,
        "test_rows": 744_354,
        "test_eras": 110,
    },
    "artifacts": {},
}
for name, filename in FILES.items():
    artifact = OUT / filename
    raw = artifact.read_bytes()
    payload = json.loads(raw)
    assert payload["steps"] == 1_000_000
    assert payload["architecture"] == [2376, 2048, 1024, 512, 1]
    assert payload["parameter_count"] == 7_491_585
    assert payload["train_rows"] == 4_017_510
    assert payload["seed"] == 20260805
    assert payload["learning_rate"] == 3e-5
    assert payload["rank"] == EXPECTED_RANKS[name]
    assert payload["paired_batch_stream"] is True
    assert payload["test_touched"] is True
    assert payload["test_role"] == "repeated exploratory trajectory; not an untouched holdout"
    assert payload["valid_scope"] == payload["test_scope"] == "full"
    if name == "AdamW":
        assert payload["arm"] == "adamw"
        assert payload["blockwise"] is False
    else:
        assert payload["arm"] == "spectral"
        assert payload["blockwise"] is True
        assert payload["block_size"] == 250_000
        assert payload["projection_mode"] == "top"
        assert payload["relative_eig_tol"] == 1e-12
    curve = payload["curve"]
    assert [row["step"] for row in curve] == EXPECTED_STEPS
    for row in curve:
        assert row["valid"]["n_rows"] == 578_430 and row["valid"]["n_eras"] == 96
        assert row["test"]["n_rows"] == 744_354 and row["test"]["n_eras"] == 110
        for split in ("train", "valid", "test"):
            assert all(math.isfinite(row[split][key]) for key in ("mse", "mean_per_era_corr"))
        if name != "AdamW":
            filt = row["filter"]
            assert filt["step"] == row["step"]
            assert filt["block_count"] == 30
            assert filt["requested_total_rank"] == EXPECTED_RANKS[name]
            assert filt["realized_total_basis_rank"] == EXPECTED_RANKS[name]
            assert filt["projection_mode"] == "top"
            assert math.isfinite(filt["mean_effective_rank"])
    data[name] = payload
    audit["artifacts"][name] = {
        "file": filename,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
        "evaluation_count": len(curve),
        "terminal_step": curve[-1]["step"],
        "protocol_checks": "passed",
    }

atomic_json(OUT / "million-audit.json", audit)

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
        "train_corr_at_valid_selected_step": selected["train"]["mean_per_era_corr"],
        "test_corr_at_valid_selected_step": selected["test"]["mean_per_era_corr"],
        "test_mse_at_valid_selected_step": selected["test"]["mse"],
        "exploratory_best_test_corr": max(row["test"]["mean_per_era_corr"] for row in curve),
        "raw_best_valid_corr": max(row["valid"]["mean_per_era_corr"] for row in curve),
        "raw_best_valid_step": curve[int(np.argmax(valid))]["step"],
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
    if payload["blockwise"]:
        summary["arms"][name].update({
            "terminal_realized_basis_rank": terminal["filter"]["realized_total_basis_rank"],
            "terminal_mean_effective_rank": terminal["filter"]["mean_effective_rank"],
        })
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
fig.savefig(FIG / "correlation_million.png", dpi=180, facecolor="white")
plt.close(fig)

fig, axes = plt.subplots(3, 1, figsize=(12, 11), sharex=True)
for name, payload in data.items():
    curve = payload["curve"]
    steps = [row["step"] for row in curve]
    for axis, split in zip(axes, ("train", "valid", "test")):
        axis.plot(steps, [row[split]["mse"] for row in curve],
                  color=COLORS[name], linewidth=1.8, label=name)
for axis, title in zip(axes, ("TRAIN monitor MSE", "Full VALID MSE",
                              "Full exploratory TEST MSE")):
    axis.set_title(title)
    axis.set_ylabel("MSE"); axis.grid(alpha=.2)
axes[0].legend(frameon=False, ncol=4)
axes[-1].set_xlabel("Training step")
fig.tight_layout()
fig.savefig(FIG / "mse_million.png", dpi=180, facecolor="white")
plt.close(fig)

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
fig.savefig(FIG / "rank_saturation_million.png", dpi=180, facecolor="white")
plt.close(fig)

fig, axis = plt.subplots(figsize=(11, 5))
for name, payload in data.items():
    if not payload["blockwise"]:
        continue
    curve = payload["curve"]
    axis.plot([row["step"] for row in curve],
              [row["filter"]["mean_effective_rank"] for row in curve],
              color=COLORS[name], linewidth=1.8, label=name)
axis.set_title("Mean numerically effective covariance rank per parameter block")
axis.set_xlabel("Training step")
axis.set_ylabel("Mean effective rank")
axis.grid(alpha=.2); axis.legend(frameon=False)
fig.tight_layout()
fig.savefig(FIG / "effective_rank_million.png", dpi=180, facecolor="white")
plt.close(fig)

print(json.dumps(summary, indent=2))
