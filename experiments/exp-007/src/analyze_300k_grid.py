#!/usr/bin/env python3
"""Audit and plot the revised common-horizon 300k rank grid."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out"
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

ARMS = {
    "AdamW": (OUT / "adamw-r0-seed20260805-million.json", 0, "L40"),
    "Top 1,024": (OUT / "spectral-top-r1024-seed20260805-300k-3090.json", 1024, "RTX 3090"),
    "Top 1,536": (OUT / "spectral-top-r1536-seed20260805-300k-3090.json", 1536, "RTX 3090"),
    "Top 2,048": (OUT / "spectral-top-r2048-seed20260805-million.json", 2048, "L40"),
    "Top 3,072": (OUT / "recovery/spectral-top-r3072-seed20260805-stopped-760k.json", 3072, "L40"),
    "Top 4,096": (OUT / "recovery/spectral-top-r4096-seed20260805-pre-second-handoff-550k.json", 4096, "L40"),
}
COLORS = {
    "AdamW": "#222222", "Top 1,024": "#4c78a8", "Top 1,536": "#2ca02c",
    "Top 2,048": "#7440a4", "Top 3,072": "#d95f02", "Top 4,096": "#b2182b",
}
EXPECTED_STEPS = [100, 1_000, *range(10_000, 300_001, 10_000)]


def atomic_json(path: Path, value: object) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as handle:
        json.dump(value, handle, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def rolling(values, width=5):
    return np.convolve(np.asarray(values, dtype=float), np.ones(width) / width, mode="valid")


data = {}
audit = {
    "status": "passed",
    "scope": "common trajectory through step 300000",
    "evaluation_steps": EXPECTED_STEPS,
    "hardware_caveat": (
        "Ranks 1024 and 1536 ran concurrently on RTX 3090; all other arms ran on L40. "
        "Small between-rank differences are not clean rank-only causal estimates."
    ),
    "artifacts": {},
}

for name, (path, rank, hardware) in ARMS.items():
    raw = path.read_bytes()
    payload = json.loads(raw)
    assert payload["architecture"] == [2376, 2048, 1024, 512, 1]
    assert payload["parameter_count"] == 7_491_585
    assert payload["train_rows"] == 4_017_510
    assert payload["seed"] == 20260805
    assert payload["learning_rate"] == 3e-5
    assert payload["rank"] == rank
    assert payload["paired_batch_stream"] is True
    assert payload["test_touched"] is True
    assert payload["test_role"] == "repeated exploratory trajectory; not an untouched holdout"
    assert payload["valid_scope"] == payload["test_scope"] == "full"
    if rank == 0:
        assert payload["arm"] == "adamw" and payload["blockwise"] is False
    else:
        assert payload["arm"] == "spectral" and payload["blockwise"] is True
        assert payload["block_size"] == 250_000
        assert payload["projection_mode"] == "top"
        assert payload["relative_eig_tol"] == 1e-12
    curve = [row for row in payload["curve"] if row["step"] <= 300_000]
    assert [row["step"] for row in curve] == EXPECTED_STEPS
    if hardware == "RTX 3090":
        assert payload["steps"] == 300_000
        assert payload["run_tag"] == "300k-3090"
        assert payload["curve"][-1]["step"] == 300_000
    for row in curve:
        assert row["valid"]["n_rows"] == 578_430 and row["valid"]["n_eras"] == 96
        assert row["test"]["n_rows"] == 744_354 and row["test"]["n_eras"] == 110
        for split in ("train", "valid", "test"):
            assert all(math.isfinite(row[split][key]) for key in ("mse", "mean_per_era_corr"))
        if rank:
            filt = row["filter"]
            assert filt["step"] == row["step"]
            assert filt["block_count"] == 30
            assert filt["requested_total_rank"] == rank
            assert filt["realized_total_basis_rank"] == rank
            assert filt["projection_mode"] == "top"
            assert math.isfinite(filt["mean_effective_rank"])
    data[name] = {"payload": payload, "curve": curve, "hardware": hardware}
    audit["artifacts"][name] = {
        "file": str(path.relative_to(ROOT)),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
        "rank": rank,
        "hardware": hardware,
        "evaluation_count_used": len(curve),
        "source_terminal_step": payload["curve"][-1]["step"],
        "checks": "passed",
    }

summary = {
    "selection_rule": "maximum five-checkpoint rolling VALID correlation through step 300000; TEST not used",
    "test_role": "repeated exploratory trajectory, not an untouched holdout",
    "hardware_caveat": audit["hardware_caveat"],
    "arms": {},
}
for name, item in data.items():
    curve = item["curve"]
    valid = [row["valid"]["mean_per_era_corr"] for row in curve]
    smoothed = rolling(valid)
    index = int(np.argmax(smoothed)) + 4
    selected, terminal = curve[index], curve[-1]
    arm = {
        "rank": item["payload"]["rank"],
        "hardware": item["hardware"],
        "selected_step": selected["step"],
        "selected_smoothed_valid_corr": float(smoothed[index - 4]),
        "selected_raw_valid_corr": selected["valid"]["mean_per_era_corr"],
        "train_corr_at_valid_selected_step": selected["train"]["mean_per_era_corr"],
        "test_corr_at_valid_selected_step": selected["test"]["mean_per_era_corr"],
        "terminal_train_corr": terminal["train"]["mean_per_era_corr"],
        "terminal_valid_corr": terminal["valid"]["mean_per_era_corr"],
        "terminal_test_corr": terminal["test"]["mean_per_era_corr"],
        "terminal_valid_mse": terminal["valid"]["mse"],
        "terminal_test_mse": terminal["test"]["mse"],
    }
    if arm["rank"]:
        arm["terminal_mean_effective_rank"] = terminal["filter"]["mean_effective_rank"]
    summary["arms"][name] = arm

atomic_json(OUT / "rank-grid-300k-audit.json", audit)
atomic_json(OUT / "rank-grid-300k-summary.json", summary)


def plot_trajectories(metric, filename, ylabel):
    fig, axes = plt.subplots(3, 1, figsize=(12, 11), sharex=True)
    for name, item in data.items():
        curve = item["curve"]
        steps = [row["step"] for row in curve]
        style = "--" if item["hardware"] == "RTX 3090" else "-"
        label = f"{name} ({item['hardware']})"
        for axis, split in zip(axes, ("train", "valid", "test")):
            axis.plot(steps, [row[split][metric] for row in curve], color=COLORS[name],
                      linestyle=style, linewidth=1.65, label=label)
    labels = ("TRAIN monitor", "Full VALID", "Full exploratory TEST")
    for axis, split in zip(axes, labels):
        axis.set_ylabel(f"{split}\n{ylabel}")
        axis.grid(alpha=.2)
        if metric == "mean_per_era_corr":
            axis.axhline(0, color=".6", linewidth=.8)
    axes[0].legend(frameon=False, ncol=2, fontsize=9)
    axes[-1].set_xlabel("Training step")
    fig.suptitle("Overparameterized Numerai MLP — common 300k horizon")
    fig.patch.set_facecolor("white")
    fig.tight_layout(rect=(0, 0, 1, .97))
    fig.savefig(FIG / filename, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


plot_trajectories("mean_per_era_corr", "correlation_rank_grid_300k.png", "correlation")
plot_trajectories("mse", "mse_rank_grid_300k.png", "MSE")

names = list(ARMS)
x = np.arange(len(names)); width = .36
fig, axis = plt.subplots(figsize=(11, 5.5))
axis.bar(x - width / 2,
         [summary["arms"][name]["selected_smoothed_valid_corr"] for name in names], width,
         label="VALID-selected rolling VALID", color="#4c78a8")
axis.bar(x + width / 2,
         [summary["arms"][name]["test_corr_at_valid_selected_step"] for name in names], width,
         label="Exploratory TEST at selected step", color="#f58518")
axis.set_xticks(x, [name.replace("Top ", "Top\n") for name in names])
axis.set_ylabel("Mean per-era correlation")
axis.set_title("Rank comparison at validation-selected checkpoints (0–300k)")
axis.grid(axis="y", alpha=.2); axis.legend(frameon=False)
axis.text(.99, .02, "Dashed trajectory arms (1,024/1,536) ran on RTX 3090; others on L40",
          transform=axis.transAxes, ha="right", va="bottom", fontsize=8, color="#666")
fig.patch.set_facecolor("white"); fig.tight_layout()
fig.savefig(FIG / "rank_saturation_300k.png", dpi=180, bbox_inches="tight", facecolor="white")
plt.close(fig)

fig, axis = plt.subplots(figsize=(11, 5.5))
for name, item in data.items():
    if item["payload"]["rank"] == 0:
        continue
    curve = item["curve"]
    style = "--" if item["hardware"] == "RTX 3090" else "-"
    axis.plot([row["step"] for row in curve],
              [row["filter"]["mean_effective_rank"] for row in curve],
              color=COLORS[name], linestyle=style, linewidth=1.65,
              label=f"{name} ({item['hardware']})")
axis.set_title("Mean numerically effective covariance rank per parameter block")
axis.set_xlabel("Training step"); axis.set_ylabel("Mean effective rank")
axis.grid(alpha=.2); axis.legend(frameon=False, ncol=2)
fig.patch.set_facecolor("white"); fig.tight_layout()
fig.savefig(FIG / "effective_rank_300k.png", dpi=180, bbox_inches="tight", facecolor="white")
plt.close(fig)

print(json.dumps(summary, indent=2))
