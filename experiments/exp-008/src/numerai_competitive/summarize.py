"""Aggregate a completed paired stage into auditable tables and plots."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

TASK = re.compile(
    r"stage-(?P<split>.+)-u(?P<updates>\d+)-s(?P<seed>\d+)-"
    r"(?P<arm>adamw|spectral)-c(?P<config_id>\d+)"
)


def collect_stage(results: Path, *, split: str, updates: int, seed: int,
                  expected_configs: int = 40) -> pd.DataFrame:
    rows = []
    for result_path in sorted(results.glob("*/result.json")):
        match = TASK.fullmatch(result_path.parent.name)
        if not match or (match["split"], int(match["updates"]), int(match["seed"])) != (
                split, updates, seed):
            continue
        result = json.loads(result_path.read_text())
        rows.append({
            "arm": match["arm"], "config_id": int(match["config_id"]),
            "split": split, "updates": updates, "seed": seed,
            "corr_mean": result["validation"]["corr"]["mean"],
            "corr_sharpe": result["validation"]["corr"]["sharpe"],
            "bmc_mean": result["validation"]["bmc"]["mean"],
            "parameters": result["parameter_count"],
            "peak_cuda_gib": result["peak_cuda_memory_bytes"] / 2**30,
            "elapsed_seconds": result["logs"][-1]["elapsed_seconds"],
        })
    frame = pd.DataFrame(rows)
    if frame.duplicated(["arm", "config_id"]).any():
        raise ValueError("duplicate arm/config results")
    counts = frame.groupby("arm")["config_id"].nunique().to_dict()
    expected = {"adamw": expected_configs, "spectral": expected_configs}
    if counts != expected:
        raise ValueError(f"incomplete stage: got {counts}, expected {expected}")
    return frame.sort_values(["arm", "config_id"]).reset_index(drop=True)


def write_summary(frame: pd.DataFrame, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "scores.csv", index=False)
    paired = frame.pivot(index="config_id", columns="arm", values="corr_mean")
    paired["spectral_minus_adamw"] = paired["spectral"] - paired["adamw"]
    paired.to_csv(output / "paired-corr.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].scatter(paired["adamw"], paired["spectral"], alpha=0.8)
    low, high = paired[["adamw", "spectral"]].min().min(), paired[["adamw", "spectral"]].max().max()
    axes[0].plot([low, high], [low, high], "k--", linewidth=1)
    axes[0].set(xlabel="AdamW validation CORR", ylabel="Spectral validation CORR",
                title="Paired configurations")
    paired["spectral_minus_adamw"].sort_values().plot.bar(ax=axes[1], color="#4c78a8")
    axes[1].axhline(0, color="black", linewidth=1)
    axes[1].set(ylabel="Spectral − AdamW CORR", title="Per-configuration delta")
    fig.tight_layout()
    fig.savefig(output / "paired-corr.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--updates", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--expected-configs", type=int, default=40)
    args = parser.parse_args()
    write_summary(collect_stage(args.results, split=args.split, updates=args.updates,
                                seed=args.seed, expected_configs=args.expected_configs), args.output)


if __name__ == "__main__":
    main()
