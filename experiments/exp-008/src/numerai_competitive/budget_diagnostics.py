"""Create audited development-only training-budget tables and plots."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .data import atomic_json, sha256
from .select_budgeted_configs import restrict_candidates, select_budgeted_configs


def build_budget_diagnostics(scores: pd.DataFrame, config_ids: list[int], output: Path,
                             *, top: int = 1) -> dict:
    frame = restrict_candidates(scores, config_ids)
    selected = select_budgeted_configs(frame, top)
    grouped = (frame.groupby(["arm", "config_id", "updates"], as_index=False)
               .agg(corr_mean=("corr_mean", "mean"),
                    corr_std=("corr_mean", "std"),
                    corr_worst=("corr_mean", "min"), cells=("corr_mean", "size"))
               .sort_values(["arm", "config_id", "updates"]))
    grouped["corr_std"] = grouped["corr_std"].fillna(0.0)

    output.mkdir(parents=True, exist_ok=True)
    table = output / "budget-sensitivity.csv"
    grouped.to_csv(table, index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for axis, arm in zip(axes, ("adamw", "spectral"), strict=True):
        arm_frame = grouped[grouped["arm"].eq(arm)]
        for config_id, values in arm_frame.groupby("config_id"):
            values = values.sort_values("updates")
            axis.plot(values["updates"], values["corr_mean"], marker="o",
                      label=f"config {int(config_id)}")
        axis.axhline(0, color="black", linewidth=0.8, alpha=0.5)
        axis.set_xscale("log")
        axis.set_title(arm.upper())
        axis.set_xlabel("Training updates (log scale)")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Mean exact CORR across development cells")
    axes[1].legend(loc="best", fontsize=8)
    fig.suptitle("Development-only training-budget sensitivity")
    fig.tight_layout()
    plot = output / "budget-sensitivity.png"
    temporary = output / "budget-sensitivity.png.tmp"
    fig.savefig(temporary, dpi=180, format="png", facecolor="white")
    plt.close(fig)
    os.replace(temporary, plot)

    report = {
        "status": "development_budget_sensitivity_complete",
        "config_ids": config_ids,
        "top": top,
        "selected": selected,
        "cells": len(frame),
        "artifacts": {path.name: sha256(path) for path in (table, plot)},
    }
    atomic_json(output / "budget-sensitivity-report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", type=Path, nargs="+", required=True)
    parser.add_argument("--config-id", type=int, nargs="+", required=True)
    parser.add_argument("--top", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    frame = pd.concat([pd.read_csv(path) for path in args.scores], ignore_index=True)
    report = build_budget_diagnostics(frame, args.config_id, args.output, top=args.top)
    report["score_files"] = [str(path) for path in args.scores]
    report["score_sha256"] = {str(path): sha256(path) for path in args.scores}
    atomic_json(args.output / "budget-sensitivity-report.json", report)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
