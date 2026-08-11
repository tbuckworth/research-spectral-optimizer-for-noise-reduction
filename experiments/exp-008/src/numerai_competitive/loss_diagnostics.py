"""Plot paired optimizer convergence from retained train-only stage logs."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .data import atomic_json, sha256
from .summarize import TASK


def collect_loss(results: Path, *, split: str, updates: int, seed: int) -> pd.DataFrame:
    rows = []
    for result_path in sorted(results.glob("*/result.json")):
        match = TASK.fullmatch(result_path.parent.name)
        if not match or (match["split"], int(match["updates"]), int(match["seed"])) != (
                split, updates, seed):
            continue
        result = json.loads(result_path.read_text())
        arm, config_id = match["arm"], int(match["config_id"])
        if (result.get("status") != "complete" or result.get("updates") != updates
                or result.get("config", {}).get("arm") != arm
                or result.get("config", {}).get("seed") != seed):
            raise ValueError(f"{result_path}: loss-log provenance differs from task path")
        logs = result.get("logs", [])
        if not logs or any("update" not in row or "loss" not in row for row in logs):
            raise ValueError(f"{result_path}: missing update/loss diagnostics")
        initial = float(logs[0]["loss"])
        if not np.isfinite(initial) or initial <= 0:
            raise ValueError(f"{result_path}: initial loss must be finite and positive")
        previous = 0
        for log in logs:
            update, loss = int(log["update"]), float(log["loss"])
            if update <= previous or not np.isfinite(loss) or loss <= 0:
                raise ValueError(f"{result_path}: loss diagnostics are invalid or unordered")
            rows.append({
                "arm": arm, "config_id": config_id, "update": update, "loss": loss,
                "loss_over_initial": loss / initial, "objective": result["config"]["loss"],
            })
            previous = update
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("no matching complete loss logs")
    coverage = {arm: set(frame.loc[frame["arm"].eq(arm), "config_id"])
                for arm in ("adamw", "spectral")}
    if coverage["adamw"] != coverage["spectral"]:
        raise ValueError("AdamW and spectral loss-log config coverage differs")
    identities = frame.groupby(["arm", "config_id"])["update"].apply(tuple)
    for config_id in coverage["adamw"]:
        if identities[("adamw", config_id)] != identities[("spectral", config_id)]:
            raise ValueError(f"config {config_id}: optimizer loss update grids differ")
    return frame.sort_values(["arm", "config_id", "update"]).reset_index(drop=True)


def write_loss_diagnostics(frame: pd.DataFrame, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    marker = output / "loss-diagnostics-complete.json"
    marker.unlink(missing_ok=True)
    csv_path = output / "training-loss-curves.csv"
    csv_tmp = output / "training-loss-curves.csv.tmp"
    frame.to_csv(csv_tmp, index=False)
    os.replace(csv_tmp, csv_path)

    grouped = (frame.groupby(["arm", "update"])["loss_over_initial"]
               .agg(median="median", q25=lambda values: values.quantile(0.25),
                    q75=lambda values: values.quantile(0.75)).reset_index())
    final = (frame.sort_values("update").groupby(["arm", "config_id"], as_index=False).tail(1)
             .pivot(index="config_id", columns="arm", values="loss_over_initial"))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    colors = {"adamw": "#4c78a8", "spectral": "#e45756"}
    for arm in ("adamw", "spectral"):
        values = grouped[grouped["arm"].eq(arm)]
        axes[0].plot(values["update"], values["median"], label=arm, color=colors[arm])
        axes[0].fill_between(values["update"], values["q25"], values["q75"],
                             color=colors[arm], alpha=0.18)
    axes[0].set(xlabel="Training update", ylabel="Loss / initial loss",
                title="Median paired convergence (IQR)", yscale="log")
    axes[0].legend()
    axes[1].scatter(final["adamw"], final["spectral"], alpha=0.8)
    low, high = final[["adamw", "spectral"]].min().min(), final[["adamw", "spectral"]].max().max()
    axes[1].plot([low, high], [low, high], "k--", linewidth=1)
    axes[1].set(xlabel="AdamW final loss / initial", ylabel="Spectral final loss / initial",
                title="Paired final normalized loss", xscale="log", yscale="log")
    fig.tight_layout()
    plot = output / "training-loss.png"
    plot_tmp = output / "training-loss.png.tmp"
    fig.savefig(plot_tmp, dpi=180, format="png", facecolor="white")
    plt.close(fig)
    os.replace(plot_tmp, plot)
    atomic_json(marker, {
        "status": "complete", "runs": int(frame.groupby(["arm", "config_id"]).ngroups),
        "configs_per_arm": int(frame["config_id"].nunique()),
        "artifacts": {path.name: sha256(path) for path in (csv_path, plot)},
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--updates", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write_loss_diagnostics(
        collect_loss(args.results, split=args.split, updates=args.updates, seed=args.seed),
        args.output,
    )


if __name__ == "__main__":
    main()
