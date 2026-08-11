"""Audit selected nested-outer results before they enter the OOF estimator."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import _load_features, atomic_json, sha256
from .metrics import per_era_corr, per_era_correlation_contribution, summarize_era_scores
from .run_task import resolve_split
from .splits import EraSplit
from .summarize import _verify_search_identity


def _manifest(path: Path) -> list[dict]:
    rows = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        fields = line.split("\t")
        if len(fields) != 6:
            raise ValueError(f"manifest line {line_number} is not six columns")
        job, split, updates, seed, arm, config_id = fields
        rows.append({"job": job.split(";", 1)[0], "split": split, "updates": int(updates),
                     "seed": int(seed), "arm": arm, "config_id": int(config_id)})
    if not rows:
        raise ValueError("outer manifest is empty")
    return rows


def _prediction_arrays(path: Path, result: dict, split: EraSplit) -> dict[str, np.ndarray]:
    prediction_path = path.parent / result["prediction_file"]
    if not prediction_path.is_file():
        raise ValueError(f"{prediction_path}: prediction artifact is missing")
    with np.load(prediction_path) as saved:
        required = {"row_index", "era", "target", "benchmark", "prediction",
                    "per_era_corr", "per_era_bmc"}
        if not required <= set(saved.files):
            raise ValueError(f"{prediction_path}: incomplete prediction schema")
        arrays = {key: saved[key].copy() for key in required}
    n = len(arrays["row_index"])
    row_keys = ("era", "target", "benchmark", "prediction")
    if (n != result["validation"]["rows"]
            or any(len(arrays[key]) != n for key in row_keys)
            or len(np.unique(arrays["row_index"])) != n
            or not np.isfinite(arrays["prediction"]).all()
            or not np.isfinite(arrays["target"]).all()):
        raise ValueError(f"{prediction_path}: corrupt or misaligned prediction arrays")
    observed_eras = {f"{int(era):04d}" for era in np.unique(arrays["era"])}
    if observed_eras != set(split.valid_eras):
        raise ValueError(f"{prediction_path}: eras differ from untouched outer split")
    ids = pd.Index(arrays["row_index"], name="row_index")
    eras = pd.Series([f"{int(era):04d}" for era in arrays["era"]], index=ids, name="era")
    prediction = pd.Series(arrays["prediction"], index=ids, name="prediction")
    target = pd.Series(arrays["target"], index=ids, name="target")
    corr = per_era_corr(prediction, target, eras)["prediction"]
    summary = summarize_era_scores(corr).loc["prediction"]
    recorded = result["validation"]["corr"]
    for key in ("mean", "std", "sharpe", "max_drawdown", "cumulative"):
        if not np.isclose(float(summary[key]), float(recorded[key]), rtol=1e-10, atol=1e-12):
            raise ValueError(f"{prediction_path}: recorded CORR {key} does not reproduce")
    if (len(arrays["per_era_corr"]) != len(corr)
            or not np.allclose(arrays["per_era_corr"], corr.to_numpy(), equal_nan=False)):
        raise ValueError(f"{prediction_path}: stored per-era CORR does not reproduce")
    covered = np.isfinite(arrays["benchmark"])
    benchmark = pd.Series(arrays["benchmark"][covered], index=ids[covered], name="benchmark")
    bmc = per_era_correlation_contribution(
        prediction[covered], benchmark, target[covered], eras[covered]
    )["prediction"]
    bmc_summary = summarize_era_scores(bmc).loc["prediction"]
    recorded_bmc = result["validation"]["bmc"]
    for key in ("mean", "std", "sharpe", "max_drawdown", "cumulative"):
        if not np.isclose(float(bmc_summary[key]), float(recorded_bmc[key]),
                          rtol=1e-10, atol=1e-12):
            raise ValueError(f"{prediction_path}: recorded BMC {key} does not reproduce")
    if (len(arrays["per_era_bmc"]) != len(bmc)
            or not np.allclose(arrays["per_era_bmc"], bmc.to_numpy(), equal_nan=False)):
        raise ValueError(f"{prediction_path}: stored per-era BMC does not reproduce")
    arrays["prediction_sha256"] = np.array(sha256(prediction_path))
    return arrays


def audit_outer(manifest: Path, results: Path, selection: dict, search_draws: list[dict],
                feature_dimensions: dict[str, int], split: EraSplit, updates: int | None,
                seeds: tuple[int, ...], output: Path) -> dict:
    selected = selection.get("selected", {})
    selected_updates = selection.get("selected_updates", {})
    winners = {}
    winner_updates = {}
    for arm in ("adamw", "spectral"):
        ids = selected.get(arm, [])
        if len(ids) != 1:
            raise ValueError(f"{arm} outer selection must contain exactly one config")
        winners[arm] = int(ids[0])
        if selected_updates:
            budgets = selected_updates.get(arm, [])
            if len(budgets) != 1:
                raise ValueError(f"{arm} outer selection must contain exactly one update budget")
            winner_updates[arm] = int(budgets[0])
        elif updates is not None:
            winner_updates[arm] = updates
        else:
            raise ValueError("outer selection does not specify update budgets")
    manifest_rows = _manifest(manifest)
    actual_cells = {(row["split"], row["updates"], row["seed"], row["arm"],
                     row["config_id"]) for row in manifest_rows}
    expected_cells = {(split.name, winner_updates[arm], seed, arm, winners[arm])
                      for arm in winners for seed in seeds}
    if (len({row["job"] for row in manifest_rows}) != len(manifest_rows)
            or len(actual_cells) != len(manifest_rows) or actual_cells != expected_cells):
        raise ValueError("outer manifest differs from selected arm/config/seed cells")

    rows, reference = [], None
    for row in sorted(manifest_rows, key=lambda value: (value["arm"], value["seed"])):
        task = (f"stage-{split.name}-u{row['updates']}-s{row['seed']}-"
                f"{row['arm']}-c{row['config_id']}")
        result_path = results / task / "result.json"
        if not result_path.is_file():
            raise ValueError(f"{result_path}: selected outer result is missing")
        result = json.loads(result_path.read_text())
        if (result.get("status") != "complete" or result.get("updates") != row["updates"]
                or result.get("split") != split.to_dict()
                or result.get("config", {}).get("arm") != row["arm"]
                or result.get("config", {}).get("seed") != row["seed"]):
            raise ValueError(f"{result_path}: result provenance differs from outer manifest")
        provenance = _verify_search_identity(
            result, arm=row["arm"], config_id=row["config_id"], updates=row["updates"],
            seed=row["seed"], search_draws=search_draws,
            feature_dimensions=feature_dimensions,
        )
        arrays = _prediction_arrays(result_path, result, split)
        alignment = (arrays["row_index"], arrays["era"], arrays["target"],
                     arrays["benchmark"])
        if reference is None:
            reference = alignment
        elif any(not np.array_equal(left, right, equal_nan=True)
                 for left, right in zip(reference, alignment)):
            raise ValueError(f"{result_path}: outer rows/targets differ across arms or seeds")
        rows.append({
            **row, "corr_mean": result["validation"]["corr"]["mean"],
            "corr_sharpe": result["validation"]["corr"]["sharpe"],
            "bmc_mean": result["validation"]["bmc"]["mean"],
            "rows": result["validation"]["rows"], "provenance": provenance,
            "signature": result["signature"], "result_sha256": sha256(result_path),
            "prediction_sha256": str(arrays["prediction_sha256"]),
        })
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows).sort_values(["arm", "seed"])
    frame.to_csv(output / "outer-results.csv", index=False)
    reported_updates: int | dict[str, int] = (
        updates if not selected_updates and updates is not None else winner_updates
    )
    report = {
        "status": "audit_complete", "split": split.to_dict(), "updates": reported_updates,
        "seeds": list(seeds), "selected": winners, "cells": len(frame),
        "rows_per_cell": int(frame["rows"].iloc[0]), "manifest_sha256": sha256(manifest),
        "selection": selection,
    }
    atomic_json(output / "outer-audit.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--search", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--outer-split", choices=["outer_1", "outer_2", "outer_3"], required=True)
    parser.add_argument("--updates", default="selected")
    parser.add_argument("--seed", type=int, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    dimensions = {name: len(_load_features(args.features, name)) for name in ("medium", "all")}
    if args.updates == "selected":
        updates = None
    elif args.updates.isdigit() and int(args.updates) > 0:
        updates = int(args.updates)
    else:
        raise ValueError("--updates must be a positive integer or 'selected'")
    report = audit_outer(
        args.manifest, args.results, json.loads(args.selection.read_text()),
        json.loads(args.search.read_text())["configs"], dimensions,
        resolve_split(args.outer_split), updates, tuple(args.seed), args.output,
    )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
