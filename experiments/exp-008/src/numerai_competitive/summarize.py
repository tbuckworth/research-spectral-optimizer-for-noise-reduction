"""Aggregate a completed paired stage into auditable tables and plots."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .data import _load_features, atomic_json, sha256
from .materialize import materialize_config

TASK = re.compile(
    r"stage-(?P<split>.+)-u(?P<updates>\d+)-s(?P<seed>\d+)-"
    r"(?P<arm>adamw|spectral)-c(?P<config_id>\d+)"
)


def _verify_search_identity(result: dict, *, arm: str, config_id: int, updates: int,
                            seed: int, search_draws: list[dict],
                            feature_dimensions: dict[str, int]) -> str:
    matches = [draw for draw in search_draws
               if draw["arm"] == arm and draw["config_id"] == config_id]
    if len(matches) != 1:
        raise ValueError(f"expected one frozen search draw for {arm}/{config_id}")
    draw, actual = matches[0], result["config"]
    expected = materialize_config(
        draw, input_dim=feature_dimensions[draw["feature_set"]], updates=updates, seed=seed,
    )
    scientific_keys = (
        "model", "arm", "target", "benchmark", "seed", "batch_mode", "batch_size", "updates",
        "examples", "learning_rate", "weight_decay", "loss", "schedule",
        "warmup_updates", "clip_norm",
    )
    if any(actual.get(key) != expected.get(key) for key in scientific_keys):
        raise ValueError(f"result config differs from frozen search draw {arm}/{config_id}")
    if actual.get("feature_set", draw["feature_set"]) != draw["feature_set"]:
        raise ValueError(f"result feature set differs from frozen search draw {arm}/{config_id}")
    if arm == "spectral" and actual.get("filter") != expected["filter"]:
        raise ValueError(f"result filter differs from frozen search draw {arm}/{config_id}")
    if arm == "adamw" and actual.get("filter", {}) != {}:
        raise ValueError(f"AdamW result unexpectedly contains a spectral filter {config_id}")
    embedded_id = actual.get("search_config_id")
    if embedded_id is not None and embedded_id != config_id:
        raise ValueError(f"embedded config ID differs from task path {arm}/{config_id}")
    signature_payload = json.dumps(
        {"config": actual, "split": result["split"]}, sort_keys=True,
    ).encode()
    if hashlib.sha256(signature_payload).hexdigest() != result.get("signature"):
        raise ValueError(f"result signature is invalid for {arm}/{config_id}")
    return "embedded_id" if embedded_id is not None else "legacy_frozen_config_match"


def collect_stage(results: Path, *, split: str, updates: int, seed: int,
                  expected_configs: int = 40, search_draws: list[dict] | None = None,
                  feature_dimensions: dict[str, int] | None = None,
                  expected_config_ids: set[int] | None = None,
                  expected_arm_config_ids: dict[str, set[int]] | None = None) -> pd.DataFrame:
    if expected_config_ids is not None and expected_arm_config_ids is not None:
        raise ValueError("use either paired or arm-specific expected configuration IDs")
    if (expected_arm_config_ids is not None
            and (set(expected_arm_config_ids) != {"adamw", "spectral"}
                 or any(not values or any(value < 0 for value in values)
                        for values in expected_arm_config_ids.values()))):
        raise ValueError("arm-specific expected IDs must cover both arms")
    if expected_config_ids is not None:
        if not expected_config_ids or any(value < 0 for value in expected_config_ids):
            raise ValueError("expected config IDs must be a non-empty set of non-negative IDs")
        expected_configs = len(expected_config_ids)
    rows = []
    for result_path in sorted(results.glob("*/result.json")):
        match = TASK.fullmatch(result_path.parent.name)
        if not match or (match["split"], int(match["updates"]), int(match["seed"])) != (
                split, updates, seed):
            continue
        result = json.loads(result_path.read_text())
        arm, config_id = match["arm"], int(match["config_id"])
        if expected_config_ids is not None and config_id not in expected_config_ids:
            continue
        if (expected_arm_config_ids is not None
                and config_id not in expected_arm_config_ids[arm]):
            continue
        expected = {
            "status": "complete", "split": split, "updates": updates, "seed": seed,
            "arm": arm, "config_id": config_id,
        }
        actual = {
            "status": result.get("status"), "split": result.get("split", {}).get("name"),
            "updates": result.get("updates"), "seed": result.get("config", {}).get("seed"),
            "arm": result.get("config", {}).get("arm"),
            "config_id": result.get("config", {}).get("search_config_id"),
        }
        if actual != expected and not (search_draws is not None
                and actual | {"config_id": config_id} == expected):
            raise ValueError(f"{result_path}: result provenance differs from task path")
        provenance = "embedded_id"
        if search_draws is not None:
            if feature_dimensions is None:
                raise ValueError("feature dimensions are required with a frozen search")
            provenance = _verify_search_identity(
                result, arm=arm, config_id=config_id, updates=updates, seed=seed,
                search_draws=search_draws, feature_dimensions=feature_dimensions,
            )
        prediction_path = result_path.parent / result["prediction_file"]
        if not prediction_path.is_file():
            raise ValueError(f"{result_path}: prediction artifact is missing")
        with np.load(prediction_path) as saved:
            required = {"row_index", "era", "target", "benchmark", "prediction",
                        "per_era_corr", "per_era_bmc"}
            if not required <= set(saved.files):
                raise ValueError(f"{prediction_path}: incomplete prediction schema")
            n = len(saved["row_index"])
            if (n != result["validation"]["rows"]
                    or any(len(saved[key]) != n for key in
                           ("era", "target", "benchmark", "prediction"))
                    or len(np.unique(saved["row_index"])) != n
                    or not np.isfinite(saved["prediction"]).all()
                    or not np.isfinite(saved["target"]).all()
                    or not np.isfinite(saved["per_era_corr"]).all()):
                raise ValueError(f"{prediction_path}: corrupt or misaligned prediction arrays")
        rows.append({
            "arm": arm, "config_id": config_id,
            "split": split, "updates": updates, "seed": seed,
            "corr_mean": result["validation"]["corr"]["mean"],
            "corr_sharpe": result["validation"]["corr"]["sharpe"],
            "bmc_mean": result["validation"]["bmc"]["mean"],
            "parameters": result["parameter_count"],
            "peak_cuda_gib": result["peak_cuda_memory_bytes"] / 2**30,
            "elapsed_seconds": result["logs"][-1]["elapsed_seconds"],
            "examples": result["examples"], "signature": result["signature"],
            "provenance": provenance,
            "result_sha256": sha256(result_path),
            "prediction_sha256": sha256(prediction_path),
        })
    frame = pd.DataFrame(rows)
    if frame.duplicated(["arm", "config_id"]).any():
        raise ValueError("duplicate arm/config results")
    counts = frame.groupby("arm")["config_id"].nunique().to_dict()
    expected = ({arm: len(values) for arm, values in expected_arm_config_ids.items()}
                if expected_arm_config_ids is not None else
                {"adamw": expected_configs, "spectral": expected_configs})
    if counts != expected:
        raise ValueError(f"incomplete stage: got {counts}, expected {expected}")
    if expected_config_ids is not None:
        actual_ids = {
            arm: set(frame.loc[frame["arm"].eq(arm), "config_id"].astype(int))
            for arm in ("adamw", "spectral")
        }
        expected_ids = {arm: expected_config_ids for arm in ("adamw", "spectral")}
        if actual_ids != expected_ids:
            raise ValueError(f"stage config IDs differ: got {actual_ids}, expected {expected_ids}")
    if expected_arm_config_ids is not None:
        actual_ids = {
            arm: set(frame.loc[frame["arm"].eq(arm), "config_id"].astype(int))
            for arm in ("adamw", "spectral")
        }
        if actual_ids != expected_arm_config_ids:
            raise ValueError(
                f"stage arm/config IDs differ: got {actual_ids}, expected {expected_arm_config_ids}"
            )
    return frame.sort_values(["arm", "config_id"]).reset_index(drop=True)


def write_summary(frame: pd.DataFrame, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary-complete.json").unlink(missing_ok=True)
    scores = output / "scores.csv"
    scores_tmp = output / "scores.csv.tmp"
    frame.to_csv(scores_tmp, index=False)
    os.replace(scores_tmp, scores)
    paired = frame.pivot(index="config_id", columns="arm", values="corr_mean")
    paired["spectral_minus_adamw"] = paired["spectral"] - paired["adamw"]
    paired_path = output / "paired-corr.csv"
    paired_tmp = output / "paired-corr.csv.tmp"
    paired.to_csv(paired_tmp)
    os.replace(paired_tmp, paired_path)
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
    plot = output / "paired-corr.png"
    plot_tmp = output / "paired-corr.png.tmp"
    fig.savefig(plot_tmp, dpi=180, format="png")
    plt.close(fig)
    os.replace(plot_tmp, plot)
    atomic_json(output / "summary-complete.json", {
        "status": "complete", "rows": len(frame),
        "config_ids": sorted(frame["config_id"].astype(int).unique().tolist()),
        "artifacts": {path.name: sha256(path) for path in (scores, paired_path, plot)},
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--updates", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--expected-configs", type=int, default=40)
    parser.add_argument("--expected-config-id", type=int, action="append")
    parser.add_argument("--expected-arm-config-id", action="append")
    parser.add_argument("--search", type=Path)
    parser.add_argument("--features", type=Path)
    args = parser.parse_args()
    search_draws = None
    feature_dimensions = None
    if (args.search is None) != (args.features is None):
        parser.error("--search and --features must be supplied together")
    if args.search is not None:
        search_draws = json.loads(args.search.read_text())["configs"]
        feature_dimensions = {
            name: len(_load_features(args.features, name)) for name in ("medium", "all")
        }
    arm_ids = None
    if args.expected_arm_config_id:
        arm_ids = {"adamw": set(), "spectral": set()}
        for value in args.expected_arm_config_id:
            try:
                arm, raw_id = value.split(":", 1)
                config_id = int(raw_id)
            except (ValueError, AttributeError) as exc:
                parser.error(f"invalid --expected-arm-config-id {value!r}: {exc}")
            if arm not in arm_ids or config_id < 0:
                parser.error(f"invalid --expected-arm-config-id {value!r}")
            arm_ids[arm].add(config_id)
    write_summary(collect_stage(args.results, split=args.split, updates=args.updates,
                                seed=args.seed, expected_configs=args.expected_configs,
                                search_draws=search_draws,
                                feature_dimensions=feature_dimensions,
                                expected_config_ids=(set(args.expected_config_id)
                                                     if args.expected_config_id else None),
                                expected_arm_config_ids=arm_ids),
                  args.output)


if __name__ == "__main__":
    main()
