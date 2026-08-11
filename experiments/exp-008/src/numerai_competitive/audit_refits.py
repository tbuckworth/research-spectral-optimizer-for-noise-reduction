"""Audit the six selected full-train refits before procedure freeze."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from .data import _load_features, atomic_json, sha256
from .model import MLPConfig, ResidualMLP
from .summarize import _verify_search_identity


def _manifest(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text().splitlines():
        fields = line.split("\t")
        if len(fields) != 5:
            raise ValueError("refit manifest rows must have five columns")
        job, arm, config_id, updates, seed = fields
        if (not job.split(";", 1)[0].isdigit() or arm not in {"adamw", "spectral"}
                or not config_id.isdigit() or not updates.isdigit() or not seed.isdigit()):
            raise ValueError("invalid refit manifest value")
        rows.append({
            "job": job, "arm": arm, "config_id": int(config_id),
            "updates": int(updates), "seed": int(seed),
        })
    if not rows:
        raise ValueError("refit manifest is empty")
    return rows


def audit_refits(manifest: Path, results: Path, selection: dict, search_draws: list[dict],
                 feature_dimensions: dict[str, int], output: Path,
                 updates: int = 100000, seeds: tuple[int, ...] = (0, 1, 2)) -> dict:
    selected = selection.get("selected", {})
    winners = {}
    for arm in ("adamw", "spectral"):
        ids = selected.get(arm, [])
        if len(ids) != 1 or not isinstance(ids[0], int):
            raise ValueError(f"{arm} final selection must contain one config ID")
        winners[arm] = ids[0]
    rows = _manifest(manifest)
    actual = {(row["arm"], row["config_id"], row["updates"], row["seed"]) for row in rows}
    expected = {(arm, winners[arm], updates, seed) for arm in winners for seed in seeds}
    if (len(rows) != len(actual) or len({row["job"] for row in rows}) != len(rows)
            or actual != expected):
        raise ValueError("refit manifest differs from selected arm/config/seed cells")

    audited = []
    expected_eras = [f"{era:04d}" for era in range(1, 575)]
    for row in sorted(rows, key=lambda value: (value["arm"], value["seed"])):
        directory = results / (
            f"final-refit-u{updates}-s{row['seed']}-{row['arm']}-c{row['config_id']}"
        )
        result_path, model_path = directory / "result.json", directory / "model.pt"
        if not result_path.is_file() or not model_path.is_file():
            raise ValueError(f"{directory}: refit result or model is missing")
        result = json.loads(result_path.read_text())
        split = result.get("split", {})
        if (result.get("status") != "complete" or result.get("updates") != updates
                or result.get("validation") is not None
                or result.get("model_file") != "model.pt"
                or split.get("name") != "all_train_refit"
                or split.get("train_eras") != expected_eras
                or split.get("valid_eras") or split.get("purged_eras")):
            raise ValueError(f"{result_path}: invalid full-train refit provenance")
        provenance = _verify_search_identity(
            result, arm=row["arm"], config_id=row["config_id"], updates=updates,
            seed=row["seed"], search_draws=search_draws,
            feature_dimensions=feature_dimensions,
        )
        artifact = torch.load(model_path, map_location="cpu", weights_only=False, mmap=True)
        with torch.device("meta"):
            expected_state = ResidualMLP(MLPConfig(**result["config"]["model"])).state_dict()
        actual_state = artifact.get("model")
        if (artifact.get("signature") != result["signature"]
                or artifact.get("train_config") != result["config"]
                or artifact.get("train_split") != split
                or artifact.get("data_version") != "v5.3"
                or artifact.get("target") != "target"
                or len(artifact.get("feature_names", []))
                != feature_dimensions[result["config"]["feature_set"]]
                or not isinstance(actual_state, dict)
                or actual_state.keys() != expected_state.keys()
                or any(actual_state[key].shape != value.shape
                       for key, value in expected_state.items())
                or sum(value.numel() for value in expected_state.values())
                != result["parameter_count"]):
            raise ValueError(f"{model_path}: model artifact differs from refit result")
        audited.append({
            **row, "signature": result["signature"], "provenance": provenance,
            "parameters": result["parameter_count"], "result_sha256": sha256(result_path),
            "model_bytes": model_path.stat().st_size,
        })
    output.mkdir(parents=True, exist_ok=True)
    table = output / "refits.csv"
    pd.DataFrame(audited).to_csv(table, index=False)
    report = {
        "status": "audit_complete", "updates": updates, "seeds": list(seeds),
        "selected": winners, "cells": len(audited), "manifest_sha256": sha256(manifest),
        "table_sha256": sha256(table),
        "note": "full model SHA-256 hashes are computed by the immutable freeze step",
    }
    atomic_json(output / "refit-audit.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--search", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    dimensions = {name: len(_load_features(args.features, name)) for name in ("medium", "all")}
    report = audit_refits(
        args.manifest, args.results, json.loads(args.selection.read_text()),
        json.loads(args.search.read_text())["configs"], dimensions, args.output,
    )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
