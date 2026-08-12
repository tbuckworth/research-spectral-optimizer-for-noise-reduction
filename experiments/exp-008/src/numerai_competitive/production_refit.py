"""Refit the frozen winning procedure on all currently resolved labels for live use."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .data import ProductionShard, sha256
from .materialize import materialize_config
from .model import MLPConfig
from .splits import EraSplit
from .train import TrainConfig, run_training


def run_production_refit(search_path: Path, shard_root: Path, freeze_path: Path,
                         evaluation_marker: Path, output: Path, seed: int,
                         device: str = "auto") -> dict:
    search = json.loads(search_path.read_text())
    freeze = json.loads(freeze_path.read_text())
    marker = json.loads(evaluation_marker.read_text())
    if freeze.get("status") != "frozen" or marker.get("status") != "complete":
        raise ValueError("production refit requires freeze and sealed evaluation")
    arm = freeze.get("candidate_transform", {}).get("arm")
    if arm not in {"adamw", "spectral"}:
        raise ValueError("freeze lacks a valid candidate arm")
    selected = freeze["selected"][arm]
    if seed not in selected["seeds"]:
        raise ValueError("production seed was not frozen for the candidate")
    matches = [draw for draw in search["configs"]
               if draw["arm"] == arm and draw["config_id"] == selected["config_id"]]
    if len(matches) != 1:
        raise ValueError("expected exactly one frozen candidate search draw")
    draw = matches[0]
    shard = ProductionShard.open(shard_root)
    if (shard.manifest["feature_set"] != selected["feature_set"]
            or shard.manifest["freeze_manifest_sha256"] != sha256(freeze_path)
            or shard.manifest["sealed_evaluation_sha256"] != sha256(evaluation_marker)):
        raise ValueError("production shard differs from frozen procedure/evaluation")
    value = materialize_config(
        draw, input_dim=shard.X.shape[1], updates=selected["updates"], seed=seed,
    )
    value["model"] = MLPConfig(**value["model"])
    value["save_model"] = True
    value["device"] = device
    eras = tuple(f"{int(era):04d}" for era in np.unique(shard.eras))
    split = EraSplit("production_live_refit_resolved", eras, (), ())
    return run_training(
        shard_root, split, TrainConfig(**value), output,
        refit=True, production_refit=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--search", type=Path, required=True)
    parser.add_argument("--shard", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--evaluation-marker", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    print(json.dumps(run_production_refit(
        args.search, args.shard, args.freeze, args.evaluation_marker, args.output, args.seed,
        args.device,
    ), sort_keys=True))


if __name__ == "__main__":
    main()
