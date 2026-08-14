"""Freeze, refit and audit the paired bounded-study live candidates."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from .data import ProductionShard, TrainShard, atomic_json, sha256
from .materialize import materialize_config
from .model import MLPConfig
from .splits import EraSplit
from .train import TrainConfig, run_training

ARMS = ("adamw", "spectral")
SEEDS = (0, 1, 2)


def _identity(value: dict) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def create_freeze(selection_path: Path, outer_audit_path: Path, search_path: Path,
                  protocol_path: Path, code_commit: str, output: Path) -> dict:
    if len(code_commit) != 40 or any(c not in "0123456789abcdef" for c in code_commit):
        raise ValueError("code commit must be a full lowercase Git SHA")
    selection = json.loads(selection_path.read_text())
    audit = json.loads(outer_audit_path.read_text())
    search = json.loads(search_path.read_text())
    if (selection.get("status") != "one_day_development_selection_frozen"
            or selection.get("outer_reselection_allowed") is not False
            or audit.get("status") != "audit_complete"):
        raise ValueError("bounded selection and outer audit must be complete and frozen")
    selected = {}
    for arm in ARMS:
        ids = selection["selected"].get(arm, [])
        budgets = selection["selected_updates"].get(arm, [])
        if len(ids) != 1 or len(budgets) != 1 or audit["selected"].get(arm) != ids[0]:
            raise ValueError(f"{arm} selection differs from outer audit")
        matches = [row for row in search["configs"]
                   if row["arm"] == arm and row["config_id"] == ids[0]]
        if len(matches) != 1 or matches[0]["feature_set"] != "all":
            raise ValueError(f"{arm} frozen draw is missing or not all-feature")
        selected[arm] = {"config_id": ids[0], "updates": budgets[0],
                         "seeds": list(SEEDS), "feature_set": "all"}
    value = {
        "status": "bounded_live_frozen", "code_commit": code_commit,
        "target": "target", "selected": selected, "staking_authorized": False,
        "upload_authorized": False,
        "selection_sha256": sha256(selection_path),
        "outer_audit_sha256": sha256(outer_audit_path),
        "search_sha256": sha256(search_path), "protocol_sha256": sha256(protocol_path),
    }
    atomic_json(output, value)
    return value


def build_train_only_shard(train_root: Path, freeze_path: Path, destination: Path) -> dict:
    if destination.exists() or destination.with_name(destination.name + ".tmp").exists():
        raise ValueError("bounded production shard destination already exists")
    freeze = json.loads(freeze_path.read_text())
    if freeze.get("status") != "bounded_live_frozen":
        raise ValueError("bounded live freeze is required")
    train = TrainShard.open(train_root)
    target_index = train.target_index(freeze["target"])
    if not np.isfinite(train.targets[:, target_index]).all():
        raise ValueError("train shard contains unresolved main-target labels")
    identity_inputs = {
        "freeze_sha256": sha256(freeze_path),
        "train_manifest_sha256": sha256(train.root / "manifest.json"),
        "feature_set": train.manifest["feature_set"], "target": freeze["target"],
        "rows": len(train.X), "era_min": int(train.eras.min()),
        "era_max": int(train.eras.max()),
    }
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.mkdir(parents=True)
    for name in ("X_u8.npy", "targets_f32.npy", "era_i16.npy", "benchmarks_f32.npy"):
        os.link(train.root / name, temporary / name)
    manifest = {
        **train.manifest, "split": "production_train",
        "bounded_live_freeze_sha256": sha256(freeze_path),
        "freeze_manifest_sha256": sha256(freeze_path),
        "sealed_evaluation_sha256": None,
        "resolved_validation_rows": 0,
        "frozen_train_rows": len(train.X),
        "training_data_sha256": _identity(identity_inputs),
        "identity_inputs": identity_inputs,
    }
    atomic_json(temporary / "manifest.json", manifest)
    ProductionShard.open(temporary)
    os.replace(temporary, destination)
    return manifest


def run_refit(search_path: Path, shard_root: Path, freeze_path: Path, output: Path,
              arm: str, seed: int, device: str = "auto") -> dict:
    freeze = json.loads(freeze_path.read_text())
    search = json.loads(search_path.read_text())
    if freeze.get("status") != "bounded_live_frozen" or arm not in ARMS:
        raise ValueError("invalid bounded live freeze or arm")
    selected = freeze["selected"][arm]
    if seed not in selected["seeds"]:
        raise ValueError("seed was not frozen")
    matches = [row for row in search["configs"]
               if row["arm"] == arm and row["config_id"] == selected["config_id"]]
    if len(matches) != 1:
        raise ValueError("frozen search draw is not unique")
    shard = ProductionShard.open(shard_root)
    if (shard.manifest["bounded_live_freeze_sha256"] != sha256(freeze_path)
            or shard.manifest["feature_set"] != selected["feature_set"]):
        raise ValueError("production shard differs from bounded freeze")
    value = materialize_config(matches[0], input_dim=shard.X.shape[1],
                               updates=selected["updates"], seed=seed)
    value["model"] = MLPConfig(**value["model"])
    value["save_model"] = True
    value["device"] = device
    eras = tuple(f"{int(era):04d}" for era in np.unique(shard.eras))
    split = EraSplit("bounded_live_refit_train_v53", eras, (), ())
    return run_training(shard_root, split, TrainConfig(**value), output,
                        refit=True, production_refit=True)


def audit_refits(freeze_path: Path, shard_root: Path, results_root: Path,
                 output: Path) -> dict:
    freeze = json.loads(freeze_path.read_text())
    shard = ProductionShard.open(shard_root)
    rows = []
    for arm in ARMS:
        selected = freeze["selected"][arm]
        for seed in SEEDS:
            root = results_root / f"bounded-live-refit-s{seed}-{arm}-c{selected['config_id']}"
            result_path, model_path = root / "result.json", root / "model.pt"
            if not result_path.is_file() or not model_path.is_file():
                raise ValueError(f"missing refit artifact for {arm} seed {seed}")
            result = json.loads(result_path.read_text())
            artifact = torch.load(model_path, map_location="cpu", weights_only=False)
            config = result.get("config", {})
            if (result.get("status") != "complete" or result.get("validation") is not None
                    or config.get("arm") != arm or config.get("seed") != seed
                    or config.get("search_config_id") != selected["config_id"]
                    or result.get("updates") != selected["updates"]
                    or result.get("training_data_sha256") != shard.manifest["training_data_sha256"]
                    or artifact.get("signature") != result.get("signature")
                    or artifact.get("training_data_sha256") != shard.manifest["training_data_sha256"]):
                raise ValueError(f"refit provenance mismatch for {arm} seed {seed}")
            rows.append({"arm": arm, "seed": seed, "config_id": selected["config_id"],
                         "updates": selected["updates"], "signature": result["signature"],
                         "parameter_count": result["parameter_count"],
                         "model": str(model_path), "model_sha256": sha256(model_path),
                         "result_sha256": sha256(result_path)})
    report = {
        "status": "audit_complete", "freeze_sha256": sha256(freeze_path),
        "training_data_sha256": shard.manifest["training_data_sha256"],
        "cells": rows,
    }
    atomic_json(output, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    freeze = sub.add_parser("freeze")
    freeze.add_argument("--selection", type=Path, required=True)
    freeze.add_argument("--outer-audit", type=Path, required=True)
    freeze.add_argument("--search", type=Path, required=True)
    freeze.add_argument("--protocol", type=Path, required=True)
    freeze.add_argument("--code-commit", required=True)
    freeze.add_argument("--output", type=Path, required=True)
    shard = sub.add_parser("build-shard")
    shard.add_argument("--train", type=Path, required=True)
    shard.add_argument("--freeze", type=Path, required=True)
    shard.add_argument("--destination", type=Path, required=True)
    refit = sub.add_parser("refit")
    refit.add_argument("--search", type=Path, required=True)
    refit.add_argument("--shard", type=Path, required=True)
    refit.add_argument("--freeze", type=Path, required=True)
    refit.add_argument("--output", type=Path, required=True)
    refit.add_argument("--arm", choices=ARMS, required=True)
    refit.add_argument("--seed", type=int, required=True)
    refit.add_argument("--device", default="auto")
    audit = sub.add_parser("audit")
    audit.add_argument("--freeze", type=Path, required=True)
    audit.add_argument("--shard", type=Path, required=True)
    audit.add_argument("--results", type=Path, required=True)
    audit.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "freeze":
        value = create_freeze(args.selection, args.outer_audit, args.search, args.protocol,
                              args.code_commit, args.output)
    elif args.command == "build-shard":
        value = build_train_only_shard(args.train, args.freeze, args.destination)
    elif args.command == "refit":
        value = run_refit(args.search, args.shard, args.freeze, args.output,
                          args.arm, args.seed, args.device)
    else:
        value = audit_refits(args.freeze, args.shard, args.results, args.output)
    print(json.dumps(value, sort_keys=True))


if __name__ == "__main__":
    main()
