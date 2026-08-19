"""Select, refit, audit, and export validation-stopped live candidates."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from . import PRIMARY_BENCHMARK
from .data import ProductionShard, TrainShard, atomic_json, sha256
from .live import export_callable
from .materialize import materialize_config
from .model import MLPConfig
from .splits import EraSplit
from .train import TrainConfig, run_training

ARMS = ("adamw", "spectral")
SEEDS = (0, 1, 2)
CHECKPOINT_EXAMPLES = (2_560_000, 5_120_000, 10_240_000, 20_480_000, 40_960_000)
CONFIG_IDS = {"adamw": 38, "spectral": 39}
MAX_UPDATES = {"adamw": 40_000, "spectral": 20_000}


def _identity(value: dict) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def select_stopping(results: list[Path], protocol: Path, code_commit: str,
                    output: Path, final_eras: int = 574) -> dict:
    """Freeze arm-specific peaks using only the two inner-fold metric histories."""
    if len(code_commit) != 40 or any(c not in "0123456789abcdef" for c in code_commit):
        raise ValueError("code commit must be a full lowercase Git SHA")
    cells: dict[str, list[tuple[Path, dict]]] = {arm: [] for arm in ARMS}
    for path in results:
        value = json.loads(path.read_text())
        arm = value.get("config", {}).get("arm")
        if arm not in ARMS:
            raise ValueError(f"invalid calibration arm in {path}")
        cells[arm].append((path, value))
    selected = {}
    evidence = []
    for arm in ARMS:
        rows = cells[arm]
        folds = {value.get("split", {}).get("name") for _, value in rows}
        if folds != {"outer_1_inner_1", "outer_1_inner_2"} or len(rows) != 2:
            raise ValueError(f"{arm} requires exactly the two frozen inner folds")
        by_examples = {examples: [] for examples in CHECKPOINT_EXAMPLES}
        later_train_eras = None
        batch_size = None
        for path, value in rows:
            config = value["config"]
            if (value.get("status") != "complete"
                    or config.get("search_config_id") != CONFIG_IDS[arm]
                    or config.get("seed") != 0
                    or config.get("updates") != MAX_UPDATES[arm]
                    or config.get("schedule_updates") != MAX_UPDATES[arm]):
                raise ValueError(f"calibration configuration mismatch in {path}")
            history = value.get("validation_history", [])
            if tuple(row.get("examples") for row in history) != CHECKPOINT_EXAMPLES:
                raise ValueError(f"checkpoint grid mismatch in {path}")
            for row in history:
                by_examples[row["examples"]].append(row["validation"]["corr"]["mean"])
            if value["split"]["name"] == "outer_1_inner_2":
                later_train_eras = len(value["split"]["train_eras"])
            batch_size = config["batch_size"]
            evidence.append({"arm": arm, "fold": value["split"]["name"],
                             "result": str(path), "sha256": sha256(path)})
        scores = [{"examples": examples, "mean_corr": float(np.mean(by_examples[examples])),
                   "fold_corr": by_examples[examples]} for examples in CHECKPOINT_EXAMPLES]
        winner = max(scores, key=lambda row: (row["mean_corr"], -row["examples"]))
        calibration_updates = winner["examples"] // batch_size
        scale = final_eras / later_train_eras
        selected[arm] = {
            "config_id": CONFIG_IDS[arm], "batch_size": batch_size,
            "calibration_examples": winner["examples"],
            "calibration_updates": calibration_updates,
            "calibration_schedule_updates": MAX_UPDATES[arm],
            "refit_updates": round(calibration_updates * scale),
            "refit_schedule_updates": round(MAX_UPDATES[arm] * scale),
            "calibration_train_eras": later_train_eras, "final_train_eras": final_eras,
            "era_scale": scale, "scores": scores, "seeds": list(SEEDS),
        }
    result = {
        "status": "corrected_live_frozen", "code_commit": code_commit,
        "protocol": str(protocol), "protocol_sha256": sha256(protocol),
        "checkpoint_examples": list(CHECKPOINT_EXAMPLES), "selected": selected,
        "evidence": evidence, "upload_authorized": True, "staking_authorized": False,
    }
    atomic_json(output, result)
    return result


def build_production_shard(train_root: Path, freeze_path: Path, destination: Path) -> dict:
    if destination.exists() or destination.with_name(destination.name + ".tmp").exists():
        raise ValueError("production shard destination already exists")
    freeze = json.loads(freeze_path.read_text())
    if freeze.get("status") != "corrected_live_frozen":
        raise ValueError("corrected live freeze is required")
    train = TrainShard.open(train_root)
    if not np.isfinite(train.targets[:, train.target_index("target")]).all():
        raise ValueError("train shard contains unresolved target labels")
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.mkdir(parents=True)
    for name in ("X_u8.npy", "targets_f32.npy", "era_i16.npy", "benchmarks_f32.npy"):
        os.link(train.root / name, temporary / name)
    identity_inputs = {
        "freeze_sha256": sha256(freeze_path),
        "train_manifest_sha256": sha256(train.root / "manifest.json"),
        "rows": len(train.X), "era_min": int(train.eras.min()),
        "era_max": int(train.eras.max()), "feature_set": train.manifest["feature_set"],
    }
    manifest = {
        **train.manifest, "split": "production_train",
        "corrected_live_freeze_sha256": sha256(freeze_path),
        "training_data_sha256": _identity(identity_inputs), "identity_inputs": identity_inputs,
    }
    atomic_json(temporary / "manifest.json", manifest)
    ProductionShard.open(temporary)
    os.replace(temporary, destination)
    return manifest


def run_refit(search_path: Path, shard_root: Path, freeze_path: Path, output: Path,
              arm: str, seed: int, device: str = "auto") -> dict:
    freeze = json.loads(freeze_path.read_text())
    search = json.loads(search_path.read_text())
    if freeze.get("status") != "corrected_live_frozen" or arm not in ARMS:
        raise ValueError("invalid corrected freeze or arm")
    chosen = freeze["selected"][arm]
    if seed not in chosen["seeds"]:
        raise ValueError("seed was not frozen")
    matches = [row for row in search["configs"]
               if row["arm"] == arm and row["config_id"] == chosen["config_id"]]
    if len(matches) != 1:
        raise ValueError("frozen search draw is not unique")
    shard = ProductionShard.open(shard_root)
    if shard.manifest.get("corrected_live_freeze_sha256") != sha256(freeze_path):
        raise ValueError("production shard differs from corrected freeze")
    value = materialize_config(
        matches[0], input_dim=shard.X.shape[1], updates=chosen["refit_updates"], seed=seed,
        schedule_updates=chosen["refit_schedule_updates"],
    )
    value.update(model=MLPConfig(**value["model"]), save_model=True, device=device)
    eras = tuple(f"{int(era):04d}" for era in np.unique(shard.eras))
    split = EraSplit("corrected_live_refit_train_v53", eras, (), ())
    return run_training(shard_root, split, TrainConfig(**value), output,
                        refit=True, production_refit=True)


def audit_refits(freeze_path: Path, shard_root: Path, results_root: Path,
                 output: Path) -> dict:
    freeze = json.loads(freeze_path.read_text())
    shard = ProductionShard.open(shard_root)
    cells = []
    for arm in ARMS:
        chosen = freeze["selected"][arm]
        for seed in SEEDS:
            root = results_root / f"corrected-live-refit-s{seed}-{arm}-c{chosen['config_id']}"
            result_path, model_path = root / "result.json", root / "model.pt"
            if not result_path.is_file() or not model_path.is_file():
                raise ValueError(f"missing refit for {arm} seed {seed}")
            result = json.loads(result_path.read_text())
            artifact = torch.load(model_path, map_location="cpu", weights_only=False)
            config = result.get("config", {})
            if (result.get("status") != "complete" or result.get("validation") is not None
                    or config.get("search_config_id") != chosen["config_id"]
                    or config.get("seed") != seed or config.get("arm") != arm
                    or result.get("updates") != chosen["refit_updates"]
                    or config.get("schedule_updates") != chosen["refit_schedule_updates"]
                    or result.get("training_data_sha256")
                    != shard.manifest["training_data_sha256"]
                    or artifact.get("signature") != result.get("signature")):
                raise ValueError(f"refit provenance mismatch for {arm} seed {seed}")
            cells.append({
                "arm": arm, "seed": seed, "config_id": chosen["config_id"],
                "updates": result["updates"], "schedule_updates": config["schedule_updates"],
                "examples": result["examples"], "parameter_count": result["parameter_count"],
                "model": str(model_path), "model_sha256": sha256(model_path),
                "result_sha256": sha256(result_path), "signature": result["signature"],
            })
    report = {"status": "audit_complete", "freeze_sha256": sha256(freeze_path),
              "training_data_sha256": shard.manifest["training_data_sha256"], "cells": cells}
    atomic_json(output, report)
    return report


def export_bundles(freeze_path: Path, audit_path: Path, output: Path,
                   batch_size: int = 2048) -> dict:
    freeze, audit = json.loads(freeze_path.read_text()), json.loads(audit_path.read_text())
    if (freeze.get("status") != "corrected_live_frozen"
            or audit.get("status") != "audit_complete"
            or audit.get("freeze_sha256") != sha256(freeze_path)):
        raise ValueError("corrected freeze and audit differ")
    output.mkdir(parents=True, exist_ok=False)
    bundles = {}
    for arm in ARMS:
        cells = sorted((row for row in audit["cells"] if row["arm"] == arm),
                       key=lambda row: row["seed"])
        if [row["seed"] for row in cells] != list(SEEDS):
            raise ValueError(f"{arm} audit lacks the exact ensemble")
        models = [Path(row["model"]) for row in cells]
        if [sha256(path) for path in models] != [row["model_sha256"] for row in cells]:
            raise ValueError(f"{arm} model hash changed")
        plan = output / f"{arm}-candidate-plan.json"
        atomic_json(plan, {"status": "corrected_validation_stopped",
                           "selected": {"arm": arm, "benchmark": PRIMARY_BENCHMARK},
                           "freeze_sha256": sha256(freeze_path),
                           "audit_sha256": sha256(audit_path)})
        callable_path = output / f"{arm}-predictor.pkl"
        export_callable(models, callable_path, batch_size=batch_size, candidate_plan=plan)
        bundles[arm] = {"callable": str(callable_path),
                        "callable_sha256": sha256(callable_path)}
    report = {"status": "bundles_complete", "freeze_sha256": sha256(freeze_path),
              "audit_sha256": sha256(audit_path), "bundles": bundles,
              "upload_authorized": True, "staking_authorized": False}
    atomic_json(output / "bundle-audit.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    select = sub.add_parser("select")
    select.add_argument("--result", type=Path, action="append", required=True)
    select.add_argument("--protocol", type=Path, required=True)
    select.add_argument("--code-commit", required=True)
    select.add_argument("--output", type=Path, required=True)
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
    export = sub.add_parser("export")
    export.add_argument("--freeze", type=Path, required=True)
    export.add_argument("--audit", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "select":
        value = select_stopping(args.result, args.protocol, args.code_commit, args.output)
    elif args.command == "build-shard":
        value = build_production_shard(args.train, args.freeze, args.destination)
    elif args.command == "refit":
        value = run_refit(args.search, args.shard, args.freeze, args.output,
                          args.arm, args.seed, args.device)
    elif args.command == "audit":
        value = audit_refits(args.freeze, args.shard, args.results, args.output)
    else:
        value = export_bundles(args.freeze, args.audit, args.output)
    print(json.dumps(value, sort_keys=True))


if __name__ == "__main__":
    main()
