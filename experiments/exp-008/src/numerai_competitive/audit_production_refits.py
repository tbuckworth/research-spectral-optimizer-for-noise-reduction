"""Audit production-live refits without replacing sealed-validation evidence."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch

from .code_snapshot import verify_snapshot
from .data import ProductionShard, atomic_json, sha256


def audit_production_refits(freeze_path: Path, evaluation_marker: Path, shard_root: Path,
                            models: list[Path], output: Path, production_code_commit: str,
                            production_code_snapshot: Path, code_root: Path) -> dict:
    if not re.fullmatch(r"[0-9a-f]{40}", production_code_commit):
        raise ValueError("production code commit must be a full Git SHA")
    verify_snapshot(code_root, production_code_snapshot, production_code_commit)
    freeze = json.loads(freeze_path.read_text())
    marker = json.loads(evaluation_marker.read_text())
    shard = ProductionShard.open(shard_root)
    if freeze.get("status") != "frozen" or marker.get("status") != "complete":
        raise ValueError("freeze and sealed evaluation must be complete")
    arm = freeze.get("candidate_transform", {}).get("arm")
    if arm not in {"adamw", "spectral"}:
        raise ValueError("freeze lacks a candidate arm")
    selected = freeze["selected"][arm]
    seeds = selected["seeds"]
    if len(models) != len(seeds):
        raise ValueError("one production model is required per frozen seed")
    if (shard.manifest["freeze_manifest_sha256"] != sha256(freeze_path)
            or shard.manifest["sealed_evaluation_sha256"] != sha256(evaluation_marker)):
        raise ValueError("production shard provenance differs from freeze/evaluation")
    expected_eras = [f"{int(era):04d}" for era in sorted(set(shard.eras.tolist()))]
    signatures, hashes = [], []
    for path, seed in zip(models, seeds):
        artifact = torch.load(path, map_location="cpu", weights_only=False)
        config, split = artifact["train_config"], artifact["train_split"]
        expected = {
            "arm": arm, "search_config_id": selected["config_id"], "seed": seed,
            "updates": selected["updates"], "target": "target",
            "benchmark": freeze["primary_benchmark"],
            "feature_set": selected["feature_set"],
        }
        if {key: config[key] for key in expected} != expected:
            raise ValueError(f"seed {seed}: production model config differs from freeze")
        if (split.get("name") != "production_live_refit_resolved"
                or split.get("train_eras") != expected_eras
                or split.get("valid_eras") or split.get("purged_eras")):
            raise ValueError(f"seed {seed}: production model split is invalid")
        if artifact.get("training_data_sha256") != shard.manifest["training_data_sha256"]:
            raise ValueError(f"seed {seed}: production training-data identity differs")
        signatures.append(artifact["signature"])
        hashes.append(sha256(path))
    report = {
        "status": "audit_complete", "purpose": "unstaked_forward_live_candidate",
        "arm": arm, "config_id": selected["config_id"], "updates": selected["updates"],
        "seeds": seeds, "feature_set": selected["feature_set"],
        "freeze_manifest_sha256": sha256(freeze_path),
        "procedure_code_commit": freeze["code_commit"],
        "production_code_commit": production_code_commit,
        "production_code_snapshot_sha256": sha256(production_code_snapshot),
        "sealed_evaluation_sha256": sha256(evaluation_marker),
        "production_manifest_sha256": sha256(shard.root / "manifest.json"),
        "production_manifest": shard.manifest,
        "training_data_sha256": shard.manifest["training_data_sha256"],
        "training_era_max": shard.manifest["era_max"],
        "resolved_validation_rows": shard.manifest["resolved_validation_rows"],
        "model_signatures": signatures, "model_sha256": hashes,
    }
    atomic_json(output, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--evaluation-marker", type=Path, required=True)
    parser.add_argument("--shard", type=Path, required=True)
    parser.add_argument("--model", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--production-code-commit", required=True)
    parser.add_argument("--production-code-snapshot", type=Path, required=True)
    parser.add_argument("--code-root", type=Path, default=Path("."))
    args = parser.parse_args()
    print(json.dumps(audit_production_refits(
        args.freeze, args.evaluation_marker, args.shard, args.model, args.output,
        args.production_code_commit, args.production_code_snapshot, args.code_root,
    ), sort_keys=True))


if __name__ == "__main__":
    main()
