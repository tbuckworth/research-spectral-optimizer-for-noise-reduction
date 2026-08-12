"""Build an audited train+resolved-validation shard for production live refits."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from .data import ProductionShard, TrainShard, ValidationShard, atomic_json, sha256


def _identity(value: dict) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def build_production_shard(train_root: Path, validation_root: Path, freeze_path: Path,
                           evaluation_marker: Path, destination: Path) -> dict:
    """Combine frozen train data with only target-resolved validation rows.

    This is intentionally unavailable until sealed evaluation is complete. Hyperparameters
    remain frozen; only the training-data cutoff changes for forward live prediction.
    """
    temporary = destination.with_name(destination.name + ".tmp")
    if destination.exists() or temporary.exists():
        raise ValueError("production shard destination or temporary output already exists")
    freeze = json.loads(freeze_path.read_text())
    marker = json.loads(evaluation_marker.read_text())
    if freeze.get("status") != "frozen" or marker.get("status") != "complete":
        raise ValueError("frozen procedure and completed sealed evaluation are required")
    report_path = evaluation_marker.parent / "official-validation-report.json"
    report_hash = sha256(report_path)
    if marker.get("artifacts", {}).get(report_path.name) != report_hash:
        raise ValueError("sealed evaluation marker does not authenticate its report")
    report = json.loads(report_path.read_text())
    freeze_hash = sha256(freeze_path)
    if report.get("freeze_manifest_sha256") != freeze_hash:
        raise ValueError("sealed evaluation and freeze manifest differ")

    candidate_arm = freeze.get("candidate_transform", {}).get("arm")
    if candidate_arm not in {"adamw", "spectral"}:
        raise ValueError("freeze lacks one candidate arm")
    expected_feature_set = freeze["selected"][candidate_arm]["feature_set"]
    train = TrainShard.open(train_root)
    validation = ValidationShard.open(validation_root)
    if train.manifest.get("feature_set") != expected_feature_set:
        raise ValueError("train shard does not match the frozen candidate feature set")
    if validation.manifest["freeze_manifest_sha256"] != freeze_hash:
        raise ValueError("validation shard and freeze manifest differ")
    if train.manifest["targets"] != validation.manifest["targets"]:
        raise ValueError("train and validation target schemas differ")
    if train.manifest["benchmarks"] != validation.manifest["benchmarks"]:
        raise ValueError("train and validation benchmark schemas differ")

    validation_positions = {name: i for i, name in enumerate(validation.manifest["feature_names"])}
    train_features = list(train.manifest["feature_names"])
    if any(name not in validation_positions for name in train_features):
        raise ValueError("validation shard lacks frozen candidate features")
    feature_indices = np.asarray([validation_positions[name] for name in train_features])
    target_index = train.target_index("target")
    if not np.isfinite(train.targets[:, target_index]).all():
        raise ValueError("frozen train target contains unresolved rows")
    covered = np.isfinite(validation.targets[:, target_index])
    validation_rows = np.flatnonzero(covered)
    if not len(validation_rows):
        raise ValueError("validation has no resolved main-target rows")
    if int(validation.eras[validation_rows].min()) <= int(train.eras.max()):
        raise ValueError("resolved validation eras overlap frozen train eras")

    identity_inputs = {
        "freeze_manifest_sha256": freeze_hash,
        "sealed_evaluation_sha256": sha256(evaluation_marker),
        "train_manifest_sha256": sha256(train.root / "manifest.json"),
        "validation_manifest_sha256": sha256(validation.root / "manifest.json"),
        "feature_set": expected_feature_set,
        "target": "target",
        "resolved_validation_rows": len(validation_rows),
        "resolved_validation_era_min": int(validation.eras[validation_rows].min()),
        "resolved_validation_era_max": int(validation.eras[validation_rows].max()),
    }
    training_data_sha256 = _identity(identity_inputs)
    temporary.mkdir(parents=True)
    total = len(train.X) + len(validation_rows)
    arrays = {
        "X": np.lib.format.open_memmap(
            temporary / "X_u8.npy", mode="w+", dtype=np.uint8,
            shape=(total, len(train_features)),
        ),
        "targets": np.lib.format.open_memmap(
            temporary / "targets_f32.npy", mode="w+", dtype=np.float32,
            shape=(total, train.targets.shape[1]),
        ),
        "eras": np.lib.format.open_memmap(
            temporary / "era_i16.npy", mode="w+", dtype=np.int16, shape=(total,),
        ),
        "benchmarks": np.lib.format.open_memmap(
            temporary / "benchmarks_f32.npy", mode="w+", dtype=np.float32,
            shape=(total, train.benchmarks.shape[1]),
        ),
    }
    boundary = len(train.X)
    arrays["X"][:boundary] = train.X
    arrays["targets"][:boundary] = train.targets
    arrays["eras"][:boundary] = train.eras
    arrays["benchmarks"][:boundary] = train.benchmarks
    chunk_size = 100_000
    for start in range(0, len(validation_rows), chunk_size):
        source_rows = validation_rows[start:start + chunk_size]
        destination_rows = slice(boundary + start, boundary + start + len(source_rows))
        # Read only the frozen feature subset; never materialize the all-feature validation
        # matrix in RAM while constructing a medium-feature production shard.
        arrays["X"][destination_rows] = validation.X[np.ix_(source_rows, feature_indices)]
        arrays["targets"][destination_rows] = validation.targets[source_rows]
        arrays["eras"][destination_rows] = validation.eras[source_rows]
        arrays["benchmarks"][destination_rows] = validation.benchmarks[source_rows]
    for array in arrays.values():
        array.flush()

    manifest = {
        "split": "production_train", "data_version": "v5.3",
        "feature_set": expected_feature_set, "feature_names": train_features,
        "targets": train.manifest["targets"], "benchmarks": train.manifest["benchmarks"],
        "rows": total, "eras": len(np.unique(arrays["eras"])),
        "era_min": int(arrays["eras"].min()), "era_max": int(arrays["eras"].max()),
        "frozen_train_rows": boundary,
        "resolved_validation_rows": len(validation_rows),
        "freeze_manifest_sha256": freeze_hash,
        "sealed_evaluation_sha256": sha256(evaluation_marker),
        "training_data_sha256": training_data_sha256,
        "identity_inputs": identity_inputs,
    }
    atomic_json(temporary / "manifest.json", manifest)
    ProductionShard.open(temporary)
    os.replace(temporary, destination)
    ProductionShard.open(destination)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--evaluation-marker", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build_production_shard(
        args.train, args.validation, args.freeze, args.evaluation_marker, args.destination,
    ), sort_keys=True))


if __name__ == "__main__":
    main()
