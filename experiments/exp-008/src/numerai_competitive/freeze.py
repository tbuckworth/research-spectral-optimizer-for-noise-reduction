"""Create the immutable procedure/model manifest that unlocks validation."""
from __future__ import annotations

import argparse
import json
import re
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from . import PRIMARY_BENCHMARK
from .code_snapshot import verify_snapshot
from .data import atomic_json, sha256


def _verify_arm(arm: str, config_id: int, models: list[Path], seeds: list[int],
                updates: int, search: dict) -> dict:
    draws = [draw for draw in search["configs"]
             if draw["arm"] == arm and draw["config_id"] == config_id]
    if len(draws) != 1:
        raise ValueError(f"{arm}: expected one search draw")
    if len(models) != len(seeds) or len(set(seeds)) != len(seeds):
        raise ValueError(f"{arm}: model paths and unique frozen seeds must align")
    signatures, model_hashes = [], []
    for path, seed in zip(models, seeds):
        artifact = torch.load(path, map_location="cpu", weights_only=False)
        config, split = artifact["train_config"], artifact["train_split"]
        expected = {
            "arm": arm, "search_config_id": config_id, "seed": seed,
            "updates": updates, "target": "target", "benchmark": PRIMARY_BENCHMARK,
            "feature_set": draws[0]["feature_set"],
        }
        actual = {key: config[key] for key in expected}
        if actual != expected:
            raise ValueError(f"{arm} seed {seed}: model training provenance differs from freeze")
        if (split["name"] != "sealed_validation_refit_60d" or split["valid_eras"]
                or split["purged_eras"] != [f"{era:04d}" for era in range(559, 575)]):
            raise ValueError(f"{arm} seed {seed}: model is not a purged validation refit")
        if split["train_eras"] != [f"{era:04d}" for era in range(1, 559)]:
            raise ValueError(f"{arm} seed {seed}: refit does not contain eras 0001--0558")
        if artifact["data_version"] != "v5.3":
            raise ValueError(f"{arm} seed {seed}: wrong data version")
        signatures.append(artifact["signature"])
        model_hashes.append(sha256(path))
    return {
        "config_id": config_id, "updates": updates, "seeds": seeds,
        "feature_set": draws[0]["feature_set"], "model_signatures": signatures,
        "model_sha256": model_hashes,
    }


def create_freeze(search_path: Path, protocol_path: Path, candidate_path: Path,
                  output: Path, code_commit: str, code_snapshot_path: Path, code_root: Path,
                  adamw_config: int, spectral_config: int, adamw_models: list[Path],
                  spectral_models: list[Path], updates: int | dict[str, int],
                  seeds: list[int],
                  authorize_validation_reveal: bool) -> dict:
    if not authorize_validation_reveal:
        raise ValueError("validation reveal requires explicit authorization")
    if not re.fullmatch(r"[0-9a-f]{40}", code_commit):
        raise ValueError("code commit must be a full lowercase Git SHA")
    verify_snapshot(code_root, code_snapshot_path, code_commit)
    search = json.loads(search_path.read_text())
    candidate = json.loads(candidate_path.read_text())
    if candidate.get("status") != "frozen_train_only_selection":
        raise ValueError("candidate plan is not a frozen train-only selection")
    selected_candidate = candidate.get("selected", {})
    arm_updates = (
        {"adamw": updates, "spectral": updates}
        if isinstance(updates, int) else updates
    )
    if (set(arm_updates) != {"adamw", "spectral"}
            or any(value not in {5000, 20000, 100000} for value in arm_updates.values())):
        raise ValueError("freeze requires one allowed update budget per arm")
    if (selected_candidate.get("arm") not in {"adamw", "spectral"}
            or selected_candidate.get("benchmark") != PRIMARY_BENCHMARK
            or not 0 < selected_candidate.get("model_weight", 0) <= 1
            or not np.isclose(selected_candidate["model_weight"]
                              + selected_candidate.get("benchmark_weight", -1), 1)):
        raise ValueError("candidate plan has an invalid frozen transformation")
    manifest = {
        "status": "frozen", "protocol": search["protocol"], "code_commit": code_commit,
        "code_snapshot_sha256": sha256(code_snapshot_path),
        "created_at": datetime.now(UTC).isoformat(),
        "search_sha256": sha256(search_path), "fidelity_protocol_sha256": sha256(protocol_path),
        "primary_target": "target", "primary_benchmark": PRIMARY_BENCHMARK,
        "primary_metric": "exact standalone CORR",
        "validation_reveal_authorized": True,
        "candidate_plan_sha256": sha256(candidate_path),
        "candidate_transform": selected_candidate,
        "selected": {
            "adamw": _verify_arm(
                "adamw", adamw_config, adamw_models, seeds, arm_updates["adamw"], search
            ),
            "spectral": _verify_arm(
                "spectral", spectral_config, spectral_models, seeds,
                arm_updates["spectral"], search
            ),
        },
    }
    atomic_json(output, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--search", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--candidate-plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--code-snapshot", type=Path, required=True)
    parser.add_argument("--code-root", type=Path, default=Path("."))
    parser.add_argument("--adamw-config", type=int, required=True)
    parser.add_argument("--spectral-config", type=int, required=True)
    parser.add_argument("--adamw-model", type=Path, action="append", required=True)
    parser.add_argument("--spectral-model", type=Path, action="append", required=True)
    parser.add_argument("--updates", type=int)
    parser.add_argument("--adamw-updates", type=int)
    parser.add_argument("--spectral-updates", type=int)
    parser.add_argument("--seed", type=int, action="append", required=True)
    parser.add_argument("--authorize-validation-reveal", action="store_true")
    args = parser.parse_args()
    if args.updates is not None:
        if args.adamw_updates is not None or args.spectral_updates is not None:
            raise ValueError("--updates cannot be combined with arm-specific updates")
        updates: int | dict[str, int] = args.updates
    elif args.adamw_updates is not None and args.spectral_updates is not None:
        updates = {"adamw": args.adamw_updates, "spectral": args.spectral_updates}
    else:
        raise ValueError("provide --updates or both arm-specific update counts")
    manifest = create_freeze(
        args.search, args.protocol, args.candidate_plan, args.output, args.code_commit,
        args.code_snapshot, args.code_root,
        args.adamw_config, args.spectral_config, args.adamw_model, args.spectral_model,
        updates, args.seed, args.authorize_validation_reveal,
    )
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
