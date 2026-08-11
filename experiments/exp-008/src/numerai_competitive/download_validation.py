"""Download pinned official validation artifacts only after procedure freeze."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from numerapi import NumerAPI

from .data import (
    FEATURES_SHA256,
    VALIDATION_BENCH_SHA256,
    VALIDATION_SHA256,
    _validate_freeze_manifest,
    atomic_json,
    sha256,
)

DATASETS = {
    "v5.3/validation.parquet": ("validation.parquet", VALIDATION_SHA256),
    "v5.3/validation_benchmark_models.parquet": (
        "validation_benchmark_models.parquet", VALIDATION_BENCH_SHA256,
    ),
    "v5.3/features.json": ("features.json", FEATURES_SHA256),
}


def download_validation(destination: Path, freeze_manifest: Path,
                        api: NumerAPI | None = None) -> dict:
    freeze = _validate_freeze_manifest(freeze_manifest)
    api = NumerAPI() if api is None else api
    available = set(api.list_datasets())
    missing = sorted(set(DATASETS) - available)
    if missing:
        raise ValueError(f"official API lacks pinned datasets: {missing}")
    destination.mkdir(parents=True, exist_ok=True)
    artifacts = {}
    for dataset, (filename, expected_hash) in DATASETS.items():
        final = destination / filename
        if final.exists():
            if sha256(final) != expected_hash:
                raise ValueError(f"existing {filename} differs from frozen hash")
            mode = "reused_verified"
        else:
            temporary = final.with_suffix(final.suffix + ".download")
            if temporary.exists():
                temporary.unlink()
            api.download_dataset(dataset, str(temporary))
            actual_hash = sha256(temporary)
            if actual_hash != expected_hash:
                temporary.unlink(missing_ok=True)
                raise ValueError(f"downloaded {dataset} hash mismatch")
            os.replace(temporary, final)
            mode = "downloaded_verified"
        artifacts[filename] = {
            "dataset": dataset, "sha256": expected_hash,
            "bytes": final.stat().st_size, "mode": mode,
        }
    report = {
        "status": "complete", "data_version": "v5.3",
        "freeze_manifest": str(freeze_manifest.resolve()),
        "freeze_manifest_sha256": sha256(freeze_manifest),
        "freeze_code_commit": freeze["code_commit"], "artifacts": artifacts,
    }
    atomic_json(destination / "download-complete.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(download_validation(args.destination, args.freeze), sort_keys=True))


if __name__ == "__main__":
    main()
