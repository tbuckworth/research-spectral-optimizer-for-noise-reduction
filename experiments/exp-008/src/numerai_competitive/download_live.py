"""Download one internally consistent target-free live round after final freeze."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd
from numerapi import NumerAPI

from .data import _validate_freeze_manifest, atomic_json, sha256

DATASETS = {
    "v5.3/live.parquet": "live.parquet",
    "v5.3/live_benchmark_models.parquet": "live_benchmark_models.parquet",
}


def download_live(destination: Path, freeze_manifest: Path,
                  api: NumerAPI | None = None) -> dict:
    freeze = _validate_freeze_manifest(freeze_manifest)
    api = NumerAPI() if api is None else api
    round_before = int(api.get_current_round())
    available = set(api.list_datasets())
    missing = sorted(set(DATASETS) - available)
    if missing:
        raise ValueError(f"official API lacks live datasets: {missing}")
    destination.mkdir(parents=True, exist_ok=True)
    temporary_paths = {}
    try:
        for dataset, filename in DATASETS.items():
            final = destination / filename
            if final.exists():
                raise ValueError(f"live destination already contains {filename}")
            temporary = final.with_suffix(final.suffix + ".download")
            temporary.unlink(missing_ok=True)
            api.download_dataset(dataset, str(temporary))
            temporary_paths[filename] = temporary
        round_after = int(api.get_current_round())
        if round_after != round_before:
            raise RuntimeError(
                f"live round changed during download: {round_before} -> {round_after}"
            )
        live = pd.read_parquet(temporary_paths["live.parquet"])
        benchmark = pd.read_parquet(temporary_paths["live_benchmark_models.parquet"])
        if (live.empty or live.index.has_duplicates or benchmark.index.has_duplicates
                or not live.index.equals(benchmark.index)):
            raise ValueError("live and benchmark IDs are empty, duplicated, or misaligned")
        if any(column.startswith("target") for column in live.columns):
            raise ValueError("live fixture unexpectedly contains target columns")
        if "v53_lgbm_ender20" not in benchmark.columns:
            raise ValueError("live benchmark lacks frozen Ender20 column")
        if benchmark["v53_lgbm_ender20"].isna().any():
            raise ValueError("live Ender20 benchmark contains missing predictions")
        artifacts = {}
        for filename, temporary in temporary_paths.items():
            final = destination / filename
            digest = sha256(temporary)
            size = temporary.stat().st_size
            os.replace(temporary, final)
            artifacts[filename] = {"sha256": digest, "bytes": size}
        report = {
            "status": "complete", "round": round_before, "data_version": "v5.3",
            "rows": len(live), "freeze_manifest_sha256": sha256(freeze_manifest),
            "freeze_code_commit": freeze["code_commit"], "artifacts": artifacts,
        }
        atomic_json(destination / "download-complete.json", report)
        return report
    finally:
        for temporary in temporary_paths.values():
            temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(download_live(args.destination, args.freeze), sort_keys=True))


if __name__ == "__main__":
    main()
