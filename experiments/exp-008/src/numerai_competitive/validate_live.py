"""Validate a frozen callable against the target-free live schema and runtime limits."""
from __future__ import annotations

import argparse
import json
import os
import resource
import time
from pathlib import Path

import cloudpickle
import numpy as np
import pandas as pd

from .data import atomic_json, sha256


def validate(callable_path: Path, live_path: Path, output: Path,
             benchmark_path: Path | None = None, max_bytes: int = 4_000_000_000,
             max_seconds: float = 600.0,
             prediction_output: Path | None = None) -> dict:
    started = time.perf_counter()
    with callable_path.open("rb") as handle:
        predictor = cloudpickle.load(handle)
    # Match the official numerai-predict runner, which loads the complete parquet before
    # invoking the callable. Column projection here would understate peak memory materially.
    live = pd.read_parquet(live_path)
    benchmark = None
    if predictor.model_weight < 1:
        if benchmark_path is None:
            raise ValueError("frozen blend requires a live benchmark parquet")
        benchmark = pd.read_parquet(benchmark_path)
    predictions = predictor(live, benchmark)
    if not predictions.index.equals(live.index):
        raise ValueError("callable output IDs/order differ from live features")
    if list(predictions.columns) != ["prediction"] or len(predictions) != len(live):
        raise ValueError("callable output must have exactly one prediction per live row")
    values = predictions["prediction"].to_numpy(float)
    if not np.isfinite(values).all() or not ((values > 0) & (values < 1)).all():
        raise ValueError("live predictions must be finite and strictly inside (0,1)")
    prediction_sha256 = None
    if prediction_output is not None:
        prediction_output.parent.mkdir(parents=True, exist_ok=True)
        temporary = prediction_output.with_suffix(prediction_output.suffix + ".tmp")
        predictions.to_csv(temporary, index=True, index_label=live.index.name or "id")
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, prediction_output)
        prediction_sha256 = sha256(prediction_output)
    elapsed = time.perf_counter() - started
    artifact_bytes = callable_path.stat().st_size
    peak_rss_bytes = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
    report = {
        "status": "pass" if artifact_bytes < max_bytes and elapsed < max_seconds
        and peak_rss_bytes < max_bytes else "fail",
        "rows": len(live), "artifact_bytes": artifact_bytes,
        "artifact_sha256": sha256(callable_path), "elapsed_seconds": elapsed,
        "peak_rss_bytes": peak_rss_bytes, "max_bytes": max_bytes,
        "max_seconds": max_seconds, "model_count": len(predictor.models),
        "model_signatures": list(predictor.model_signatures),
        "model_weight": predictor.model_weight, "benchmark_name": predictor.benchmark_name,
        "prediction_file": str(prediction_output) if prediction_output else None,
        "prediction_sha256": prediction_sha256,
    }
    atomic_json(output, report)
    if report["status"] != "pass":
        raise RuntimeError(f"Model Upload resource contract failed: {report}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--callable", type=Path, required=True)
    parser.add_argument("--live", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-bytes", type=int, default=4_000_000_000)
    parser.add_argument("--max-seconds", type=float, default=600.0)
    parser.add_argument("--predictions", type=Path)
    args = parser.parse_args()
    print(json.dumps(validate(args.callable, args.live, args.output, args.benchmark,
                              args.max_bytes, args.max_seconds, args.predictions), sort_keys=True))


if __name__ == "__main__":
    main()
