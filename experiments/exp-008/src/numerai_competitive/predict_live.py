"""Generate a conventional Numerai live CSV without uploading or submitting it."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import cloudpickle
import pandas as pd


def predict(callable_path: Path, live_path: Path, output: Path,
            benchmark_path: Path | None = None) -> Path:
    with callable_path.open("rb") as handle:
        predictor = cloudpickle.load(handle)
    live = pd.read_parquet(live_path, columns=list(predictor.feature_names))
    benchmark = None
    if predictor.model_weight < 1:
        if benchmark_path is None:
            raise ValueError("frozen blend requires a live benchmark parquet")
        benchmark = pd.read_parquet(benchmark_path, columns=[predictor.benchmark_name])
    predictions = predictor(live, benchmark)
    if not predictions.index.equals(live.index) or predictions.index.has_duplicates:
        raise ValueError("prediction IDs/order must exactly match unique live IDs")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    predictions.to_csv(temporary, index=True, index_label=live.index.name or "id")
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--callable", type=Path, required=True)
    parser.add_argument("--live", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(predict(args.callable, args.live, args.output, args.benchmark))


if __name__ == "__main__":
    main()
