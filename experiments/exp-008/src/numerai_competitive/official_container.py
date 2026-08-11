"""Audit predictions emitted by the pinned official numerai-predict container."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .data import atomic_json, sha256


def audit(expected_path: Path, official_path: Path, output: Path, *, runner_commit: str,
          image_id: str, elapsed_seconds: float, max_seconds: float = 600.0,
          max_abs_difference: float = 1e-4) -> dict:
    if not re.fullmatch(r"[0-9a-f]{40}", runner_commit):
        raise ValueError("runner commit must be a full lowercase Git SHA")
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", image_id):
        raise ValueError("official image ID must be content-addressed")
    expected = pd.read_csv(expected_path, index_col=0)
    official = pd.read_csv(official_path, index_col=0)
    for name, frame in (("expected", expected), ("official", official)):
        if list(frame.columns) != ["prediction"] or frame.empty or frame.index.has_duplicates:
            raise ValueError(f"{name} predictions have an invalid schema")
        values = frame["prediction"].to_numpy(float)
        if not np.isfinite(values).all() or not ((values >= 0) & (values <= 1)).all():
            raise ValueError(f"{name} predictions are non-finite or outside [0,1]")
    if not expected.index.equals(official.index):
        raise ValueError("official runner changed prediction IDs or row order")
    difference = np.abs(
        expected["prediction"].to_numpy(float) - official["prediction"].to_numpy(float)
    )
    correlation = float(expected["prediction"].corr(official["prediction"]))
    report = {
        "status": "pass" if elapsed_seconds < max_seconds
        and float(difference.max()) <= max_abs_difference else "fail",
        "rows": len(expected), "runner_commit": runner_commit, "image_id": image_id,
        "elapsed_seconds": float(elapsed_seconds), "max_seconds": float(max_seconds),
        "cpu_limit": 1, "memory_limit_bytes": 4_000_000_000,
        "max_abs_difference": float(difference.max()),
        "mean_abs_difference": float(difference.mean()), "prediction_correlation": correlation,
        "allowed_max_abs_difference": float(max_abs_difference),
        "expected_sha256": sha256(expected_path), "official_sha256": sha256(official_path),
    }
    atomic_json(output, report)
    if report["status"] != "pass":
        raise RuntimeError(f"official container compatibility failed: {report}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected", type=Path, required=True)
    parser.add_argument("--official", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runner-commit", required=True)
    parser.add_argument("--image-id", required=True)
    parser.add_argument("--elapsed-seconds", type=float, required=True)
    parser.add_argument("--max-seconds", type=float, default=600.0)
    parser.add_argument("--max-abs-difference", type=float, default=1e-4)
    args = parser.parse_args()
    print(json.dumps(audit(
        args.expected, args.official, args.output, runner_commit=args.runner_commit,
        image_id=args.image_id, elapsed_seconds=args.elapsed_seconds,
        max_seconds=args.max_seconds, max_abs_difference=args.max_abs_difference,
    ), sort_keys=True))


if __name__ == "__main__":
    main()
