from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from numerai_competitive.oof import aggregate


def _result(root: Path, arm: str, split: str, seed: int, rows: np.ndarray,
            eras: np.ndarray, *, corrupt_target: bool = False) -> Path:
    directory = root / f"{arm}-{split}-s{seed}"
    directory.mkdir(parents=True)
    rng = np.random.default_rng(seed + (0 if arm == "adamw" else 20))
    target = ((rows * 17) % 101 / 101).astype(np.float64)
    if corrupt_target:
        target[0] += 0.01
    benchmark = ((rows * 29 + 3) % 103 / 103).astype(np.float64)
    signal = target + rng.normal(0, 0.15 if arm == "spectral" else 0.2, len(rows))
    np.savez_compressed(
        directory / "validation_predictions.npz", row_index=rows, era=eras,
        target=target, benchmark=benchmark, prediction=signal,
    )
    (directory / "result.json").write_text(json.dumps({
        "status": "complete", "prediction_file": "validation_predictions.npz",
        "config": {"arm": arm, "seed": seed, "search_config_id": 7},
        "split": {"name": split},
    }))
    return directory


def _paths(tmp_path: Path, corrupt: bool = False):
    paths = {"adamw": [], "spectral": []}
    for split_index, split in enumerate(("outer_1", "outer_2")):
        first_era = 10 + split_index * 4
        eras = np.repeat(np.arange(first_era, first_era + 4), 20)
        rows = np.arange(split_index * 80, (split_index + 1) * 80)
        for arm, arm_paths in paths.items():
            for seed in (0, 1):
                arm_paths.append(_result(
                    tmp_path, arm, split, seed, rows, eras,
                    corrupt_target=corrupt and arm == "spectral" and split_index == seed == 0,
                ))
    return paths


def test_nested_outer_aggregation_is_aligned_seed_ensembled_and_audited(tmp_path: Path):
    paths = _paths(tmp_path)
    output = tmp_path / "out"
    report = aggregate(paths["adamw"], paths["spectral"], output, (0, 1))
    assert report["status"] == "complete"
    assert report["rows"] == 160 and report["eras"] == 8
    assert report["outer_splits"] == ["outer_1", "outer_2"]
    assert report["spectral_minus_adamw"]["samples"] == 10_000
    assert (output / "nested-outer-corr.png").is_file()
    assert (output / "nested-outer-predictions.npz").is_file()


def test_nested_outer_aggregation_rejects_cross_arm_target_mismatch(tmp_path: Path):
    paths = _paths(tmp_path, corrupt=True)
    with pytest.raises(ValueError, match="disagree on target"):
        aggregate(paths["adamw"], paths["spectral"], tmp_path / "out", (0, 1))
