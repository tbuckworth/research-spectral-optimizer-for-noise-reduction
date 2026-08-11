import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from numerai_competitive.data import sha256
from numerai_competitive.evaluate_frozen import _prediction_correlation, _rank_within_era, evaluate
from numerai_competitive.model import MLPConfig, ResidualMLP


def test_rank_within_era_preserves_order_ties_and_boundaries():
    values = np.array([3.0, 1.0, 1.0, 9.0, 8.0])
    eras = np.array([1, 1, 1, 2, 2])
    ranked = _rank_within_era(values, eras)
    np.testing.assert_allclose(ranked, [5 / 6, 1 / 3, 1 / 3, 3 / 4, 1 / 4])
    assert np.all((ranked > 0) & (ranked < 1))


def test_rank_within_era_matches_pandas_reference():
    rng = np.random.default_rng(7)
    values = rng.normal(size=30)
    eras = np.repeat([3, 7, 9], 10)
    expected = pd.Series(values).groupby(eras).rank(method="average")
    expected = (expected - 0.5) / 10
    np.testing.assert_allclose(_rank_within_era(values, eras), expected)


def test_prediction_correlation_reports_per_era_not_only_pooled_value():
    index = pd.Index([f"row_{i}" for i in range(8)])
    eras = pd.Series(["1"] * 4 + ["2"] * 4, index=index)
    benchmark = pd.Series([0.1, 0.2, 0.3, 0.4] * 2, index=index)
    candidate = np.array([0.1, 0.2, 0.3, 0.4, 0.4, 0.3, 0.2, 0.1])
    result = _prediction_correlation(candidate, benchmark, eras)
    assert result["eras"] == 2
    assert result["per_era_mean"] == pytest.approx(0.0, abs=1e-12)
    assert result["per_era_min"] == pytest.approx(-1.0)
    assert result["per_era_max"] == pytest.approx(1.0)


def _model_artifact(path: Path, signature: str, feature_names: list[str], seed: int) -> None:
    torch.manual_seed(seed)
    config = MLPConfig(input_dim=len(feature_names), width=7, depth=2)
    torch.save({
        "signature": signature, "feature_names": feature_names,
        "model_config": config.__dict__, "model": ResidualMLP(config).state_dict(),
    }, path)


def test_frozen_evaluator_scores_candidate_and_named_benchmark_column(tmp_path: Path):
    rng = np.random.default_rng(12)
    rows, feature_names = 160, ["a", "b", "c"]
    shard = tmp_path / "validation-shard"
    shard.mkdir()
    np.save(shard / "X_u8.npy", rng.integers(0, 5, (rows, 3), dtype=np.uint8))
    targets = rng.uniform(0, 1, (rows, 5)).astype(np.float32)
    targets[:, 4] = targets[:, 0]
    np.save(shard / "targets_f32.npy", targets)
    np.save(shard / "era_i16.npy", np.repeat(np.arange(100, 108), 20).astype(np.int16))
    # Ender60 is deliberately column 2: the evaluator must resolve by name.
    np.save(shard / "benchmarks_f32.npy", rng.uniform(0, 1, (rows, 3)).astype(np.float32))
    adamw, spectral = tmp_path / "adamw.pt", tmp_path / "spectral.pt"
    _model_artifact(adamw, "adamw-signature", ["a", "c"], 2)
    _model_artifact(spectral, "spectral-signature", feature_names, 3)
    freeze = tmp_path / "freeze.json"
    freeze.write_text(json.dumps({
        "selected": {
            "adamw": {"model_signatures": ["adamw-signature"], "seeds": [0],
                      "model_sha256": [sha256(adamw)]},
            "spectral": {"model_signatures": ["spectral-signature"], "seeds": [0],
                         "model_sha256": [sha256(spectral)]},
        },
        "candidate_transform": {
            "arm": "adamw", "model_weight": 0.5, "benchmark_weight": 0.5,
            "benchmark": "v53_lgbm_ender60",
        },
    }))
    (shard / "manifest.json").write_text(json.dumps({
        "split": "validation", "data_version": "v5.3", "feature_set": "all", "rows": rows,
        "feature_names": feature_names,
        "targets": ["target", "target_cyrusd_20", "target_ender_20",
                    "target_teager2b_20", "target_ender_60"],
        "benchmarks": ["decoy", "v53_lgbm_ender20", "v53_lgbm_ender60"],
        "freeze_manifest_sha256": sha256(freeze),
    }))
    output = tmp_path / "evaluation"
    report = evaluate(shard, freeze, [adamw], [spectral], output, "cpu", 32)
    assert report["status"] == "complete" and report["resolved_eras"] == 8
    assert report["candidate_transform"]["model_weight"] == 0.5
    assert report["target_alias_audit"]["target_equals_target_ender_60"] is True
    assert report["candidate_minus_ender60"]["samples"] == 10_000
    assert report["prediction_correlation"]["candidate_vs_ender60"]["eras"] == 8
    assert set(report["secondary"]) == {
        "target_cyrusd_20", "target_ender_20", "target_teager2b_20", "target_ender_60",
        "target_20_rank_ensemble",
    }
    assert report["secondary"]["target_ender_60"]["benchmark"] == "v53_lgbm_ender60"
    saved = np.load(output / "official-validation-predictions.npz")
    np.testing.assert_array_equal(saved["benchmark"], np.load(
        shard / "benchmarks_f32.npy"
    )[:, 2])
    assert (output / "official-validation-corr.png").is_file()
    marker = json.loads((output / "evaluation-complete.json").read_text())
    assert marker["status"] == "complete" and len(marker["artifacts"]) == 5
    artifact = torch.load(adamw, weights_only=False)
    first = next(iter(artifact["model"]))
    artifact["model"][first] = artifact["model"][first] + 1
    torch.save(artifact, adamw)
    with pytest.raises(ValueError, match="model hashes/order"):
        evaluate(shard, freeze, [adamw], [spectral], tmp_path / "tampered", "cpu", 32)
