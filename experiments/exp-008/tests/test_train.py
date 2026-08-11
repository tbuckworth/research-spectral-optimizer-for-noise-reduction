from __future__ import annotations

import json
import signal
from pathlib import Path

import numpy as np
import pytest
import torch

from numerai_competitive.model import MLPConfig, ResidualMLP
from numerai_competitive.splits import EraSplit
from numerai_competitive.train import BatchStream, TrainConfig, _evaluate, run_training


def _shard(root: Path) -> Path:
    rng = np.random.default_rng(4)
    rows, features = 48, 5
    np.save(root / "X_u8.npy", rng.integers(0, 5, (rows, features), dtype=np.uint8))
    targets = np.column_stack([rng.uniform(0, 1, rows), rng.uniform(0, 1, rows)]).astype(np.float32)
    np.save(root / "targets_f32.npy", targets)
    np.save(root / "era_i16.npy", np.repeat(np.arange(1, 7), 8).astype(np.int16))
    np.save(root / "benchmarks_f32.npy", rng.uniform(0, 1, (rows, 1)).astype(np.float32))
    (root / "manifest.json").write_text(json.dumps({
        "split": "train", "data_version": "v5.3", "rows": rows,
        "feature_names": [f"f{i}" for i in range(features)],
        "targets": ["target", "other"], "benchmarks": ["benchmark"],
    }))
    return root


def _config(arm="adamw", **changes):
    values = {
        "model": MLPConfig(5, width=8, depth=2, dropout=0.1), "arm": arm,
        "benchmark": "benchmark", "device": "cpu", "seed": 12, "batch_size": 8,
        "updates": 6, "examples": 48, "learning_rate": 2e-3, "warmup_updates": 2,
        "schedule": "cosine", "clip_norm": 1.0, "log_every": 1,
        "checkpoint_every": 2, "eval_batch_size": 8,
    }
    values.update(changes)
    return TrainConfig(**values)


SPLIT = EraSplit("synthetic", ("1", "2", "3", "4"), ("5", "6"), ())


def test_residual_mlp_configurations_and_budget_validation():
    for residual in (False, True):
        model = ResidualMLP(MLPConfig(5, width=7, depth=3, residual=residual,
                                      normalization="none", activation="silu"))
        assert model(torch.ones(4, 5)).shape == (4,)
    with pytest.raises(ValueError, match=r"updates \* batch_size"):
        _config(examples=47)


def test_era_batches_are_contained_and_restartable():
    eras = np.repeat(np.arange(4), 5)
    first = BatchStream(np.arange(20), eras, 7, "era", 3)
    for _ in range(4):
        assert len(np.unique(eras[first.next()])) == 1
    state = first.state_dict()
    restarted = BatchStream(np.arange(20), eras, 7, "era", 3)
    restarted.load_state_dict(state)
    np.testing.assert_array_equal(first.next(), restarted.next())


@pytest.mark.parametrize("arm", ["adamw", "spectral"])
def test_full_cpu_run_writes_exact_metrics_and_atomic_predictions(tmp_path: Path, arm: str):
    shard = _shard(tmp_path / "shard") if (tmp_path / "shard").mkdir() is None else None
    config = _config(arm, filter={"rank": 3, "warmup": 1, "seed": 2})
    result = run_training(shard, SPLIT, config, tmp_path / "out")
    assert result["status"] == "complete"
    assert result["updates"] == 6 and result["examples"] == 48
    assert len(result["logs"]) == 6 and result["peak_cuda_memory_bytes"] == 0
    assert np.isfinite(result["validation"]["corr"]["mean"])
    saved = np.load(tmp_path / "out" / "validation_predictions.npz")
    assert len(saved["prediction"]) == 16 and len(saved["per_era_corr"]) == 2
    assert not list((tmp_path / "out").glob("*.tmp"))


def test_checkpoint_resume_is_deterministic(tmp_path: Path):
    shard_dir = tmp_path / "shard"
    shard_dir.mkdir()
    shard = _shard(shard_dir)
    config = _config("spectral", filter={"rank": 3, "warmup": 0, "seed": 8})
    partial = run_training(shard, SPLIT, config, tmp_path / "resume", stop_after_updates=3)
    assert partial["status"] == "checkpointed"
    resumed = run_training(shard, SPLIT, config, tmp_path / "resume")
    direct = run_training(shard, SPLIT, config, tmp_path / "direct")
    without_time = lambda logs: [
        {key: value for key, value in row.items() if key != "elapsed_seconds"} for row in logs
    ]
    assert without_time(resumed["logs"]) == without_time(direct["logs"])
    left = np.load(tmp_path / "resume" / "validation_predictions.npz")["prediction"]
    right = np.load(tmp_path / "direct" / "validation_predictions.npz")["prediction"]
    np.testing.assert_array_equal(left, right)


def test_sigusr1_creates_restartable_checkpoint(tmp_path: Path, monkeypatch):
    shard_dir = tmp_path / "shard"
    shard_dir.mkdir()
    shard = _shard(shard_dir)
    original_next = BatchStream.next
    calls = 0

    def interrupted_next(self):
        nonlocal calls
        calls += 1
        rows = original_next(self)
        if calls == 2:
            signal.raise_signal(signal.SIGUSR1)
        return rows

    previous = signal.getsignal(signal.SIGUSR1)
    monkeypatch.setattr(BatchStream, "next", interrupted_next)
    result = run_training(shard, SPLIT, _config(), tmp_path / "resume")
    assert result["status"] == "checkpointed" and result["reason"] == "SIGUSR1"
    assert json.loads((tmp_path / "resume" / "checkpoint-status.json").read_text())["update"] == 2
    assert signal.getsignal(signal.SIGUSR1) == previous
    resumed = run_training(shard, SPLIT, _config(), tmp_path / "resume")
    assert resumed["status"] == "complete"
    assert not (tmp_path / "resume" / "checkpoint-status.json").exists()


def test_corr_keeps_era_without_benchmark_coverage(tmp_path: Path):
    shard_dir = tmp_path / "shard"
    shard_dir.mkdir()
    shard = _shard(shard_dir)
    from numerai_competitive.data import TrainShard

    benchmarks = np.load(shard / "benchmarks_f32.npy")
    benchmarks[:8, 0] = np.nan
    np.save(shard / "benchmarks_f32.npy", benchmarks)
    opened = TrainShard.open(shard)
    summary, per_era = _evaluate(np.linspace(0, 1, 16), opened, np.arange(16), 0, 0)
    assert summary["eras"] == 2
    assert summary["bmc_eras"] == 1
    assert len(per_era["corr"].dropna()) == 2
    assert len(per_era["bmc"].dropna()) == 1


def test_validation_named_path_is_rejected(tmp_path: Path):
    with pytest.raises(ValueError, match="sealed"):
        run_training(tmp_path / "validation.parquet", SPLIT, _config(), tmp_path / "out")


def test_explicit_final_model_artifact_is_target_free(tmp_path: Path):
    shard_dir = tmp_path / "shard"
    shard_dir.mkdir()
    result = run_training(_shard(shard_dir), SPLIT, _config(save_model=True), tmp_path / "out")
    artifact = torch.load(tmp_path / "out" / result["model_file"], weights_only=False)
    assert set(artifact) == {
        "signature", "model_config", "model", "train_config", "train_split", "target",
        "feature_names", "data_version",
    }
    assert not any("target" in key for key in artifact["model"])
    assert artifact["train_config"]["seed"] == 12
    assert artifact["train_split"]["name"] == "synthetic"


def test_refit_requires_model_artifact_and_has_no_validation_score(tmp_path: Path):
    shard_dir = tmp_path / "shard"
    shard_dir.mkdir()
    split = EraSplit("all_train", ("1", "2", "3", "4", "5", "6"), (), ())
    result = run_training(_shard(shard_dir), split, _config(save_model=True),
                          tmp_path / "out", refit=True)
    assert result["validation"] is None
    assert result["prediction_file"] is None
    assert (tmp_path / "out" / "model.pt").is_file()
    with pytest.raises(ValueError, match="save_model"):
        run_training(shard_dir, split, _config(save_model=False), tmp_path / "bad", refit=True)
