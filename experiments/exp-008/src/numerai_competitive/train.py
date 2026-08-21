"""One-fold, one-configuration Exp-008 training runner."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import signal
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from . import PRIMARY_BENCHMARK
from .data import ProductionShard, TrainShard, atomic_json
from .filters import StreamingSpectralFilter
from .metrics import per_era_corr, per_era_correlation_contribution, summarize_era_scores
from .model import MLPConfig, ResidualMLP
from .splits import EraSplit


@dataclass(frozen=True)
class TrainConfig:
    model: MLPConfig
    search_config_id: int | None = None
    feature_set: str | None = None
    arm: str = "adamw"
    target: str = "target"
    benchmark: str = PRIMARY_BENCHMARK
    seed: int = 0
    device: str = "auto"
    batch_mode: str = "row"
    batch_size: int = 1024
    updates: int = 1000
    examples: int | None = None
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    loss: str = "mse"
    huber_delta: float = 1.0
    schedule: str = "constant"
    schedule_updates: int | None = None
    warmup_updates: int = 0
    clip_norm: float | None = None
    eval_batch_size: int = 4096
    log_every: int = 50
    checkpoint_every: int = 250
    validation_updates: tuple[int, ...] = ()
    save_model: bool = False
    filter: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.arm not in {"adamw", "spectral"}:
            raise ValueError("arm must be adamw or spectral")
        if self.batch_mode not in {"row", "era"}:
            raise ValueError("batch_mode must be row or era")
        if self.loss not in {"mse", "huber"} or self.schedule not in {"constant", "cosine"}:
            raise ValueError("unsupported loss or schedule")
        if min(self.batch_size, self.updates, self.eval_batch_size, self.log_every) < 1:
            raise ValueError("batch sizes, updates, and log interval must be positive")
        expected = self.updates * self.batch_size
        if self.examples is not None and self.examples != expected:
            raise ValueError(f"examples must equal updates * batch_size ({expected})")
        horizon = self.updates if self.schedule_updates is None else self.schedule_updates
        if horizon < self.updates:
            raise ValueError("schedule_updates must be at least updates")
        if not 0 <= self.warmup_updates <= horizon:
            raise ValueError("warmup_updates must be between zero and the schedule horizon")
        if self.clip_norm is not None and self.clip_norm <= 0:
            raise ValueError("clip_norm must be positive")
        if (tuple(sorted(set(self.validation_updates))) != tuple(self.validation_updates)
                or any(update < 1 or update > self.updates
                       for update in self.validation_updates)):
            raise ValueError("validation_updates must be unique, increasing, and within updates")

    @property
    def example_budget(self) -> int:
        return self.updates * self.batch_size


class BatchStream:
    """Deterministic fixed-size row or single-era batches, sampled with replacement."""

    def __init__(self, indices: np.ndarray, eras: np.ndarray, batch_size: int,
                 mode: str, seed: int) -> None:
        self.indices = np.asarray(indices, dtype=np.int64)
        if not len(self.indices):
            raise ValueError("training split has no rows")
        self.batch_size, self.mode = int(batch_size), mode
        self.rng = np.random.default_rng(seed)
        self.by_era = [self.indices[np.asarray(eras[self.indices]) == era]
                       for era in np.unique(np.asarray(eras[self.indices]))]

    def next(self) -> np.ndarray:
        pool = self.indices
        if self.mode == "era":
            pool = self.by_era[int(self.rng.integers(len(self.by_era)))]
        return self.rng.choice(pool, self.batch_size, replace=True)

    def state_dict(self) -> dict[str, Any]:
        return {"rng": self.rng.bit_generator.state}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.rng.bit_generator.state = state["rng"]


def _device(value: str) -> torch.device:
    selected = "cuda" if value == "auto" and torch.cuda.is_available() else value
    if selected == "auto":
        selected = "cpu"
    device = torch.device(selected)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return device


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def _lr_multiplier(update: int, config: TrainConfig) -> float:
    # LambdaLR calls this with update=0 before the first optimizer step.
    if config.warmup_updates and update < config.warmup_updates:
        return (update + 1) / config.warmup_updates
    if config.schedule == "constant":
        return 1.0
    horizon = config.updates if config.schedule_updates is None else config.schedule_updates
    span = max(1, horizon - config.warmup_updates)
    progress = min(1.0, max(0.0, (update - config.warmup_updates) / span))
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def _atomic_torch(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        torch.save(value, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _config_hash(config: TrainConfig, split: EraSplit,
                 training_data_sha256: str | None = None) -> str:
    value = {"config": asdict(config), "split": split.to_dict()}
    if training_data_sha256 is not None:
        value["training_data_sha256"] = training_data_sha256
    payload = json.dumps(value, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def _predict(model: torch.nn.Module, shard: TrainShard, indices: np.ndarray,
             device: torch.device, batch_size: int) -> np.ndarray:
    model.eval()
    result = []
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            rows = indices[start:start + batch_size]
            inputs = torch.from_numpy(np.asarray(shard.X[rows], dtype=np.float32) / 4.0).to(device)
            result.append(model(inputs).float().cpu().numpy())
    return np.concatenate(result).astype(np.float32, copy=False)


def _evaluate(predictions: np.ndarray, shard: TrainShard, indices: np.ndarray,
              target_index: int, benchmark_index: int) -> tuple[dict[str, Any], pd.DataFrame]:
    ids = pd.Index([f"row_{int(row)}" for row in indices], name="id")
    pred = pd.Series(predictions.astype(np.float64), index=ids, name="prediction")
    target = pd.Series(np.asarray(shard.targets[indices, target_index], dtype=np.float64),
                       index=ids, name="target")
    eras = pd.Series([f"{int(x):04d}" for x in shard.eras[indices]], index=ids, name="era")
    benchmark = pd.Series(np.asarray(shard.benchmarks[indices, benchmark_index], dtype=np.float64),
                          index=ids, name="benchmark")
    corr = per_era_corr(pred, target, eras)["prediction"]
    benchmark_covered = benchmark.notna()
    bmc = per_era_correlation_contribution(
        pred[benchmark_covered], benchmark[benchmark_covered],
        target[benchmark_covered], eras[benchmark_covered]
    )["prediction"]
    per_era = pd.DataFrame({"corr": corr, "bmc": bmc})
    summary = {
        "rows": len(indices), "eras": int(eras.nunique()),
        "mse": float(np.mean((predictions - target.to_numpy()) ** 2)),
        "corr": summarize_era_scores(corr).loc["prediction"].to_dict(),
        "bmc": summarize_era_scores(bmc).loc["prediction"].to_dict(),
        "bmc_rows": int(benchmark_covered.sum()),
        "bmc_eras": int(eras[benchmark_covered].nunique()),
    }
    return summary, per_era


def run_training(shard_root: Path, split: EraSplit, config: TrainConfig, output_dir: Path,
                 *, resume: bool = True, stop_after_updates: int | None = None,
                 refit: bool = False, production_refit: bool = False) -> dict[str, Any]:
    """Train exactly one frozen-shard split/config and write restart-safe artifacts."""
    if "validation" in str(Path(shard_root)).lower() and not production_refit:
        raise ValueError("validation paths are sealed from the training runner")
    if production_refit and not refit:
        raise ValueError("production data is allowed only for a target-free refit")
    _seed_everything(config.seed)
    shard = (ProductionShard.open(Path(shard_root)) if production_refit
             else TrainShard.open(Path(shard_root)))
    if config.model.input_dim != shard.X.shape[1]:
        raise ValueError("model input_dim does not match shard features")
    if config.feature_set is not None and config.feature_set != shard.manifest.get("feature_set"):
        raise ValueError("training config feature set does not match shard")
    train_indices = shard.row_indices(list(split.train_eras))
    valid_indices = shard.row_indices(list(split.valid_eras))
    if refit and (len(valid_indices) or not config.save_model):
        raise ValueError("refit requires no validation eras and save_model=True")
    if not refit and not len(valid_indices):
        raise ValueError("validation fold has no shard rows")
    device = _device(config.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model = ResidualMLP(config.model).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
                                  weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda update: _lr_multiplier(update, config)
    )
    spectral = (StreamingSpectralFilter(model, **config.filter)
                if config.arm == "spectral" else None)
    stream = BatchStream(train_indices, shard.eras, config.batch_size,
                         config.batch_mode, config.seed + 1)
    output_dir = Path(output_dir)
    checkpoint_path = output_dir / "checkpoint.pt"
    result_path = output_dir / "result.json"
    prediction_path = output_dir / "validation_predictions.npz"
    model_path = output_dir / "model.pt"
    checkpoint_status_path = output_dir / "checkpoint-status.json"
    training_data_sha256 = shard.manifest.get("training_data_sha256")
    signature = _config_hash(config, split, training_data_sha256)
    logs: list[dict[str, Any]] = []
    validation_history: list[dict[str, Any]] = []
    start_update = 0
    if resume and checkpoint_path.exists():
        saved = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if saved["signature"] != signature:
            raise ValueError("checkpoint config/split mismatch")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        if spectral is not None:
            spectral.load_state_dict(saved["filter"])
        stream.load_state_dict(saved["batch_stream"])
        random.setstate(saved["python_rng"])
        np.random.set_state(saved["numpy_rng"])
        torch.set_rng_state(saved["torch_rng"].cpu())
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
        start_update, logs = int(saved["update"]), saved["logs"]
        validation_history = saved.get("validation_history", [])

    limit = config.updates if stop_after_updates is None else min(config.updates, stop_after_updates)
    started = time.perf_counter()

    def save_checkpoint(update: int, reason: str) -> dict:
        _atomic_torch(checkpoint_path, {
            "signature": signature, "update": update, "model": model.state_dict(),
            "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            "filter": None if spectral is None else spectral.state_dict(),
            "batch_stream": stream.state_dict(), "python_rng": random.getstate(),
            "numpy_rng": np.random.get_state(), "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
            "logs": logs, "validation_history": validation_history,
        })
        status = {"status": "checkpointed", "reason": reason, "update": update,
                  "signature": signature}
        atomic_json(checkpoint_status_path, status)
        return status

    stop_requested = False
    previous_signal_handler = None
    if threading.current_thread() is threading.main_thread() and hasattr(signal, "SIGUSR1"):
        previous_signal_handler = signal.getsignal(signal.SIGUSR1)

        def request_stop(_signum, _frame) -> None:
            nonlocal stop_requested
            stop_requested = True

        signal.signal(signal.SIGUSR1, request_stop)
    try:
        for update in range(start_update + 1, limit + 1):
            rows = stream.next()
            inputs = torch.from_numpy(np.asarray(shard.X[rows], dtype=np.float32) / 4.0).to(device)
            targets = torch.from_numpy(np.asarray(
                shard.targets[rows, shard.target_index(config.target)], dtype=np.float32
            )).to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(inputs)
            loss = (F.mse_loss(prediction, targets) if config.loss == "mse" else
                    F.huber_loss(prediction, targets, delta=config.huber_delta))
            loss.backward()
            unclipped_norm = float(torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.clip_norm
                if config.clip_norm is not None else float("inf")
            ))
            filter_diagnostics = spectral.filter_grad() if spectral is not None else None
            should_log = update == 1 or update == limit or update % config.log_every == 0
            before = ([parameter.detach().clone() for parameter in model.parameters()]
                      if should_log else None)
            optimizer.step()
            scheduler.step()
            if should_log:
                update_sq = sum(float((parameter.detach() - old).double().square().sum())
                                for parameter, old in zip(model.parameters(), before))
                row: dict[str, Any] = {
                    "update": update, "examples": update * config.batch_size,
                    "loss": float(loss.detach()), "learning_rate": scheduler.get_last_lr()[0],
                    "pre_clip_gradient_norm": unclipped_norm,
                    "adamw_parameter_update_norm": math.sqrt(update_sq),
                    "elapsed_seconds": time.perf_counter() - started,
                }
                if filter_diagnostics is not None:
                    row["filter"] = filter_diagnostics
                logs.append(row)
            if not refit and update in config.validation_updates:
                checkpoint_predictions = _predict(
                    model, shard, valid_indices, device, config.eval_batch_size
                )
                checkpoint_validation, _ = _evaluate(
                    checkpoint_predictions, shard, valid_indices,
                    shard.target_index(config.target), shard.benchmark_index(config.benchmark)
                )
                validation_history.append({
                    "update": update,
                    "examples": update * config.batch_size,
                    "validation": checkpoint_validation,
                })
            if stop_requested and update < config.updates:
                return save_checkpoint(update, "SIGUSR1")
            if (config.checkpoint_every and update < config.updates
                    and update % config.checkpoint_every == 0):
                _atomic_torch(checkpoint_path, {
                    "signature": signature, "update": update, "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                    "filter": None if spectral is None else spectral.state_dict(),
                    "batch_stream": stream.state_dict(), "python_rng": random.getstate(),
                    "numpy_rng": np.random.get_state(), "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
                    "logs": logs, "validation_history": validation_history,
                })
    finally:
        if previous_signal_handler is not None:
            signal.signal(signal.SIGUSR1, previous_signal_handler)
    if limit < config.updates:
        return save_checkpoint(limit, "stop_after_updates")

    validation = None
    if not refit:
        predictions = _predict(model, shard, valid_indices, device, config.eval_batch_size)
        validation, per_era = _evaluate(predictions, shard, valid_indices,
                                        shard.target_index(config.target),
                                        shard.benchmark_index(config.benchmark))
        _atomic_npz(
            prediction_path, row_index=valid_indices, era=np.asarray(shard.eras[valid_indices]),
            target=np.asarray(shard.targets[valid_indices, shard.target_index(config.target)]),
            benchmark=np.asarray(shard.benchmarks[valid_indices,
                                                 shard.benchmark_index(config.benchmark)]),
            prediction=predictions, per_era_name=per_era.index.to_numpy(str),
            per_era_corr=per_era["corr"].to_numpy(), per_era_bmc=per_era["bmc"].to_numpy(),
        )
    peak = (int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0)
    result = {
        "status": "complete", "signature": signature, "split": split.to_dict(),
        "config": asdict(config), "parameter_count": sum(p.numel() for p in model.parameters()),
        "updates": config.updates, "examples": config.example_budget,
        "validation": validation, "validation_history": validation_history,
        "logs": logs, "peak_cuda_memory_bytes": peak,
        "prediction_file": None if refit else prediction_path.name,
        "model_file": model_path.name if config.save_model else None,
        "training_data_sha256": training_data_sha256,
    }
    if config.save_model:
        _atomic_torch(model_path, {
            "signature": signature,
            "model_config": asdict(config.model),
            "model": model.state_dict(),
            "train_config": asdict(config),
            "train_split": split.to_dict(),
            "target": config.target,
            "feature_names": shard.manifest["feature_names"],
            "data_version": shard.manifest["data_version"],
            "training_data_sha256": training_data_sha256,
        })
    atomic_json(result_path, result)
    if checkpoint_path.exists():
        checkpoint_path.unlink()
    if checkpoint_status_path.exists():
        checkpoint_status_path.unlink()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    split_value = json.loads(args.split.read_text())
    split = EraSplit(split_value["name"], tuple(split_value["train_eras"]),
                     tuple(split_value["valid_eras"]), tuple(split_value.get("purged_eras", [])))
    value = json.loads(args.config.read_text())
    value["model"] = MLPConfig(**value["model"])
    result = run_training(args.shard, split, TrainConfig(**value), args.output,
                          resume=not args.no_resume)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
