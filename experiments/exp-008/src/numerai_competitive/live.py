"""Target-free conventional and Model Upload prediction callable."""
from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import cloudpickle
import numpy as np
import pandas as pd
import torch

from .model import MLPConfig, ResidualMLP


class NumeraiMLPPredictor:
    """CPU-only callable with the official two-argument Model Upload interface."""

    def __init__(self, artifacts: dict | Sequence[dict], batch_size: int = 4096) -> None:
        artifacts = [artifacts] if isinstance(artifacts, dict) else list(artifacts)
        if not artifacts:
            raise ValueError("at least one model artifact is required")
        self.feature_names = tuple(artifacts[0]["feature_names"])
        self.data_version = artifacts[0]["data_version"]
        self.batch_size = int(batch_size)
        self.models = []
        for artifact in artifacts:
            if tuple(artifact["feature_names"]) != self.feature_names:
                raise ValueError("ensemble model feature schemas differ")
            if artifact["data_version"] != self.data_version:
                raise ValueError("ensemble model data versions differ")
            model = ResidualMLP(MLPConfig(**artifact["model_config"]))
            model.load_state_dict(artifact["model"])
            model.eval()
            model.to("cpu")
            self.models.append(model)

    @classmethod
    def from_file(cls, path: Path, batch_size: int = 4096) -> NumeraiMLPPredictor:
        return cls.from_files([path], batch_size=batch_size)

    @classmethod
    def from_files(cls, paths: Sequence[Path], batch_size: int = 4096) -> NumeraiMLPPredictor:
        artifacts = [torch.load(path, map_location="cpu", weights_only=False) for path in paths]
        return cls(artifacts, batch_size=batch_size)

    def _features(self, frame: pd.DataFrame) -> np.ndarray:
        missing = sorted(set(self.feature_names) - set(frame.columns))
        if missing:
            raise ValueError(f"live frame is missing {len(missing)} required features")
        raw = frame.loc[:, self.feature_names].to_numpy(dtype=np.float32, copy=True)
        raw = np.nan_to_num(raw, nan=0.5)
        if raw.size and float(raw.max()) > 1.0:
            if raw.min() < 0 or raw.max() > 4 or not np.allclose(raw, np.rint(raw), atol=1e-6):
                raise ValueError("live integer-bin features must be in 0..4")
            raw /= 4.0
        if raw.size and (raw.min() < 0 or raw.max() > 1):
            raise ValueError("live features must be normalized to [0,1]")
        return raw

    def __call__(self, live_features: pd.DataFrame,
                 live_benchmark_models: pd.DataFrame | None = None) -> pd.DataFrame:
        del live_benchmark_models  # Reserved for a separately frozen blend.
        torch.set_num_threads(1)
        values = self._features(live_features)
        predictions = [[] for _ in self.models]
        with torch.inference_mode():
            for start in range(0, len(values), self.batch_size):
                batch = torch.from_numpy(values[start:start + self.batch_size])
                for model_index, model in enumerate(self.models):
                    predictions[model_index].append(model(batch).numpy())
        ranked_models = []
        for chunks in predictions:
            raw = np.concatenate(chunks) if chunks else np.empty(0, dtype=np.float32)
            if not np.isfinite(raw).all():
                raise ValueError("model emitted non-finite predictions")
            # Match sealed evaluation: rank each seed model, then average the seed ranks.
            ranked = pd.Series(raw).rank(method="average").to_numpy(dtype=np.float64)
            ranked_models.append((ranked - 0.5) / max(1, len(ranked)))
        ensemble = np.mean(ranked_models, axis=0)
        return pd.DataFrame({"prediction": ensemble}, index=live_features.index)


def export_callable(model_artifacts: Sequence[Path], output: Path,
                    batch_size: int = 4096) -> Path:
    predictor = NumeraiMLPPredictor.from_files(model_artifacts, batch_size=batch_size)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as handle:
        cloudpickle.dump(predictor, handle)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4096)
    args = parser.parse_args()
    print(export_callable(args.model, args.output, args.batch_size))


if __name__ == "__main__":
    main()
