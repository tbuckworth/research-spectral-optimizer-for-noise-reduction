"""Target-free conventional and Model Upload prediction callable."""
from __future__ import annotations

from pathlib import Path

import cloudpickle
import numpy as np
import pandas as pd
import torch

from .model import MLPConfig, ResidualMLP


class NumeraiMLPPredictor:
    """CPU-only callable with the official two-argument Model Upload interface."""

    def __init__(self, artifact: dict, batch_size: int = 4096) -> None:
        self.feature_names = tuple(artifact["feature_names"])
        self.data_version = artifact["data_version"]
        self.batch_size = int(batch_size)
        self.model = ResidualMLP(MLPConfig(**artifact["model_config"]))
        self.model.load_state_dict(artifact["model"])
        self.model.eval()
        self.model.to("cpu")

    @classmethod
    def from_file(cls, path: Path, batch_size: int = 4096) -> NumeraiMLPPredictor:
        artifact = torch.load(path, map_location="cpu", weights_only=False)
        return cls(artifact, batch_size=batch_size)

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
        predictions = []
        with torch.inference_mode():
            for start in range(0, len(values), self.batch_size):
                batch = torch.from_numpy(values[start:start + self.batch_size])
                predictions.append(self.model(batch).numpy())
        raw = np.concatenate(predictions) if predictions else np.empty(0, dtype=np.float32)
        if not np.isfinite(raw).all():
            raise ValueError("model emitted non-finite predictions")
        # Exact scoring is rank based; interior percentile ranks preserve it and
        # satisfy the Tournament submission range without clipping ties.
        ranked = pd.Series(raw).rank(method="average").to_numpy(dtype=np.float64)
        ranked = (ranked - 0.5) / max(1, len(ranked))
        return pd.DataFrame({"prediction": ranked}, index=live_features.index)


def export_callable(model_artifact: Path, output: Path, batch_size: int = 4096) -> Path:
    predictor = NumeraiMLPPredictor.from_file(model_artifact, batch_size=batch_size)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as handle:
        cloudpickle.dump(predictor, handle)
    return output
