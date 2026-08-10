import cloudpickle
import numpy as np
import pandas as pd
import pytest
import torch

from numerai_competitive.live import NumeraiMLPPredictor
from numerai_competitive.model import MLPConfig, ResidualMLP


def _artifact():
    config = MLPConfig(input_dim=3, width=5, depth=2)
    return {
        "model_config": config.__dict__, "model": ResidualMLP(config).state_dict(),
        "feature_names": ["a", "b", "c"], "data_version": "v5.3",
    }


def test_live_callable_is_deterministic_indexed_and_in_range():
    predictor = NumeraiMLPPredictor(_artifact(), batch_size=2)
    index = pd.Index(["id3", "id1", "id2"])
    integer = pd.DataFrame([[0, 2, 4], [1, 2, 3], [4, 0, 1]], index=index,
                           columns=["a", "b", "c"])
    normalized = integer / 4
    first = predictor(integer, pd.DataFrame(index=index))
    second = cloudpickle.loads(cloudpickle.dumps(predictor))(normalized, None)
    pd.testing.assert_frame_equal(first, second)
    assert first.index.equals(index)
    assert first["prediction"].between(0, 1, inclusive="neither").all()
    assert np.isfinite(first["prediction"]).all()


def test_live_callable_rejects_schema_and_range_errors():
    predictor = NumeraiMLPPredictor(_artifact())
    with pytest.raises(ValueError, match="missing"):
        predictor(pd.DataFrame({"a": [0.5], "b": [0.5]}))
    with pytest.raises(ValueError, match="0..4"):
        predictor(pd.DataFrame({"a": [5], "b": [0], "c": [0]}))


def test_live_ensemble_ranks_each_seed_before_averaging():
    first = _artifact()
    second = _artifact()
    with torch.no_grad():
        for parameter in second["model"].values():
            parameter.mul_(-1)
    frame = pd.DataFrame(
        [[0, 2, 4], [1, 2, 3], [4, 0, 1], [3, 1, 2]], columns=["a", "b", "c"]
    )
    ensemble = NumeraiMLPPredictor([first, second], batch_size=2)(frame)["prediction"]
    individual = [NumeraiMLPPredictor(artifact, batch_size=2)(frame)["prediction"]
                  for artifact in (first, second)]
    pd.testing.assert_series_equal(ensemble, (individual[0] + individual[1]) / 2)


def test_live_ensemble_rejects_mismatched_artifacts():
    first, second = _artifact(), _artifact()
    second["feature_names"] = ["a", "b", "different"]
    with pytest.raises(ValueError, match="feature schemas differ"):
        NumeraiMLPPredictor([first, second])


def test_live_frozen_benchmark_blend_requires_exact_benchmark_alignment():
    predictor = NumeraiMLPPredictor(_artifact(), model_weight=0.5)
    index = pd.Index(["id3", "id1", "id2", "id4"])
    frame = pd.DataFrame(
        [[0, 2, 4], [1, 2, 3], [4, 0, 1], [2, 4, 0]], index=index,
        columns=["a", "b", "c"],
    )
    benchmark = pd.DataFrame(
        {"v53_lgbm_ender20": [0.1, 0.8, 0.3, 0.6]}, index=index
    )
    blended = predictor(frame, benchmark)
    assert blended.index.equals(index)
    assert blended["prediction"].between(0, 1, inclusive="neither").all()
    with pytest.raises(ValueError, match="requires benchmark"):
        predictor(frame, None)
    with pytest.raises(ValueError, match="IDs/order"):
        predictor(frame, benchmark.iloc[::-1])
