import cloudpickle
import numpy as np
import pandas as pd
import pytest

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
