import pytest

from numerai_competitive.materialize import materialize_config
from numerai_competitive.model import MLPConfig
from numerai_competitive.search_space import generate_search
from numerai_competitive.train import TrainConfig


@pytest.mark.parametrize("arm_index", [0, 1])
def test_search_draw_materializes_to_train_config(arm_index):
    draw = generate_search(n=1)[arm_index]
    value = materialize_config(draw, input_dim=780, updates=123, seed=7)
    value["model"] = MLPConfig(**value["model"])
    config = TrainConfig(**value)
    assert config.example_budget == 123 * draw["batch_size"]
    assert config.batch_mode in {"row", "era"}
    assert bool(config.filter) == (draw["arm"] == "spectral")
    assert config.search_config_id == draw["config_id"]
    assert config.feature_set == draw["feature_set"]


def test_artifact_cadence_scales_without_changing_budget():
    draw = generate_search(n=1)[0]
    value = materialize_config(draw, input_dim=780, updates=100_000)
    assert value["checkpoint_every"] == 50_000
    assert value["log_every"] == 500
    assert value["examples"] == 100_000 * draw["batch_size"]
