"""Refit one frozen search procedure on every train era and persist its model."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .data import TrainShard
from .materialize import materialize_config
from .model import MLPConfig
from .splits import EraSplit
from .train import TrainConfig, run_training


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--search", type=Path, required=True)
    parser.add_argument("--shards", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm", choices=["adamw", "spectral"], required=True)
    parser.add_argument("--config-id", type=int, required=True)
    parser.add_argument("--updates", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()

    search = json.loads(args.search.read_text())
    matches = [draw for draw in search["configs"]
               if draw["arm"] == args.arm and draw["config_id"] == args.config_id]
    if len(matches) != 1:
        raise ValueError(f"expected one frozen search draw, found {len(matches)}")
    draw = matches[0]
    shard_root = args.shards / draw["feature_set"]
    shard = TrainShard.open(shard_root)
    if shard.manifest.get("feature_set") != draw["feature_set"]:
        raise ValueError("refit shard feature set does not match frozen search draw")
    value = materialize_config(draw, input_dim=shard.X.shape[1], updates=args.updates,
                               seed=args.seed)
    value["model"] = MLPConfig(**value["model"])
    value["save_model"] = True
    eras = tuple(f"{int(era):04d}" for era in np.unique(shard.eras))
    split = EraSplit("all_train_refit", eras, (), ())
    result = run_training(shard_root, split, TrainConfig(**value), args.output, refit=True)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
