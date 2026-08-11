"""Run one indexed search draw on one named development fold."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import TrainShard
from .materialize import materialize_config
from .model import MLPConfig
from .splits import resolve_split_60d
from .train import TrainConfig, run_training


def resolve_split(name: str):
    return resolve_split_60d(name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--search", type=Path, required=True)
    parser.add_argument("--shards", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm", choices=["adamw", "spectral"], required=True)
    parser.add_argument("--config-id", type=int, required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--updates", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    payload = json.loads(args.search.read_text())
    matches = [draw for draw in payload["configs"]
               if draw["arm"] == args.arm and draw["config_id"] == args.config_id]
    if len(matches) != 1:
        raise ValueError(f"expected one search draw, found {len(matches)}")
    draw = matches[0]
    shard_root = args.shards / draw["feature_set"]
    shard = TrainShard.open(shard_root)
    if shard.manifest.get("feature_set") != draw["feature_set"]:
        raise ValueError(
            f"shard feature set {shard.manifest.get('feature_set')!r} does not match "
            f"search draw {draw['feature_set']!r}"
        )
    value = materialize_config(draw, input_dim=shard.X.shape[1], updates=args.updates,
                               seed=args.seed)
    value["model"] = MLPConfig(**value["model"])
    result = run_training(shard_root, resolve_split(args.split), TrainConfig(**value),
                          args.output)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
