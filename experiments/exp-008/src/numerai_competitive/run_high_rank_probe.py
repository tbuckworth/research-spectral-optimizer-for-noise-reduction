"""Run one high-rank training probe and record CUDA-memory rejection atomically."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .data import TrainShard, atomic_json
from .materialize import materialize_config
from .model import MLPConfig
from .run_task import resolve_split
from .train import TrainConfig, run_training


def run_probe(search_path: Path, shards: Path, output: Path, config_id: int,
              split_name: str, updates: int, seed: int) -> dict:
    payload = json.loads(search_path.read_text())
    matches = [draw for draw in payload.get("configs", [])
               if draw.get("arm") == "spectral" and draw.get("config_id") == config_id]
    if len(matches) != 1:
        raise ValueError("probe requires one spectral search draw")
    draw = matches[0]
    if updates != draw.get("required_probe_updates"):
        raise ValueError("probe update budget differs from audited extension manifest")
    shard_root = shards / draw["feature_set"]
    shard = TrainShard.open(shard_root)
    value = materialize_config(draw, input_dim=shard.X.shape[1], updates=updates, seed=seed)
    value["model"] = MLPConfig(**value["model"])
    try:
        return run_training(
            shard_root, resolve_split(split_name), TrainConfig(**value), output,
        )
    except torch.cuda.OutOfMemoryError as exc:
        peak = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
        report = {
            "status": "resource_probe_rejected", "reason": "cuda_out_of_memory",
            "updates": updates, "split": {"name": split_name},
            "peak_cuda_memory_bytes": int(peak),
            "config": {"search_config_id": config_id, "arm": "spectral",
                       "filter": {"rank": draw["rank"]}},
            "error_type": type(exc).__name__,
        }
        atomic_json(output / "result.json", report)
        return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--search", type=Path, required=True)
    parser.add_argument("--shards", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config-id", type=int, required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--updates", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    print(json.dumps(run_probe(
        args.search, args.shards, args.output, args.config_id,
        args.split, args.updates, args.seed,
    ), sort_keys=True))


if __name__ == "__main__":
    main()
