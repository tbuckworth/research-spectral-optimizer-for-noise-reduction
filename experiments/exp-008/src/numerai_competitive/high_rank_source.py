"""Select the best development configuration capable of a requested global rank."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .data import atomic_json, sha256
from .high_rank import memory_estimate


def select_source(search_path: Path, score_paths: list[Path], rank: int,
                  output: Path) -> dict:
    if rank < 1 or not score_paths:
        raise ValueError("positive rank and development score files are required")
    search = json.loads(search_path.read_text())
    configs = {config["config_id"]: config for config in search.get("configs", [])
               if config.get("arm") == "spectral"}
    scores = pd.concat([pd.read_csv(path) for path in score_paths], ignore_index=True)
    required = {"arm", "config_id", "corr_mean", "split", "seed", "updates"}
    if not required <= set(scores) or scores[list(required)].isna().any().any():
        raise ValueError("development scores have an incomplete selection schema")
    scores = scores[scores["arm"].eq("spectral")].copy()
    if scores.empty or scores.duplicated(["config_id", "split", "seed", "updates"]).any():
        raise ValueError("spectral scores are empty or duplicated")
    unknown = set(scores["config_id"].astype(int)) - set(configs)
    if unknown:
        raise ValueError(f"development scores contain unknown config IDs: {sorted(unknown)}")
    coverage = scores.groupby("config_id").apply(
        lambda frame: frozenset(zip(frame["split"], frame["seed"], frame["updates"])),
        include_groups=False,
    )
    if coverage.nunique() != 1:
        raise ValueError("candidate configurations have unequal fold/seed/update coverage")
    feasible = {
        config_id for config_id, config in configs.items()
        if config_id in set(scores["config_id"].astype(int))
        and memory_estimate(config, rank)["analytically_feasible_48gb"]
    }
    if not feasible:
        raise ValueError(f"no development-evaluated spectral config is feasible at rank {rank}")
    grouped = (scores[scores["config_id"].isin(feasible)]
               .groupby("config_id", as_index=False)
               .agg(corr_mean=("corr_mean", "mean"), corr_worst=("corr_mean", "min"),
                    cells=("corr_mean", "size"))
               .sort_values(["corr_mean", "corr_worst", "config_id"],
                            ascending=[False, False, True]))
    winner = int(grouped.iloc[0]["config_id"])
    report = {
        "status": "development_only_high_rank_source_selection",
        "criterion": "mean exact CORR; worst-cell CORR then lower config ID ties",
        "requested_rank": rank, "selected": {"spectral": [winner]},
        "feasible_config_ids": sorted(feasible),
        "winner_memory_screen": memory_estimate(configs[winner], rank),
        "winner_score": grouped.iloc[0].to_dict(),
        "search_sha256": sha256(search_path),
        "score_sha256": {str(path): sha256(path) for path in score_paths},
    }
    atomic_json(output, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--search", type=Path, required=True)
    parser.add_argument("--scores", type=Path, nargs="+", required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(select_source(
        args.search, args.scores, args.rank, args.output
    ), sort_keys=True))


if __name__ == "__main__":
    main()
