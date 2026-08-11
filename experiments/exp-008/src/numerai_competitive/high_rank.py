"""Create a development-only global-rank extension from a selected spectral draw."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import atomic_json, sha256


def create_extension(search_path: Path, selection_path: Path, ranks: list[int],
                     output: Path) -> dict:
    search = json.loads(search_path.read_text())
    selection = json.loads(selection_path.read_text())
    selected = selection.get("selected", {}).get("spectral", [])
    if len(selected) != 1:
        raise ValueError("high-rank extension requires one selected spectral config")
    source_id = selected[0]
    matches = [draw for draw in search.get("configs", [])
               if draw.get("arm") == "spectral" and draw.get("config_id") == source_id]
    if len(matches) != 1:
        raise ValueError("selected spectral config is absent or duplicated")
    source = matches[0]
    if (not ranks or len(ranks) != len(set(ranks))
            or any(not isinstance(rank, int) or rank <= source["rank"] for rank in ranks)):
        raise ValueError("extension ranks must be unique integers above the selected rank")
    configs = []
    for rank in sorted(ranks):
        config = dict(source)
        config.update({"config_id": 10_000 + rank, "rank": rank,
                       "source_config_id": source_id})
        configs.append(config)
    report = {
        "status": "development_only_high_rank_extension",
        "selection_role": "eligible before freeze; official validation remains sealed",
        "primary_target": search.get("primary_target"),
        "primary_metric": search.get("primary_metric"),
        "source_search_sha256": sha256(search_path),
        "source_selection_sha256": sha256(selection_path),
        "source_config_id": source_id,
        "original_rank": source["rank"],
        "ranks": sorted(ranks),
        "configs": configs,
    }
    atomic_json(output, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--search", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--rank", type=int, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(create_extension(
        args.search, args.selection, args.rank, args.output
    ), sort_keys=True))


if __name__ == "__main__":
    main()
