"""Add GPU-audited high-rank pairs to an ordinary F2 promotion selection."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import atomic_json, sha256


def augment_selection(selection_path: Path, search_path: Path, output: Path) -> dict:
    selection = json.loads(selection_path.read_text())
    search = json.loads(search_path.read_text())
    base = selection.get("selected", {}).get("paired_union", [])
    high_rank = search.get("high_rank_config_ids", [])
    if (search.get("status") != "development_only_augmented_search"
            or not base or not high_rank
            or len(base) != len(set(base)) or len(high_rank) != len(set(high_rank))):
        raise ValueError("base selection or augmented search is incomplete")
    available = {(config.get("arm"), config.get("config_id"))
                 for config in search.get("configs", [])}
    if any((arm, config_id) not in available
           for config_id in [*base, *high_rank] for arm in ("adamw", "spectral")):
        raise ValueError("promoted configuration is not paired in augmented search")
    report = dict(selection)
    report["status"] = "f1_selection_augmented_with_gpu_audited_high_ranks"
    report["base_selection_sha256"] = sha256(selection_path)
    report["augmented_search_sha256"] = sha256(search_path)
    report["high_rank_config_ids"] = sorted(high_rank)
    report["selected"] = dict(selection["selected"])
    # The high-rank IDs differ only in the spectral rank. Their AdamW members
    # are identical controls, so F2 trains the ordinary paired union plus only
    # the spectral member of each high-rank ID. Final winners are cross-evaluated
    # by both arms in the later canonical-fold stage.
    report["selected"]["paired_union"] = sorted(base)
    report["selected"]["high_rank_spectral"] = sorted(high_rank)
    atomic_json(output, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--search", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(augment_selection(args.selection, args.search, args.output), sort_keys=True))


if __name__ == "__main__":
    main()
