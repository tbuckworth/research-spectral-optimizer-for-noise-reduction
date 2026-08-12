"""Promote candidates without assuming configuration rankings persist across budgets."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .data import atomic_json, sha256
from .select_configs import select_configs


def select_multifidelity(score_groups: list[pd.DataFrame], top: int) -> dict:
    if not score_groups or top < 1:
        raise ValueError("at least one score group and a positive top count are required")
    by_budget: dict[int, dict[str, list[int]]] = {}
    for scores in score_groups:
        if "updates" not in scores or scores["updates"].nunique() != 1:
            raise ValueError("each score group must contain exactly one fidelity budget")
        budget = int(scores["updates"].iloc[0])
        if budget in by_budget:
            raise ValueError(f"duplicate fidelity budget: {budget}")
        by_budget[budget] = select_configs(scores, top)
    selected = {
        arm: sorted({config_id for value in by_budget.values() for config_id in value[arm]})
        for arm in ("adamw", "spectral")
    }
    selected["paired_union"] = sorted(set(selected["adamw"]) | set(selected["spectral"]))
    return {
        "selected": selected,
        "fidelity_selections": {str(key): by_budget[key] for key in sorted(by_budget)},
    }


def build_selection(score_groups: list[list[Path]], top: int, output: Path) -> dict:
    frames = [pd.concat([pd.read_csv(path) for path in group], ignore_index=True)
              for group in score_groups]
    value = select_multifidelity(frames, top)
    paths = [path for group in score_groups for path in group]
    report = {
        "status": "multifidelity_promotion",
        "criterion": (
            "top per arm independently within each update budget by mean exact CORR; "
            "worst-fold then config-id ties; union across budgets"
        ),
        "top_per_arm_per_fidelity": top,
        "score_groups": [[str(path) for path in group] for group in score_groups],
        "score_sha256": {str(path): sha256(path) for path in paths},
        **value,
    }
    atomic_json(output, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--score-group", type=Path, nargs="+", action="append", required=True)
    parser.add_argument("--top", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build_selection(args.score_group, args.top, args.output), sort_keys=True))


if __name__ == "__main__":
    main()
