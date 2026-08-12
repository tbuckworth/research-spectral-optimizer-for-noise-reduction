"""Freeze budget-specific confirmation sets and a small long-budget scout."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .data import atomic_json, sha256
from .select_configs import select_configs


def _one_budget(scores: pd.DataFrame) -> int:
    required = {"arm", "config_id", "corr_mean", "updates", "split"}
    if not required <= set(scores):
        raise ValueError(f"missing columns: {sorted(required - set(scores))}")
    if scores.empty or scores[list(required)].isna().any().any():
        raise ValueError("score group is empty or contains missing selection values")
    budgets = scores["updates"].astype(int).unique()
    if len(budgets) != 1 or budgets[0] not in {5000, 20000}:
        raise ValueError("each score group must contain exactly one 5k or 20k budget")
    return int(budgets[0])


def select_successive_halving(
    score_groups: list[pd.DataFrame], *, confirmation_top: int,
    long_scout_top: int, high_rank_source_config_id: int,
) -> dict:
    """Select using CORR only; runtime never changes admission or rank.

    The confirmation sets retain several candidates at their observed budget.
    The much more expensive 100k scout retains fewer candidates from each
    observed fidelity and always includes the separately frozen high-rank
    source, preserving its unchanged AdamW control.
    """
    if (len(score_groups) != 2 or confirmation_top < 1 or long_scout_top < 1
            or long_scout_top > confirmation_top or high_rank_source_config_id < 0):
        raise ValueError("invalid successive-halving dimensions or source config")
    groups: dict[int, pd.DataFrame] = {}
    for scores in score_groups:
        budget = _one_budget(scores)
        if budget in groups:
            raise ValueError(f"duplicate score budget {budget}")
        groups[budget] = scores
    if set(groups) != {5000, 20000}:
        raise ValueError("successive halving requires exactly 5k and 20k score groups")

    confirmations = {
        str(budget): select_configs(groups[budget], confirmation_top)
        for budget in (5000, 20000)
    }
    scouts = {
        str(budget): select_configs(groups[budget], long_scout_top)
        for budget in (5000, 20000)
    }
    long_ids = {
        config_id for selection in scouts.values()
        for arm in ("adamw", "spectral") for config_id in selection[arm]
    }
    long_ids.add(high_rank_source_config_id)
    return {
        "confirmation_selections": confirmations,
        "long_scout_nominations": scouts,
        "long_scout_paired_union": sorted(long_ids),
        "high_rank_source_config_id": high_rank_source_config_id,
    }


def build_plan(
    score_groups: list[list[Path]], *, confirmation_top: int, long_scout_top: int,
    high_rank_source_config_id: int, output: Path,
) -> dict:
    frames = [pd.concat([pd.read_csv(path) for path in group], ignore_index=True)
              for group in score_groups]
    value = select_successive_halving(
        frames, confirmation_top=confirmation_top, long_scout_top=long_scout_top,
        high_rank_source_config_id=high_rank_source_config_id,
    )
    paths = [path for group in score_groups for path in group]
    report = {
        "status": "successive_halving_plan_frozen",
        "criterion": (
            "within each observed budget: descending mean exact standalone CORR, "
            "worst-fold then lower config ID; paired unions; runtime excluded"
        ),
        "confirmation_top_per_arm_per_fidelity": confirmation_top,
        "long_scout_top_per_arm_per_fidelity": long_scout_top,
        "score_groups": [[str(path) for path in group] for group in score_groups],
        "score_sha256": {str(path): sha256(path) for path in paths},
        **value,
    }
    atomic_json(output, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--score-group", type=Path, nargs="+", action="append", required=True)
    parser.add_argument("--confirmation-top", type=int, default=2)
    parser.add_argument("--long-scout-top", type=int, default=1)
    parser.add_argument("--high-rank-source-config-id", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build_plan(
        args.score_group, confirmation_top=args.confirmation_top,
        long_scout_top=args.long_scout_top,
        high_rank_source_config_id=args.high_rank_source_config_id,
        output=args.output,
    ), sort_keys=True))


if __name__ == "__main__":
    main()
