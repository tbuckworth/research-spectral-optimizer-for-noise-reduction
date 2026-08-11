"""Rank configuration/update-budget pairs on development folds only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .data import sha256


def select_budgeted_configs(scores: pd.DataFrame, top: int) -> dict[str, list[dict[str, int]]]:
    """Select arm-specific ``(config_id, updates)`` candidates with equal coverage.

    This is deliberately separate from successive-halving selection: it is for a
    preregistered development-only sensitivity in which training budget itself is
    a hyperparameter. Every candidate must have exactly the same split/seed cells,
    and AdamW and spectral must evaluate the same configuration/budget pairs.
    """
    required = {"arm", "config_id", "corr_mean", "split", "seed", "updates"}
    if not required <= set(scores.columns):
        raise ValueError(f"missing columns: {sorted(required - set(scores.columns))}")
    if top < 1:
        raise ValueError("top must be positive")
    if scores[list(required)].isna().any().any():
        raise ValueError("selection columns contain missing values")
    if (scores["updates"] <= 0).any():
        raise ValueError("updates must be positive")

    identity = ["arm", "config_id", "updates", "split", "seed"]
    if scores.duplicated(identity).any():
        raise ValueError("duplicate arm/config/budget/split/seed score")

    candidates = {
        arm: set(zip(
            scores.loc[scores["arm"].eq(arm), "config_id"].astype(int),
            scores.loc[scores["arm"].eq(arm), "updates"].astype(int),
        ))
        for arm in ("adamw", "spectral")
    }
    if candidates["adamw"] != candidates["spectral"]:
        raise ValueError("AdamW and spectral config/budget coverage differs")

    coverage = scores.groupby(["arm", "config_id", "updates"]).apply(
        lambda frame: frozenset(zip(frame["split"], frame["seed"])),
        include_groups=False,
    )
    if coverage.empty or coverage.nunique() != 1:
        raise ValueError("candidates have unequal split/seed coverage")

    grouped = (scores.groupby(["arm", "config_id", "updates"], as_index=False)
               .agg(corr_mean=("corr_mean", "mean"),
                    corr_worst=("corr_mean", "min"), cells=("corr_mean", "size")))
    selected: dict[str, list[dict[str, int]]] = {}
    for arm in ("adamw", "spectral"):
        ranked = grouped[grouped["arm"].eq(arm)].sort_values(
            ["corr_mean", "corr_worst", "config_id", "updates"],
            ascending=[False, False, True, True],
        )
        if len(ranked) < top:
            raise ValueError(f"{arm}: requested top {top} from {len(ranked)} candidates")
        selected[arm] = [
            {"config_id": int(row.config_id), "updates": int(row.updates)}
            for row in ranked.head(top).itertuples()
        ]
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", type=Path, nargs="+", required=True)
    parser.add_argument("--top", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    frame = pd.concat([pd.read_csv(path) for path in args.scores], ignore_index=True)
    payload = {
        "status": "development_budget_sensitivity_selection",
        "criterion": (
            "descending mean exact standalone CORR; worst-cell, config-id, then update-budget ties"
        ),
        "top": args.top,
        "score_files": [str(path) for path in args.scores],
        "score_sha256": {str(path): sha256(path) for path in args.scores},
        "selected": select_budgeted_configs(frame, args.top),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
