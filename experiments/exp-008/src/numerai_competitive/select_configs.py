"""Deterministically promote configurations using development-fold CORR only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def select_configs(scores: pd.DataFrame, top: int) -> dict[str, list[int]]:
    required = {"arm", "config_id", "corr_mean", "split", "seed"}
    if not required <= set(scores.columns):
        raise ValueError(f"missing columns: {sorted(required - set(scores.columns))}")
    if scores[["arm", "config_id", "corr_mean", "split", "seed"]].isna().any().any():
        raise ValueError("selection columns contain missing values")
    identity = ["arm", "config_id", "split", "seed"]
    if scores.duplicated(identity).any():
        raise ValueError("duplicate arm/config/split/seed score")
    coverage = scores.groupby(["arm", "config_id"]).apply(
        lambda frame: frozenset(zip(frame["split"], frame["seed"])),
        include_groups=False,
    )
    if coverage.nunique() != 1:
        raise ValueError("configurations have unequal fold/seed coverage")
    grouped = (scores.groupby(["arm", "config_id"], as_index=False)
               .agg(corr_mean=("corr_mean", "mean"),
                    corr_worst=("corr_mean", "min"), folds=("corr_mean", "size")))
    selected = {}
    for arm in ("adamw", "spectral"):
        arm_scores = grouped[grouped["arm"] == arm]
        if len(arm_scores) < top:
            raise ValueError(f"{arm}: requested top {top} from {len(arm_scores)} configs")
        ranked = arm_scores.sort_values(
            ["corr_mean", "corr_worst", "config_id"], ascending=[False, False, True]
        )
        selected[arm] = ranked.head(top)["config_id"].astype(int).tolist()
    selected["paired_union"] = sorted(set(selected["adamw"]) | set(selected["spectral"]))
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", type=Path, nargs="+", required=True)
    parser.add_argument("--top", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    frame = pd.concat([pd.read_csv(path) for path in args.scores], ignore_index=True)
    payload = {
        "criterion": "descending mean exact standalone CORR; worst-fold then config-id ties",
        "top": args.top,
        "score_files": [str(path) for path in args.scores],
        "selected": select_configs(frame, args.top),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
