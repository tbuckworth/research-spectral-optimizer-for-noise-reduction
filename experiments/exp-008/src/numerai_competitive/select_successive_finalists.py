"""Select candidates that receive full fold/seed confirmation after scouting."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .data import atomic_json, sha256
from .select_configs import select_configs


def _rank_high(scores: pd.DataFrame, ids: list[int], top: int) -> list[int]:
    required = {"arm", "config_id", "corr_mean", "split", "seed", "updates"}
    if not required <= set(scores) or not ids or len(ids) != len(set(ids)) or top < 1:
        raise ValueError("invalid high-rank score table, IDs or top count")
    selected = scores[
        scores["arm"].eq("spectral") & scores["config_id"].isin(ids)
    ].copy()
    if set(selected["config_id"].astype(int)) != set(ids):
        raise ValueError("high-rank scout scores omit a requested rank")
    if selected["updates"].nunique() != 1 or int(selected["updates"].iloc[0]) != 20000:
        raise ValueError("high-rank scouting must use exactly 20k updates")
    coverage = selected.groupby("config_id").apply(
        lambda frame: frozenset(zip(frame["split"], frame["seed"])),
        include_groups=False,
    )
    if coverage.nunique() != 1 or any(seed != 0 for cells in coverage for _, seed in cells):
        raise ValueError("high-rank scouts require equal fold coverage at seed zero")
    grouped = (selected.groupby("config_id", as_index=False)
               .agg(corr_mean=("corr_mean", "mean"), corr_worst=("corr_mean", "min")))
    ranked = grouped.sort_values(
        ["corr_mean", "corr_worst", "config_id"], ascending=[False, False, True],
    )
    if len(ranked) < top:
        raise ValueError("fewer high-rank scouts than requested finalists")
    return ranked.head(top)["config_id"].astype(int).tolist()


def select_finalists(
    ordinary_scout: pd.DataFrame, high_rank_scout: pd.DataFrame, *,
    high_rank_ids: list[int], high_rank_source_config_id: int, high_rank_top: int,
) -> dict:
    if (ordinary_scout["updates"].nunique() != 1
            or int(ordinary_scout["updates"].iloc[0]) != 100000
            or ordinary_scout["split"].nunique() != 1
            or ordinary_scout["seed"].nunique() != 1
            or int(ordinary_scout["seed"].iloc[0]) != 0):
        raise ValueError("ordinary long scout must be one fold at 100k and seed zero")
    winners = select_configs(ordinary_scout, 1)
    paired = set(winners["adamw"]) | set(winners["spectral"])
    paired.add(high_rank_source_config_id)
    return {
        "ordinary_winners": {arm: winners[arm] for arm in ("adamw", "spectral")},
        "ordinary_confirmation_paired_union": sorted(paired),
        "high_rank_spectral": _rank_high(high_rank_scout, high_rank_ids, high_rank_top),
        "high_rank_source_config_id": high_rank_source_config_id,
    }


def build_selection(
    ordinary_score: Path, high_rank_scores: list[Path], *, high_rank_ids: list[int],
    high_rank_source_config_id: int, high_rank_top: int, output: Path,
) -> dict:
    report = {
        "status": "successive_halving_finalists_frozen",
        "criterion": (
            "ordinary arm winners by scout CORR; high ranks by mean 20k fold CORR, "
            "worst-fold then lower config ID; runtime excluded"
        ),
        "ordinary_score": str(ordinary_score),
        "high_rank_scores": [str(path) for path in high_rank_scores],
        "score_sha256": {
            str(path): sha256(path) for path in [ordinary_score, *high_rank_scores]
        },
        **select_finalists(
            pd.read_csv(ordinary_score),
            pd.concat([pd.read_csv(path) for path in high_rank_scores], ignore_index=True),
            high_rank_ids=high_rank_ids,
            high_rank_source_config_id=high_rank_source_config_id,
            high_rank_top=high_rank_top,
        ),
    }
    atomic_json(output, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ordinary-score", type=Path, required=True)
    parser.add_argument("--high-rank-score", type=Path, action="append", required=True)
    parser.add_argument("--high-rank-id", type=int, action="append", required=True)
    parser.add_argument("--high-rank-source-config-id", type=int, required=True)
    parser.add_argument("--high-rank-top", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build_selection(
        args.ordinary_score, args.high_rank_score, high_rank_ids=args.high_rank_id,
        high_rank_source_config_id=args.high_rank_source_config_id,
        high_rank_top=args.high_rank_top, output=args.output,
    ), sort_keys=True))


if __name__ == "__main__":
    main()
