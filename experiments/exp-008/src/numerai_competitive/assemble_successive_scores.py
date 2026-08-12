"""Assemble only preregistered equal-coverage candidates from halving stages."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .data import atomic_json, sha256


def expected_candidates(plan: dict, finalists: dict) -> dict[int, dict[str, set[int]]]:
    values: dict[int, dict[str, set[int]]] = {}
    for budget in (5000, 20000):
        selection = plan["confirmation_selections"][str(budget)]["paired_union"]
        values[budget] = {arm: set(selection) for arm in ("adamw", "spectral")}
    high = set(finalists["high_rank_spectral"])
    values[20000]["spectral"] |= high
    ordinary = set(finalists["ordinary_confirmation_paired_union"])
    values[100000] = {"adamw": set(ordinary), "spectral": ordinary | high}
    return values


def assemble(
    scores: pd.DataFrame, plan: dict, finalists: dict, *,
    expected_splits: set[str], expected_seeds: set[int],
) -> pd.DataFrame:
    required = {"arm", "config_id", "corr_mean", "split", "seed", "updates"}
    if not required <= set(scores) or not expected_splits or not expected_seeds:
        raise ValueError("scores or expected coverage is incomplete")
    identities = ["arm", "config_id", "updates", "split", "seed"]
    if scores.duplicated(identities).any():
        raise ValueError("input summaries contain duplicate exact score cells")
    expected = expected_candidates(plan, finalists)
    keep = pd.Series(False, index=scores.index)
    expected_cells = set()
    for budget, arms in expected.items():
        for arm, ids in arms.items():
            if not ids:
                raise ValueError("a successive candidate set is empty")
            keep |= (scores["updates"].eq(budget) & scores["arm"].eq(arm)
                     & scores["config_id"].isin(ids))
            expected_cells |= {
                (arm, config_id, budget, split, seed)
                for config_id in ids for split in expected_splits for seed in expected_seeds
            }
    selected = scores.loc[keep].copy()
    actual_cells = {
        (str(row.arm), int(row.config_id), int(row.updates), str(row.split), int(row.seed))
        for row in selected.itertuples()
    }
    if actual_cells != expected_cells:
        missing = sorted(expected_cells - actual_cells)[:5]
        extra = sorted(actual_cells - expected_cells)[:5]
        raise ValueError(f"successive score coverage differs: missing={missing}, extra={extra}")
    return selected.sort_values(identities).reset_index(drop=True)


def build(
    score_paths: list[Path], plan_path: Path, finalists_path: Path, *,
    expected_splits: set[str], expected_seeds: set[int], output: Path,
) -> dict:
    frame = assemble(
        pd.concat([pd.read_csv(path) for path in score_paths], ignore_index=True),
        json.loads(plan_path.read_text()), json.loads(finalists_path.read_text()),
        expected_splits=expected_splits, expected_seeds=expected_seeds,
    )
    output.mkdir(parents=True, exist_ok=False)
    scores_out = output / "scores.csv"
    frame.to_csv(scores_out, index=False)
    report = {
        "status": "successive_scores_equal_coverage",
        "splits": sorted(expected_splits), "seeds": sorted(expected_seeds),
        "rows": len(frame), "plan_sha256": sha256(plan_path),
        "finalists_sha256": sha256(finalists_path),
        "source_score_sha256": {str(path): sha256(path) for path in score_paths},
        "scores_sha256": sha256(scores_out),
    }
    atomic_json(output / "assembly-audit.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--score", type=Path, action="append", required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--finalists", type=Path, required=True)
    parser.add_argument("--split", action="append", required=True)
    parser.add_argument("--seed", type=int, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(
        args.score, args.plan, args.finalists, expected_splits=set(args.split),
        expected_seeds=set(args.seed), output=args.output,
    ), sort_keys=True))


if __name__ == "__main__":
    main()
