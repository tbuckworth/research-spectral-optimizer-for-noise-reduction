"""Create a dated, reproducible public Numerai live-leaderboard snapshot."""
from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import numpy as np
from numerapi import NumerAPI

from .data import atomic_json, sha256

METRICS = {
    "corr20V2Rep": "CORR20v2",
    "mmcRep": "MMC",
    "bmcRep": "BMC",
    "corj60Rep": "CORJ60",
}


def _serializable(value):
    if isinstance(value, Decimal):
        return str(value)
    raise TypeError(f"unsupported leaderboard value {type(value).__name__}")


def summarize(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("leaderboard response is empty")
    ranks = [int(row["rank"]) for row in rows]
    identities = [row.get("username") for row in rows]
    if any(identity is None for identity in identities) or len(set(identities)) != len(rows):
        raise ValueError("leaderboard response contains missing or duplicate model identities")
    minimum_rank = min(ranks)
    top_rows = [row for row in rows if int(row["rank"]) == minimum_rank]
    metrics = {}
    for field, label in METRICS.items():
        values = np.asarray([float(row[field]) for row in rows
                             if row.get(field) is not None and np.isfinite(float(row[field]))])
        if not len(values):
            raise ValueError(f"leaderboard metric {field} has no finite values")
        top_values = [float(row[field]) for row in top_rows
                      if row.get(field) is not None and np.isfinite(float(row[field]))]
        metrics[field] = {
            "label": label, "median": float(np.median(values)),
            "p90": float(np.quantile(values, 0.9)), "maximum": float(values.max()),
            "finite_rows": len(values), "missing_rows": len(rows) - len(values),
            "top_rank_values": top_values,
        }
    return {
        "rows": len(rows), "rank_min": min(ranks), "rank_max": max(ranks),
        "top_rank_models": [row["username"] for row in top_rows], "metrics": metrics,
    }


def _markdown(report: dict) -> str:
    lines = [
        "# Public Numerai live leaderboard snapshot", "",
        (f"Retrieved `{report['retrieved_at']}` during round `{report['round']}` from the official "
         f"NumerAPI model-leaderboard endpoint. Rows: {report['summary']['rows']}."), "",
        "| Live one-year reputation | Finite rows | Median | 90th percentile | Maximum | Top-rank row(s) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for metric in report["summary"]["metrics"].values():
        top_rank = metric["top_rank_values"]
        top_rank_text = ("n/a" if not top_rank else f"{top_rank[0]:.5f}"
                         if len(top_rank) == 1 else
                         f"{min(top_rank):.5f}–{max(top_rank):.5f}")
        lines.append(
            f"| {metric['label']} | {metric['finite_rows']} | {metric['median']:.5f} | "
            f"{metric['p90']:.5f} | "
            f"{metric['maximum']:.5f} | {top_rank_text} |"
        )
    lines += [
        "", f"Top-rank model row(s): `{', '.join(report['summary']['top_rank_models'])}`.", "",
        ("These are forward, one-year live reputations. Historical validation CORR is not on this "
         "scale and cannot be converted into a leaderboard rank. A candidate becomes directly "
         "comparable only after repeated unstaked live submissions resolve."), "",
        f"Raw response: `leaderboard-raw.json` (SHA-256 `{report['raw_sha256']}`).", "",
    ]
    return "\n".join(lines)


def create_snapshot(output: Path, *, limit: int = 1000, api=None,
                    retrieved_at: str | None = None) -> dict:
    api = NumerAPI() if api is None else api
    rows = api.get_leaderboard(limit=limit, offset=0)
    output.mkdir(parents=True, exist_ok=True)
    raw_path = output / "leaderboard-raw.json"
    raw_path.write_text(json.dumps(rows, indent=2, sort_keys=True, default=_serializable) + "\n")
    report = {
        "status": "complete", "retrieved_at": retrieved_at or datetime.now(UTC).isoformat(),
        "round": api.get_current_round(), "limit": limit, "raw_sha256": sha256(raw_path),
        "summary": summarize(rows),
    }
    atomic_json(output / "leaderboard-summary.json", report)
    (output / "leaderboard-summary.md").write_text(_markdown(report))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=1000)
    args = parser.parse_args()
    print(json.dumps(create_snapshot(args.output, limit=args.limit), sort_keys=True))


if __name__ == "__main__":
    main()
