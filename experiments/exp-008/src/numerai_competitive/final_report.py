"""Render the final audited comparison without conflating validation and live metrics."""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import matplotlib.pyplot as plt

from .data import atomic_json, sha256


def _load_complete(path: Path, label: str) -> dict:
    report = json.loads(path.read_text())
    if report.get("status") != "complete":
        raise ValueError(f"{label} report is not complete")
    return report


def _score_rows(report: dict) -> list[tuple[str, float, float]]:
    rows = []
    for name in ("adamw", "spectral", "candidate", "ender20"):
        if name in report:
            summary = report[name]["corr"] if "corr" in report[name] else report[name]
            rows.append((name, float(summary["mean"]), float(summary["sharpe"])))
    return rows


def _plot(outer: dict, validation: dict, output: Path) -> None:
    panels = [("Nested outer (development)", _score_rows(outer)),
              ("Sealed official validation", _score_rows(validation))]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    colors = {"adamw": "#377eb8", "spectral": "#e41a1c", "candidate": "#984ea3",
              "ender20": "#4daf4a"}
    for axis, (title, rows) in zip(axes, panels):
        names, means, _ = zip(*rows)
        axis.bar(names, means, color=[colors[name] for name in names])
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_title(title)
        axis.set_ylabel("Mean exact era-wise CORR")
        axis.tick_params(axis="x", rotation=25)
    fig.suptitle("Comparable historical evaluation (same target and scorer)")
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _comparison_text(comparison: dict) -> str:
    return (f"{comparison['estimate']:+.6f} (95% moving-block bootstrap CI "
            f"[{comparison['ci_low']:+.6f}, {comparison['ci_high']:+.6f}], "
            f"P(positive)={comparison['probability_positive']:.3f})")


def build_report(outer_path: Path, validation_path: Path, leaderboard_path: Path,
                 output: Path) -> dict:
    outer = _load_complete(outer_path, "nested outer")
    validation = _load_complete(validation_path, "official validation")
    leaderboard = _load_complete(leaderboard_path, "leaderboard snapshot")
    if validation.get("target") != "target_cyrusd_20":
        raise ValueError("official validation report uses an unexpected primary target")
    output.mkdir(parents=True, exist_ok=True)
    plot_path = output / "historical-comparison.png"
    _plot(outer, validation, plot_path)
    metrics = leaderboard["summary"]["metrics"]
    live_rows = "".join(
        f"<tr><td>{html.escape(metric['label'])}</td><td>{metric['median']:.5f}</td>"
        f"<td>{metric['p90']:.5f}</td><td>{metric['maximum']:.5f}</td></tr>"
        for metric in metrics.values()
    )
    outer_delta = _comparison_text(outer["spectral_minus_adamw"])
    validation_delta = _comparison_text(validation["spectral_minus_adamw"])
    candidate_delta = _comparison_text(validation["candidate_minus_ender20"])
    body = f"""<!doctype html><html><body style="font-family:Arial,sans-serif;max-width:920px;
margin:auto;line-height:1.5;color:#202124"><h1>Numerai optimizer comparison</h1>
<p><strong>Decision-relevant result.</strong> Nested outer spectral minus AdamW: {outer_delta}.<br>
Sealed validation spectral minus AdamW: {validation_delta}.<br>
Sealed validation candidate minus official Ender20 benchmark: {candidate_delta}.</p>
<img src="historical-comparison.png" alt="Historical comparison" style="max-width:100%">
<h2>What is directly comparable</h2><p>The AdamW, spectral, candidate and Ender20 values above
use the same resolved historical rows, target and exact era-wise scorer. Hyperparameters were selected
without seeing official validation; validation was opened only after the immutable freeze.</p>
<h2>Public live leaderboard context</h2><table style="border-collapse:collapse;width:100%">
<tr><th>One-year reputation</th><th>Median</th><th>90th percentile</th><th>Maximum</th></tr>
{live_rows}</table><p>Snapshot round {leaderboard['round']}, {leaderboard['summary']['rows']} rows,
retrieved {html.escape(leaderboard['retrieved_at'])}.</p>
<p style="padding:12px;background:#fff4ce"><strong>Comparability boundary:</strong> these live
one-year reputations use forward rounds and the current payout target. Historical validation CORR on
<code>target_cyrusd_20</code> is not on the same scale. It cannot honestly be translated into a public
leaderboard rank. Direct leaderboard comparability requires repeated unstaked live submissions to resolve.</p>
</body></html>"""
    (output / "report.html").write_text(body)
    markdown = f"""# Numerai optimizer comparison

## Audited result

- Nested outer spectral minus AdamW: {outer_delta}.
- Sealed validation spectral minus AdamW: {validation_delta}.
- Sealed validation candidate minus Ender20: {candidate_delta}.

The historical comparisons use the same rows, target and exact scorer. The public round
{leaderboard['round']} live reputation snapshot is context only: it cannot be converted into a rank for
historical `target_cyrusd_20`. Repeated unstaked live submissions are required for direct comparability.
"""
    (output / "report.md").write_text(markdown)
    manifest = {
        "status": "complete", "comparability": "historical-direct_live-context-only",
        "inputs": {str(path): sha256(path) for path in
                   (outer_path, validation_path, leaderboard_path)},
        "artifacts": {name: sha256(output / name) for name in
                      ("report.html", "report.md", "historical-comparison.png")},
        "nested_outer_spectral_minus_adamw": outer["spectral_minus_adamw"],
        "validation_spectral_minus_adamw": validation["spectral_minus_adamw"],
        "validation_candidate_minus_ender20": validation["candidate_minus_ender20"],
    }
    atomic_json(output / "report-manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outer", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--leaderboard", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build_report(args.outer, args.validation, args.leaderboard, args.output),
                     sort_keys=True))


if __name__ == "__main__":
    main()
