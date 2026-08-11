"""Render the final audited comparison without conflating validation and live metrics."""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import matplotlib.pyplot as plt

from . import PRIMARY_BENCHMARK
from .data import atomic_json, sha256


def _load_complete(path: Path, label: str) -> dict:
    report = json.loads(path.read_text())
    if report.get("status") != "complete":
        raise ValueError(f"{label} report is not complete")
    return report


def _score_rows(report: dict) -> list[tuple[str, float, float]]:
    rows = []
    for name in ("adamw", "spectral", "candidate", "ender60"):
        if name in report:
            summary = report[name]["corr"] if "corr" in report[name] else report[name]
            rows.append((name, float(summary["mean"]), float(summary["sharpe"])))
    return rows


def _plot(outer: dict, validation: dict, output: Path) -> None:
    panels = [("Nested outer (development)", _score_rows(outer)),
              ("Sealed official validation", _score_rows(validation))]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    colors = {"adamw": "#377eb8", "spectral": "#e41a1c", "candidate": "#984ea3",
              "ender60": "#4daf4a"}
    for axis, (title, rows) in zip(axes[0], panels):
        names, means, _ = zip(*rows)
        axis.bar(names, means, color=[colors[name] for name in names])
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_title(title)
        axis.set_ylabel("Mean exact era-wise CORR")
        axis.tick_params(axis="x", rotation=25)

    def intervals(axis, values: list[tuple[str, dict]], title: str) -> None:
        labels = [label for label, _ in values]
        estimates = [value["estimate"] for _, value in values]
        lower = [value["estimate"] - value["ci_low"] for _, value in values]
        upper = [value["ci_high"] - value["estimate"] for _, value in values]
        axis.errorbar(labels, estimates, yerr=[lower, upper], fmt="o", capsize=4,
                      color="#333333", ecolor="#777777")
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set(title=title, ylabel="Paired mean CORR difference")
        axis.tick_params(axis="x", rotation=20)

    intervals(axes[1, 0], [
        ("nested outer", outer["spectral_minus_adamw"]),
        ("sealed validation", validation["spectral_minus_adamw"]),
    ], "Spectral − AdamW (95% block-bootstrap CI)")
    intervals(axes[1, 1], [
        ("AdamW", validation["adamw_minus_ender60"]),
        ("spectral", validation["spectral_minus_ender60"]),
        ("candidate", validation["candidate_minus_ender60"]),
    ], "Sealed validation model − Ender60")
    fig.suptitle("Comparable historical evaluation (same target and scorer)")
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _comparison_text(comparison: dict) -> str:
    return (f"{comparison['estimate']:+.6f} (95% moving-block bootstrap CI "
            f"[{comparison['ci_low']:+.6f}, {comparison['ci_high']:+.6f}], "
            f"P(positive)={comparison['probability_positive']:.3f})")


def _selected_config(search: dict, freeze: dict, arm: str) -> dict:
    selected = freeze.get("selected", {}).get(arm, {})
    config_id = selected.get("config_id")
    matches = [config for config in search.get("configs", [])
               if config.get("arm") == arm and config.get("config_id") == config_id]
    if len(matches) != 1:
        raise ValueError(f"{arm}: frozen config is absent or duplicated in search manifest")
    if (selected.get("updates") not in {5_000, 20_000, 100_000}
            or selected.get("seeds") != [0, 1, 2]):
        raise ValueError(f"{arm}: final training horizon or seeds differ from protocol")
    return {**matches[0], "training_updates": selected["updates"]}


def _config_rows(configs: dict[str, dict]) -> str:
    shared = ("config_id", "training_updates", "feature_set", "width", "depth",
              "activation", "residual",
              "normalization", "dropout", "batch_mode", "batch_size", "learning_rate",
              "weight_decay", "schedule", "warmup_fraction", "clip_grad_norm")
    spectral = ("rank", "decay", "filter_mode", "filter_strength", "filter_warmup",
                "filter_update_every")
    keys = shared + spectral
    rows = []
    for key in keys:
        if any(key in config for config in configs.values()):
            values = "".join(f"<td>{html.escape(str(config.get(key, '—')))}</td>"
                             for config in configs.values())
            rows.append(f"<tr><td><code>{html.escape(key)}</code></td>{values}</tr>")
    return "".join(rows)


def build_report(outer_path: Path, validation_path: Path, leaderboard_path: Path,
                 freeze_path: Path, search_path: Path, admission_path: Path,
                 output: Path) -> dict:
    outer = _load_complete(outer_path, "nested outer")
    validation = _load_complete(validation_path, "official validation")
    leaderboard = _load_complete(leaderboard_path, "leaderboard snapshot")
    freeze = json.loads(freeze_path.read_text())
    search = json.loads(search_path.read_text())
    admission = _load_complete(admission_path, "base-search memory admission")
    high_rank_ids = search.get("high_rank_config_ids", [])
    base_per_arm = search.get("configurations_per_arm")
    admitted_ids = set(admission.get("admitted_config_ids", []))
    excluded_ids = set(admission.get("excluded_config_ids", []))
    if (validation.get("target") != "target"
            or validation.get("benchmark") != PRIMARY_BENCHMARK):
        raise ValueError("official validation uses an unexpected primary target or benchmark")
    if validation.get("target_alias_audit") != {
        "target_equals_target_ender_60": True,
        "live_corr20v2_target": "target_cyrus_20",
        "live_target_released_in_v5_3": False,
    }:
        raise ValueError("official validation report lacks the target comparability audit")
    if (freeze.get("status") != "frozen" or freeze.get("primary_target") != validation["target"]
            or freeze.get("primary_benchmark") != validation["benchmark"]
            or freeze.get("search_sha256") != sha256(search_path)
            or search.get("primary_target") != validation["target"]
            or search.get("status") != "development_only_augmented_search"
            or base_per_arm != 40 or not high_rank_ids
            or admission.get("search_sha256") != search.get("base_search_sha256")
            or admission.get("pending_probe_config_ids") != []
            or admitted_ids & excluded_ids
            or admitted_ids | excluded_ids != set(range(base_per_arm))
            or len(search.get("configs", [])) != 2 * (base_per_arm + len(high_rank_ids))):
        raise ValueError("freeze, search space and validation target are inconsistent")
    selected_configs = {arm: _selected_config(search, freeze, arm)
                        for arm in ("adamw", "spectral")}
    for arm, config in selected_configs.items():
        if config["config_id"] not in high_rank_ids and config["config_id"] not in admitted_ids:
            raise ValueError(f"{arm}: frozen base configuration failed memory admission")
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
    candidate_delta = _comparison_text(validation["candidate_minus_ender60"])
    score_rows = "".join(
        f"<tr><td>{html.escape(name)}</td><td>{mean:.6f}</td><td>{sharpe:.3f}</td></tr>"
        for name, mean, sharpe in _score_rows(validation)
    )
    config_rows = _config_rows(selected_configs)
    body = f"""<!doctype html><html><body style="font-family:Arial,sans-serif;max-width:920px;
margin:auto;line-height:1.5;color:#202124"><h1>Numerai optimizer comparison</h1>
<p><strong>Decision-relevant result.</strong> Nested outer spectral minus AdamW: {outer_delta}.<br>
Sealed validation spectral minus AdamW: {validation_delta}.<br>
Sealed validation candidate minus official Ender60 benchmark: {candidate_delta}.</p>
<img src="historical-comparison.png" alt="Historical comparison" style="max-width:100%">
<h2>Sealed historical scores</h2><table style="border-collapse:collapse;width:100%">
<tr><th>Model</th><th>Mean exact era-wise CORR</th><th>CORR Sharpe</th></tr>{score_rows}</table>
<h2>AdamW optimization and paired spectral test</h2>
<p>The frozen base search contained <strong>40 paired configuration IDs per optimizer</strong>;
<strong>{len(admitted_ids)} pairs</strong> passed the outcome-independent memory-admission rule
and entered metric-based selection. Excluded IDs: <code>{html.escape(str(sorted(excluded_ids)))}</code>.
Before any outer or official-validation reveal, an audited amendment added
<strong>{len(high_rank_ids)} GPU-feasible high-rank spectral variants</strong> of one
development-selected architecture. Each ID used the same architecture, batches, examples,
base optimizer hyperparameters and candidate update budget in both arms; the spectral arm added
only its pre-registered filter parameters. Identical AdamW controls were not redundantly retrained
for every rank during F2, but every final configuration/budget nominee was cross-evaluated in both
arms. Selection treated 5,000, 20,000 and 100,000 updates as development hyperparameters on
chronological folds, followed by three-seed refits at each arm's selected budget. Official
validation remained sealed until these choices and model hashes were frozen.</p>
<p>Protocol ID: <code>{html.escape(search['protocol'])}</code>.</p>
<table style="border-collapse:collapse;width:100%"><tr><th>Hyperparameter</th>
<th>Selected AdamW</th><th>Selected spectral</th></tr>{config_rows}</table>
<h2>What is directly comparable</h2><p>The AdamW, spectral, candidate and Ender60 values above
use the same resolved historical rows, released main <code>target</code> and exact Numerai-CORR
transformation. Hyperparameters were selected without using official-validation model scores;
model evaluation opened validation only after the immutable freeze.</p>
<h2>Public live leaderboard context</h2><table style="border-collapse:collapse;width:100%">
<tr><th>One-year reputation</th><th>Median</th><th>90th percentile</th><th>Maximum</th></tr>
{live_rows}</table><p>Snapshot round {leaderboard['round']}, {leaderboard['summary']['rows']} rows,
retrieved {html.escape(leaderboard['retrieved_at'])}.</p>
<p style="padding:12px;background:#fff4ce"><strong>Comparability boundary:</strong> these live
one-year reputations use forward rounds scored against current live <code>target_cyrus_20</code>.
The pinned historical main <code>target</code> is byte-identical to
<code>target_ender_60</code>; it is not the released auxiliary
<code>target_cyrusd_20</code>, and neither released column is established as the exact live payout
endpoint. Historical scores therefore cannot be converted into a live rank. Direct leaderboard
comparability requires repeated unstaked live submissions to resolve.</p>
</body></html>"""
    (output / "report.html").write_text(body)
    markdown = f"""# Numerai optimizer comparison

## Audited result

- Nested outer spectral minus AdamW: {outer_delta}.
- Sealed validation spectral minus AdamW: {validation_delta}.
- Sealed validation candidate minus Ender60: {candidate_delta}.

AdamW was selected from {len(admitted_ids)} memory-admitted pairs out of 40 frozen base draws
using the chronological
5,000 → 20,000 → 100,000-update multi-fidelity protocol. Final AdamW config:
`{json.dumps(selected_configs['adamw'], sort_keys=True)}`. Final spectral config:
`{json.dumps(selected_configs['spectral'], sort_keys=True)}`.

The historical comparisons use the same rows, released main target and exact Numerai-CORR
transformation. The pinned main target equals `target_ender_60`, while current live CORR20v2
uses unreleased `target_cyrus_20`. The public round {leaderboard['round']} reputation snapshot is
therefore context only: historical CORR cannot be converted into a live rank. Repeated unstaked
live submissions are required for direct comparability.
"""
    (output / "report.md").write_text(markdown)
    manifest = {
        "status": "complete", "comparability": "historical-direct_live-context-only",
        "inputs": {str(path): sha256(path) for path in
                   (outer_path, validation_path, leaderboard_path, freeze_path, search_path,
                    admission_path)},
        "artifacts": {name: sha256(output / name) for name in
                      ("report.html", "report.md", "historical-comparison.png")},
        "nested_outer_spectral_minus_adamw": outer["spectral_minus_adamw"],
        "validation_spectral_minus_adamw": validation["spectral_minus_adamw"],
        "validation_candidate_minus_ender60": validation["candidate_minus_ender60"],
        "selected_configs": selected_configs,
    }
    atomic_json(output / "report-manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outer", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--leaderboard", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--search", type=Path, required=True)
    parser.add_argument("--admission", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build_report(args.outer, args.validation, args.leaderboard,
                                  args.freeze, args.search, args.admission,
                                  args.output), sort_keys=True))


if __name__ == "__main__":
    main()
