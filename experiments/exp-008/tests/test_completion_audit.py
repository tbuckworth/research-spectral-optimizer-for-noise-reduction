import json
from pathlib import Path

import pytest

from numerai_competitive.completion_audit import audit
from numerai_competitive.data import sha256


def _write(path: Path, value) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, (dict, list)):
        path.write_text(json.dumps(value))
    else:
        path.write_bytes(value)
    return path


def _complete_tree(tmp_path: Path) -> tuple[Path, Path]:
    results = tmp_path / "results"
    for number in range(1, 4):
        _write(results / f"audit-outer_{number}-u100000" / "outer-audit.json", {
            "status": "audit_complete", "split": {"name": f"outer_{number}"},
            "updates": 100_000, "seeds": [0, 1, 2], "cells": 6,
            "selected": {"adamw": number, "spectral": number + 3},
        })
    _write(results / "nested-outer" / "nested-outer-report.json", {
        "status": "complete", "outer_splits": ["outer_1", "outer_2", "outer_3"],
        "expected_seeds": [0, 1, 2], "spectral_minus_adamw": {"estimate": 0.1},
    })
    selected = {"arm": "spectral", "model_weight": 0.75, "benchmark_weight": 0.25,
                "benchmark": "v53_lgbm_ender20"}
    candidate = _write(results / "candidate-plan.json", {
        "status": "frozen_train_only_selection", "selected": selected,
    })
    _write(results / "selection-final-outer-winner-union.json", {
        "status": "outer_winners_audited",
        "selected": {"adamw": [1, 2, 3], "spectral": [4, 5, 6],
                     "paired_union": [1, 2, 3, 4, 5, 6]},
    })
    # Final winners must come from the audited union. Use two actual outer winners.
    winners = {"adamw": 1, "spectral": 4}
    _write(results / "selection-final-top1.json", {
        "selected": {"adamw": [1], "spectral": [4]},
    })
    _write(results / "audit-final-refits-u100000" / "refit-audit.json", {
        "status": "audit_complete", "cells": 6, "updates": 100_000,
        "seeds": [0, 1, 2], "selected": winners,
    })
    frozen = {}
    for arm, config in winners.items():
        paths = [_write(results / f"final-refit-u100000-s{seed}-{arm}-c{config}"
                        / "model.pt", f"{arm}-{seed}".encode()) for seed in range(3)]
        frozen[arm] = {"config_id": config, "seeds": [0, 1, 2], "updates": 100_000,
                       "model_sha256": [sha256(path) for path in paths]}
    freeze = _write(results / "freeze.json", {
        "status": "frozen", "code_commit": "a" * 40, "primary_target": "target_cyrusd_20",
        "candidate_plan_sha256": sha256(candidate), "candidate_transform": selected,
        "selected": frozen,
    })

    validation = results / "official-validation"
    validation_report = _write(validation / "official-validation-report.json", {
        "status": "complete", "target": "target_cyrusd_20",
        "freeze_manifest_sha256": sha256(freeze), "spectral_minus_adamw": {},
        "candidate_minus_ender20": {},
    })
    plot = _write(validation / "plot.png", b"plot")
    _write(validation / "evaluation-complete.json", {
        "status": "complete", "artifacts": {
            validation_report.name: sha256(validation_report), plot.name: sha256(plot),
        },
    })

    bundle = results / "live-bundle"
    fixture = bundle / "live-fixture"
    live = _write(fixture / "live.parquet", b"live")
    benchmark = _write(fixture / "live_benchmark_models.parquet", b"benchmark")
    _write(fixture / "download-complete.json", {
        "status": "complete", "freeze_manifest_sha256": sha256(freeze),
        "artifacts": {
            live.name: {"sha256": sha256(live), "bytes": live.stat().st_size},
            benchmark.name: {"sha256": sha256(benchmark), "bytes": benchmark.stat().st_size},
        },
    })
    predictor = _write(bundle / "predictor.pkl", b"predictor")
    predictions = _write(bundle / "live_predictions.csv", b"id,prediction\na,0.5\n")
    _write(bundle / "runtime-audit.json", {
        "status": "pass", "artifact_sha256": sha256(predictor),
        "prediction_sha256": sha256(predictions), "max_bytes": 4_000_000_000,
        "max_seconds": 600,
    })
    _write(bundle / "official-container" / "official-container-audit.json", {
        "status": "pass", "expected_sha256": sha256(predictions), "cpu_limit": 1,
        "memory_limit_bytes": 4_000_000_000, "max_seconds": 600,
        "runner_commit": "b" * 40,
    })
    leaderboard = _write(tmp_path / "leaderboard.json", {
        "status": "complete", "summary": {"rows": 1000},
    })
    report_dir = results / "final-report"
    report_html = _write(report_dir / "report.html", b"report")
    _write(report_dir / "report-manifest.json", {
        "status": "complete", "comparability": "historical-direct_live-context-only",
        "inputs": {"outer": sha256(results / "nested-outer" / "nested-outer-report.json"),
                   "validation": sha256(validation_report),
                   "leaderboard": sha256(leaderboard)},
        "artifacts": {report_html.name: sha256(report_html)},
    })
    return results, leaderboard


def test_completion_audit_cross_checks_full_chain(tmp_path):
    results, leaderboard = _complete_tree(tmp_path)
    output = tmp_path / "completion.json"
    report = audit(results, leaderboard, output)
    assert report["status"] == "audit_complete"
    assert report["final_selected"] == {"adamw": 1, "spectral": 4}
    assert len(report["evidence_sha256"]) == 15


def test_completion_audit_rejects_model_changed_after_freeze(tmp_path):
    results, leaderboard = _complete_tree(tmp_path)
    (results / "final-refit-u100000-s1-spectral-c4" / "model.pt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="frozen model hashes"):
        audit(results, leaderboard, tmp_path / "bad.json")
