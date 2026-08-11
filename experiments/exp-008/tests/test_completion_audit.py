import hashlib
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
    code_files = {}
    for name in ("pyproject.toml", "uv.lock", "fidelity-protocol.md",
                 "src/numerai_competitive/code_snapshot.py",
                 "src/numerai_competitive/freeze.py", "configs/search-v1.json"):
        path = _write(tmp_path / name, name.encode())
        code_files[name] = sha256(path)
    code_digest = hashlib.sha256(
        json.dumps(code_files, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    code_snapshot = _write(tmp_path / "code-snapshot.json", {
        "status": "complete", "code_commit": "a" * 40, "source_prefix": "x",
        "files": code_files, "file_count": len(code_files),
        "file_map_sha256": code_digest,
    })
    base_search = tmp_path / "configs" / "search-v1.json"
    _write(results / "base-search-memory-admission.json", {
        "status": "complete", "search_sha256": sha256(base_search),
        "admitted_config_ids": list(range(40)), "excluded_config_ids": [],
        "pending_probe_config_ids": [],
        "rows": [{"config_id": config_id, "state": "static_safe"}
                 for config_id in range(40)],
    })
    f1_by_outer, f2_by_outer, ordinary_by_outer = {}, {}, {}
    for number in range(1, 4):
        ordinary = {number, number + 3, number + 10, number + 20}
        ordinary_by_outer[number] = ordinary
        promoted = sorted(ordinary | (set(range(12)) - ordinary))[:12]
        if not ordinary <= set(promoted):
            promoted = sorted(ordinary) + [value for value in range(40)
                                           if value not in ordinary][:8]
        split = f"outer_{number}_inner_1"
        f0 = results / f"summary-{split}-u5000-s0" / "scores.csv"
        f0_rows = ["arm,config_id,corr_mean,split,seed,updates"]
        for arm in ("adamw", "spectral"):
            f0_rows.extend(f"{arm},{config},0.1,{split},0,5000" for config in range(40))
        _write(f0, ("\n".join(f0_rows) + "\n").encode())
        _write(results / f"selection-outer_{number}-f0-top12.json", {
            "top": 12, "score_sha256": {str(f0): sha256(f0)},
            "selected": {"adamw": promoted, "spectral": promoted,
                         "paired_union": promoted},
        })
        f1_paths, f2_paths = [], []
        for inner in range(1, number + 2):
            split = f"outer_{number}_inner_{inner}"
            f1 = results / f"summary-{split}-u20000-s0" / "scores.csv"
            f1_rows = ["arm,config_id,corr_mean,split,seed,updates"]
            for arm in ("adamw", "spectral"):
                f1_rows.extend(f"{arm},{config},0.1,{split},0,20000"
                               for config in promoted)
            _write(f1, ("\n".join(f1_rows) + "\n").encode())
            f1_paths.append(f1)
            for budget in (5000, 20000, 100000):
                for seed in range(3):
                    f2 = results / f"summary-f2-{split}-u{budget}-s{seed}" / "scores.csv"
                    f2_rows = ["arm,config_id,corr_mean,split,seed,updates"]
                    f2_rows.extend(f"adamw,{config},0.1,{split},{seed},{budget}"
                                   for config in sorted(ordinary))
                    spectral_ids = (
                        ordinary | {1070512} if budget == 100000 else ordinary
                    )
                    f2_rows.extend(f"spectral,{config},0.1,{split},{seed},{budget}"
                                   for config in sorted(spectral_ids))
                    _write(f2, ("\n".join(f2_rows) + "\n").encode())
                    f2_paths.append(f2)
        f1_by_outer[number], f2_by_outer[number] = f1_paths, f2_paths
    source_path = _write(results / "selection-high-rank-source-r2048.json", {
        "status": "development_only_high_rank_source_selection", "requested_rank": 2048,
        "selected": {"spectral": [7]},
        "score_sha256": {str(path): sha256(path) for path in f1_by_outer[1]},
    })
    extension = _write(results / "search-v1-high-rank-extension.json", {
        "status": "development_only_high_rank_extension", "source_config_id": 7,
        "source_search_sha256": sha256(base_search),
        "source_selection_sha256": sha256(source_path), "configs": [
            {"arm": "adamw", "config_id": 1070512},
            {"arm": "spectral", "config_id": 1070512, "rank": 512},
        ],
    })
    probes = _write(results / "audit-high-rank-probes.json", {
        "status": "probe_audit_complete", "extension_sha256": sha256(extension),
        "eligible_ranks": [512],
    })
    search = _write(results / "search-v1-high-rank.json", {
        "status": "development_only_augmented_search", "primary_target": "target",
        "configurations_per_arm": 40, "high_rank_config_ids": [1070512],
        "base_search_sha256": sha256(base_search), "extension_sha256": sha256(extension),
        "probe_audit_sha256": sha256(probes), "configs": [],
    })
    for number in range(1, 4):
        ordinary = ordinary_by_outer[number]
        _write(results / f"selection-outer_{number}-f1-top4.json", {
            "status": "f1_selection_augmented_with_gpu_audited_high_ranks",
            "augmented_search_sha256": sha256(search),
            "score_sha256": {str(path): sha256(path) for path in f1_by_outer[number]},
            "selected": {"paired_union": sorted(ordinary),
                         "high_rank_spectral": [1070512]},
        })
        _write(results / f"selection-outer_{number}-f2-budget-top1.json", {
            "top": 1, "allow_asymmetric": True,
            "score_sha256": {str(path): sha256(path) for path in f2_by_outer[number]},
            "selected": {"adamw": [number], "spectral": [number + 3]},
            "selected_updates": {"adamw": [100000], "spectral": [100000]},
        })
        _write(results / f"audit-outer_{number}-budgeted" / "outer-audit.json", {
            "status": "audit_complete", "split": {"name": f"outer_{number}"},
            "updates": {"adamw": 100_000, "spectral": 100_000},
            "seeds": [0, 1, 2], "cells": 6,
            "selected": {"adamw": number, "spectral": number + 3},
        })
    _write(results / "nested-outer" / "nested-outer-report.json", {
        "status": "complete", "outer_splits": ["outer_1", "outer_2", "outer_3"],
        "expected_seeds": [0, 1, 2], "spectral_minus_adamw": {"estimate": 0.1},
    })
    selected = {"arm": "spectral", "model_weight": 0.75, "benchmark_weight": 0.25,
                "benchmark": "v53_lgbm_ender60"}
    candidate = _write(results / "candidate-plan.json", {
        "status": "frozen_train_only_selection", "selected": selected,
    })
    _write(results / "selection-final-outer-winner-union.json", {
        "status": "outer_winners_audited",
        "selected": {"adamw": [1, 2, 3], "spectral": [4, 5, 6],
                     "paired_union": [1, 2, 3, 4, 5, 6]},
        "budgeted_candidates": [
            {"config_id": config_id, "updates": 100000}
            for config_id in range(1, 7)
        ],
    })
    # Final winners must come from the audited union. Use two actual outer winners.
    winners = {"adamw": 1, "spectral": 4}
    _write(results / "selection-final-top1.json", {
        "selected": {"adamw": [1], "spectral": [4]},
        "selected_updates": {"adamw": [100000], "spectral": [100000]},
    })
    _write(results / "audit-final-refits-budgeted" / "refit-audit.json", {
        "status": "audit_complete", "cells": 6,
        "updates": {"adamw": 100_000, "spectral": 100_000},
        "seeds": [0, 1, 2], "selected": winners,
    })
    frozen = {}
    for arm, config in winners.items():
        paths = [_write(results / f"final-refit-u100000-s{seed}-{arm}-c{config}"
                        / "model.pt", f"{arm}-{seed}".encode()) for seed in range(3)]
        frozen[arm] = {"config_id": config, "seeds": [0, 1, 2], "updates": 100_000,
                       "model_sha256": [sha256(path) for path in paths]}
    freeze = _write(results / "freeze.json", {
        "status": "frozen", "code_commit": "a" * 40, "primary_target": "target",
        "primary_benchmark": "v53_lgbm_ender60",
        "code_snapshot_sha256": sha256(code_snapshot),
        "search_sha256": sha256(search),
        "fidelity_protocol_sha256": sha256(tmp_path / "fidelity-protocol.md"),
        "candidate_plan_sha256": sha256(candidate), "candidate_transform": selected,
        "selected": frozen,
    })

    validation = results / "official-validation"
    validation_report = _write(validation / "official-validation-report.json", {
        "status": "complete", "target": "target", "benchmark": "v53_lgbm_ender60",
        "target_alias_audit": {
            "target_equals_target_ender_60": True,
            "live_corr20v2_target": "target_cyrus_20",
            "live_target_released_in_v5_3": False,
        },
        "freeze_manifest_sha256": sha256(freeze), "spectral_minus_adamw": {},
        "candidate_minus_ender60": {},
        "candidate_transform": selected,
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
    official_predictions = _write(
        bundle / "official-container" / "live_predictions-test.csv", predictions.read_bytes()
    )
    _write(bundle / "official-container" / "official-container-audit.json", {
        "status": "pass", "expected_sha256": sha256(predictions), "cpu_limit": 1,
        "memory_limit_bytes": 4_000_000_000, "max_seconds": 600,
        "elapsed_seconds": 20, "max_abs_difference": 0.0,
        "allowed_max_abs_difference": 1e-4,
        "official_sha256": sha256(official_predictions), "runner_commit": "b" * 40,
        "image_id": "sha256:" + "c" * 64,
    })
    leaderboard_raw = _write(tmp_path / "leaderboard-raw.json", [{"rank": 1}])
    leaderboard = _write(tmp_path / "leaderboard.json", {
        "status": "complete", "round": 9, "summary": {"rows": 1000},
        "raw_sha256": sha256(leaderboard_raw),
    })
    report_dir = results / "final-report"
    report_html = _write(report_dir / "report.html", b"report")
    _write(report_dir / "report-manifest.json", {
        "status": "complete", "comparability": "historical-direct_live-context-only",
        "inputs": {"outer": sha256(results / "nested-outer" / "nested-outer-report.json"),
                   "validation": sha256(validation_report),
                   "leaderboard": sha256(leaderboard), "freeze": sha256(freeze),
                   "search": sha256(search),
                   "admission": sha256(results / "base-search-memory-admission.json")},
        "artifacts": {report_html.name: sha256(report_html)},
        "selected_configs": {"adamw": {"config_id": 1},
                             "spectral": {"config_id": 4}},
    })
    return results, leaderboard


def test_completion_audit_cross_checks_full_chain(tmp_path):
    results, leaderboard = _complete_tree(tmp_path)
    output = tmp_path / "completion.json"
    report = audit(results, leaderboard, output)
    assert report["status"] == "audit_complete"
    assert report["final_selected"] == {"adamw": 1, "spectral": 4}
    assert len(report["evidence_sha256"]) == 31


def test_completion_audit_rejects_model_changed_after_freeze(tmp_path):
    results, leaderboard = _complete_tree(tmp_path)
    (results / "final-refit-u100000-s1-spectral-c4" / "model.pt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="frozen model hashes"):
        audit(results, leaderboard, tmp_path / "bad.json")
