import json
from pathlib import Path

import pytest

from numerai_competitive.completion_audit import _audit_successive_development
from numerai_competitive.data import sha256


def _write(path: Path, value) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) if isinstance(value, dict) else value)
    return path


def _scores(path: Path, ids: set[int], *, budget: int, split: str, seed: int = 0) -> Path:
    rows = ["arm,config_id,corr_mean,split,seed,updates"]
    for arm in ("adamw", "spectral"):
        rows.extend(f"{arm},{config},0.1,{split},{seed},{budget}" for config in sorted(ids))
    return _write(path, "\n".join(rows) + "\n")


def _tree(tmp_path: Path):
    results = tmp_path / "results"
    admitted = set(range(40))
    promoted = set(range(12))
    f0 = _scores(results / "summary-outer_1_inner_1-u5000-s0/scores.csv",
                 admitted, budget=5000, split="outer_1_inner_1")
    f0_selection = _write(results / "selection-outer_1-f0-top12.json", {
        "top": 12, "score_sha256": {str(f0): sha256(f0)},
        "selected": {"adamw": list(range(12)), "spectral": list(range(12)),
                     "paired_union": list(range(12))},
    })
    f1 = [
        _scores(results / f"summary-outer_1_inner_{inner}-u20000-s0/scores.csv",
                promoted, budget=20000, split=f"outer_1_inner_{inner}")
        for inner in (1, 2)
    ]
    search = _write(results / "search-v1-high-rank.json", {
        "status": "development_only_augmented_search", "high_rank_config_ids": [101],
    })
    plan = _write(results / "selection-outer_1-successive-plan.json", {
        "status": "successive_halving_plan_frozen",
        "confirmation_top_per_arm_per_fidelity": 2,
        "long_scout_top_per_arm_per_fidelity": 1,
        "score_sha256": {str(path): sha256(path) for path in [f0, *f1]},
        "augmented_search_sha256": sha256(search), "high_rank_spectral": [101],
        "high_rank_source_config_id": 7,
        "confirmation_selections": {
            "5000": {"adamw": [1, 2], "spectral": [2, 3], "paired_union": [1, 2, 3]},
            "20000": {"adamw": [4, 7], "spectral": [5, 7],
                      "paired_union": [4, 5, 7]},
        },
        "long_scout_paired_union": [1, 5, 7],
    })
    ordinary = _scores(results / "summary-f2a-outer_1_inner_1-u100000-s0/scores.csv",
                       {1, 5, 7}, budget=100000, split="outer_1_inner_1")
    high = [
        _scores(results / f"summary-f2a-outer_1_inner_{inner}-u20000-s0/scores.csv",
                promoted | {101}, budget=20000, split=f"outer_1_inner_{inner}")
        for inner in (1, 2)
    ]
    finalists = _write(results / "selection-outer_1-successive-finalists.json", {
        "status": "successive_halving_finalists_frozen",
        "score_sha256": {str(path): sha256(path) for path in [ordinary, *high]},
        "high_rank_spectral": [101], "high_rank_source_config_id": 7,
        "ordinary_confirmation_paired_union": [1, 5, 7],
        "ordinary_winners": {"adamw": [1], "spectral": [5]},
    })
    assembled = _scores(results / "summary-outer_1-successive-equal-coverage/scores.csv",
                        {1, 5, 7}, budget=100000, split="outer_1_inner_1")
    _write(assembled.parent / "assembly-audit.json", {
        "status": "successive_scores_equal_coverage",
        "splits": ["outer_1_inner_1", "outer_1_inner_2"], "seeds": [0, 1, 2],
        "plan_sha256": sha256(plan), "finalists_sha256": sha256(finalists),
        "scores_sha256": sha256(assembled), "source_score_sha256": {},
    })
    development = _write(results / "selection-outer_1-f2-budget-top1.json", {
        "status": "development_budget_sensitivity_selection", "top": 1,
        "allow_asymmetric": True, "score_sha256": {str(assembled): sha256(assembled)},
        "selected": {"adamw": [1], "spectral": [101]},
        "selected_updates": {"adamw": [5000], "spectral": [100000]},
    })
    audits = []
    for number in range(1, 4):
        audits.append(_write(results / f"audit-outer_{number}-budgeted/outer-audit.json", {
            "status": "audit_complete", "split": {"name": f"outer_{number}"},
            "seeds": [0, 1, 2], "cells": 6,
            "selected": {"adamw": 1, "spectral": 101},
            "updates": {"adamw": 5000, "spectral": 100000},
        }))
    final = _write(results / "selection-final-top1.json", {
        "status": "fixed_config_walkforward_confirmed",
        "source_selection_sha256": sha256(development),
        "reselected_on_outer_outcomes": False,
        "selected": {"adamw": [1], "spectral": [101]},
        "selected_updates": {"adamw": [5000], "spectral": [100000]},
        "walkforward_audit_sha256": {
            f"outer_{number}": sha256(path) for number, path in enumerate(audits, 1)
        },
    })
    return results, admitted, search, {"source_config_id": 7}, [101], final, f0_selection


def test_audits_hash_bound_halving_and_fixed_outer_selection(tmp_path):
    results, admitted, search, extension, high_ids, final, _ = _tree(tmp_path)
    evidence = {}
    observed, _, path, _, _ = _audit_successive_development(
        results, admitted, search, extension, high_ids, evidence,
    )
    assert observed == json.loads(final.read_text())
    assert path == final
    assert set(evidence) >= {"successive_plan", "successive_finalists", "final_selection"}


def test_rejects_outer_reselection_flag(tmp_path):
    results, admitted, search, extension, high_ids, final, _ = _tree(tmp_path)
    value = json.loads(final.read_text())
    value["reselected_on_outer_outcomes"] = True
    final.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="changed by walk-forward"):
        _audit_successive_development(results, admitted, search, extension, high_ids, {})
