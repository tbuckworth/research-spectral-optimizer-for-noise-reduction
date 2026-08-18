from __future__ import annotations

import json

from numerai_competitive.corrected_live import CHECKPOINT_EXAMPLES, select_stopping


def _result(path, arm, fold, scores):
    batch = 1024 if arm == "adamw" else 2048
    maximum = 40_000 if arm == "adamw" else 20_000
    config_id = 38 if arm == "adamw" else 39
    path.write_text(json.dumps({
        "status": "complete",
        "config": {"arm": arm, "search_config_id": config_id, "seed": 0,
                   "updates": maximum, "schedule_updates": maximum,
                   "batch_size": batch},
        "split": {"name": fold, "train_eras": list(range(218 if fold.endswith("2") else 140))},
        "validation_history": [
            {"examples": examples, "validation": {"corr": {"mean": score}}}
            for examples, score in zip(CHECKPOINT_EXAMPLES, scores)
        ],
    }))
    return path


def test_selects_each_arm_peak_and_scales_schedule_by_eras(tmp_path):
    paths = []
    for arm, scores in (("adamw", [1, 3, 2, 0, -1]),
                        ("spectral", [0, 1, 2, 4, 3])):
        paths.append(_result(tmp_path / f"{arm}-1.json", arm, "outer_1_inner_1", scores))
        paths.append(_result(tmp_path / f"{arm}-2.json", arm, "outer_1_inner_2", scores))
    protocol = tmp_path / "protocol.md"
    protocol.write_text("frozen")
    result = select_stopping(paths, protocol, "a" * 40, tmp_path / "freeze.json")
    adam, spectral = result["selected"]["adamw"], result["selected"]["spectral"]
    assert adam["calibration_examples"] == 5_120_000
    assert spectral["calibration_examples"] == 20_480_000
    assert adam["refit_updates"] == round(5_000 * 574 / 218)
    assert spectral["refit_updates"] == round(10_000 * 574 / 218)
    assert adam["refit_schedule_updates"] == round(40_000 * 574 / 218)
    assert result["staking_authorized"] is False
