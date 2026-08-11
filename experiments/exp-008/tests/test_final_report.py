import json
from pathlib import Path

from numerai_competitive.data import sha256
from numerai_competitive.final_report import build_report


def _write(path: Path, value: dict) -> Path:
    path.write_text(json.dumps(value))
    return path


def _comparison(value: float) -> dict:
    return {"estimate": value, "ci_low": value - 0.01, "ci_high": value + 0.01,
            "probability_positive": 0.75, "eras": 20, "block_length": 8,
            "samples": 10_000, "seed": 20260810}


def test_final_report_keeps_live_and_historical_scales_separate(tmp_path: Path):
    score = {"corr": {"mean": 0.02, "sharpe": 1.1}}
    outer = _write(tmp_path / "outer.json", {"status": "complete", "adamw": score,
                   "spectral": score, "spectral_minus_adamw": _comparison(0.001)})
    validation = _write(tmp_path / "validation.json", {
        "status": "complete", "target": "target_cyrusd_20", "adamw": score,
        "spectral": score, "candidate": score,
        "ender20": {"mean": 0.019, "sharpe": 1.0},
        "spectral_minus_adamw": _comparison(0.002),
        "adamw_minus_ender20": _comparison(0.001),
        "spectral_minus_ender20": _comparison(0.003),
        "candidate_minus_ender20": _comparison(0.003),
    })
    metric = {"label": "CORR20v2", "median": 0.01, "p90": 0.02, "maximum": 0.03}
    leaderboard = _write(tmp_path / "leaderboard.json", {
        "status": "complete", "round": 9, "retrieved_at": "now",
        "summary": {"rows": 1000, "metrics": {"corr20V2Rep": metric}},
    })
    search = _write(tmp_path / "search.json", {
        "protocol": "paired-search", "primary_target": "target_cyrusd_20",
        "configurations_per_arm": 40,
        "configs": [
            {"arm": arm, "config_id": config_id, "width": 512,
             "learning_rate": 0.001, **({"rank": 32} if arm == "spectral" else {})}
            for config_id in range(40) for arm in ("adamw", "spectral")
        ],
    })
    freeze = _write(tmp_path / "freeze.json", {
        "status": "frozen", "primary_target": "target_cyrusd_20",
        "search_sha256": sha256(search),
        "selected": {
            "adamw": {"config_id": 1, "updates": 100_000, "seeds": [0, 1, 2]},
            "spectral": {"config_id": 4, "updates": 100_000, "seeds": [0, 1, 2]},
        },
    })
    output = tmp_path / "report"
    manifest = build_report(outer, validation, leaderboard, freeze, search, output)
    assert manifest["comparability"] == "historical-direct_live-context-only"
    html = (output / "report.html").read_text()
    assert "cannot honestly be translated" in html
    assert "target_cyrusd_20" in html
    assert "40 paired configuration IDs" in html
    assert "Selected AdamW" in html
    assert (output / "historical-comparison.png").stat().st_size > 1000
    saved = json.loads((output / "report-manifest.json").read_text())
    assert saved["artifacts"] == manifest["artifacts"]
