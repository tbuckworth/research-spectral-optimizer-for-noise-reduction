import json
from pathlib import Path

from numerai_competitive.base_admission import create_admission
from numerai_competitive.high_rank import memory_estimate, required_probe_updates


def _draws(width=512, rank=16):
    shared = {
        "config_id": 0, "feature_set": "medium", "width": width, "depth": 2,
        "normalization": "none", "filter_update_every": 1, "filter_warmup": 0,
    }
    return [
        {"arm": "adamw", **shared},
        {"arm": "spectral", **shared, "rank": rank},
    ]


def _write_search(tmp_path: Path, draws: list[dict]) -> Path:
    path = tmp_path / "search.json"
    path.write_text(json.dumps({"configs": draws}))
    return path


def test_static_safe_pair_is_admitted_without_probe(tmp_path):
    search = _write_search(tmp_path, _draws())
    report = create_admission(search, tmp_path / "results", tmp_path / "audit.json",
                              final=False)
    assert report["status"] == "complete"
    assert report["admitted_config_ids"] == [0]
    assert report["rows"][0]["state"] == "static_safe"


def test_borderline_pair_waits_for_probe_then_excludes_symmetrically(tmp_path):
    draws = _draws(width=4096, rank=4096)
    search = _write_search(tmp_path, draws)
    pending = create_admission(search, tmp_path / "results", tmp_path / "pending.json",
                               final=False)
    assert pending["status"] == "pending_probe"
    assert pending["pending_probe_config_ids"] == [0]
    final = create_admission(search, tmp_path / "results", tmp_path / "final.json",
                             final=True)
    assert final["admitted_config_ids"] == []
    assert final["excluded_config_ids"] == [0]


def test_borderline_pair_can_be_admitted_by_verified_gpu_probe(tmp_path):
    draws = _draws(width=4096, rank=4096)
    search = _write_search(tmp_path, draws)
    spectral = draws[1]
    estimate = memory_estimate(spectral, spectral["rank"])
    result_dir = tmp_path / "results" / "stage-outer_1_inner_1-u5000-s0-spectral-c0"
    result_dir.mkdir(parents=True)
    (result_dir / "result.json").write_text(json.dumps({
        "status": "complete", "updates": required_probe_updates(spectral, spectral["rank"]),
        "parameter_count": estimate["parameter_count"], "peak_cuda_memory_bytes": 1,
        "config": {"arm": "spectral", "search_config_id": 0,
                   "filter": {"rank": spectral["rank"]}},
    }))
    report = create_admission(search, tmp_path / "results", tmp_path / "audit.json",
                              final=True)
    assert report["admitted_config_ids"] == [0]
    assert report["rows"][0]["state"] == "empirical_probe_passed"
    assert report["rows"][0]["probe"]["passed"]


def test_unpaired_search_is_rejected(tmp_path):
    search = _write_search(tmp_path, [_draws()[0]])
    try:
        create_admission(search, tmp_path / "results", tmp_path / "audit.json", final=True)
    except ValueError as exc:
        assert "pair" in str(exc)
    else:
        raise AssertionError("unpaired search should fail")
