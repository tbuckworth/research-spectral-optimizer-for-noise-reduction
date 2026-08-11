import json

import pytest

from numerai_competitive.data import sha256
from numerai_competitive.high_rank_probe import audit_probes


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return path


def test_probe_audit_requires_realized_rank_and_safe_measured_memory(tmp_path):
    extension = _write(tmp_path / "extension.json", {
        "status": "development_only_high_rank_extension",
        "configs": [{"config_id": 10512, "rank": 512, "memory_screen": {
            "analytically_feasible_48gb": True, "parameter_count": 100,
        }}],
    })
    result = _write(tmp_path / "probe" / "result.json", {
        "status": "complete", "parameter_count": 100, "updates": 1024,
        "peak_cuda_memory_bytes": 20_000_000_000,
        "config": {"search_config_id": 10512, "arm": "spectral",
                   "filter": {"rank": 512}},
        "logs": [{"filter": {"basis_rank": 512, "orthogonality_error": 1e-5}}],
    })
    output = tmp_path / "audit.json"
    report = audit_probes(extension, [result], output)
    assert report["status"] == "probe_audit_complete"
    assert report["eligible_ranks"] == [512]
    assert report["probes"][0]["result_sha256"] == sha256(result)
    broken = json.loads(result.read_text())
    broken["logs"][-1]["filter"]["basis_rank"] = 511
    result.write_text(json.dumps(broken))
    with pytest.raises(ValueError, match="safe full-rank"):
        audit_probes(extension, [result], tmp_path / "bad.json")


def test_probe_audit_records_when_no_rank_is_analytically_feasible(tmp_path):
    extension = _write(tmp_path / "extension.json", {
        "status": "development_only_high_rank_extension",
        "configs": [{"config_id": 11024, "rank": 1024, "memory_screen": {
            "analytically_feasible_48gb": False, "parameter_count": 10_000_000,
        }}],
    })
    report = audit_probes(extension, [], tmp_path / "audit.json")
    assert report["status"] == "complete_no_analytically_feasible_rank"
    assert report["eligible_ranks"] == []
