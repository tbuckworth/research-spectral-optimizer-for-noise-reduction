import json

import pytest

from numerai_competitive.data import sha256
from numerai_competitive.high_rank_search import build_augmented_search
from numerai_competitive.high_rank_selection import augment_selection


def _write(path, value):
    path.write_text(json.dumps(value))
    return path


def test_augmented_search_includes_only_gpu_audited_paired_ranks(tmp_path):
    base = _write(tmp_path / "base.json", {
        "primary_target": "target", "configs": [
            {"arm": "adamw", "config_id": 4},
            {"arm": "spectral", "config_id": 4, "rank": 256},
        ],
    })
    extension = _write(tmp_path / "extension.json", {
        "status": "development_only_high_rank_extension",
        "source_search_sha256": sha256(base),
        "configs": [
            {"arm": "adamw", "config_id": 1_040_512},
            {"arm": "spectral", "config_id": 1_040_512, "rank": 512},
            {"arm": "adamw", "config_id": 1_041_024},
            {"arm": "spectral", "config_id": 1_041_024, "rank": 1024},
        ],
    })
    audit = _write(tmp_path / "audit.json", {
        "status": "probe_audit_complete", "extension_sha256": sha256(extension),
        "eligible_ranks": [1024],
    })
    output = tmp_path / "augmented.json"
    report = build_augmented_search(base, extension, audit, output)
    assert report["high_rank_config_ids"] == [1_041_024]
    assert [(row["arm"], row["config_id"]) for row in report["configs"]] == [
        ("adamw", 4), ("spectral", 4),
        ("adamw", 1_041_024), ("spectral", 1_041_024),
    ]
    assert report["probe_audit_sha256"] == sha256(audit)

    selection = _write(tmp_path / "selection.json", {
        "selected": {"adamw": [4], "spectral": [4], "paired_union": [4]},
    })
    promoted = augment_selection(selection, output, tmp_path / "promoted.json")
    assert promoted["selected"]["paired_union"] == [4]
    assert promoted["selected"]["high_rank_spectral"] == [1_041_024]
    assert promoted["selected"]["adamw"] == [4]


def test_augmented_search_rejects_unpaired_eligible_rank(tmp_path):
    base = _write(tmp_path / "base.json", {"configs": []})
    extension = _write(tmp_path / "extension.json", {
        "status": "development_only_high_rank_extension",
        "source_search_sha256": sha256(base),
        "configs": [{"arm": "spectral", "config_id": 1_000_512, "rank": 512}],
    })
    audit = _write(tmp_path / "audit.json", {
        "status": "probe_audit_complete", "extension_sha256": sha256(extension),
        "eligible_ranks": [512],
    })
    with pytest.raises(ValueError, match="paired"):
        build_augmented_search(base, extension, audit, tmp_path / "bad.json")
