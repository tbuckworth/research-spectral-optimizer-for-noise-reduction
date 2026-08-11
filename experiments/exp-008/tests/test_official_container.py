import json

import pandas as pd
import pytest

from numerai_competitive.official_container import audit


def test_official_container_audit_requires_same_ids_and_near_identical_predictions(tmp_path):
    expected = tmp_path / "expected.csv"
    official = tmp_path / "official.csv"
    pd.DataFrame({"prediction": [0.1, 0.4, 0.8]}, index=["a", "b", "c"]).to_csv(expected)
    pd.DataFrame({"prediction": [0.10001, 0.4, 0.79999]}, index=["a", "b", "c"]).to_csv(
        official
    )
    output = tmp_path / "audit.json"
    report = audit(expected, official, output, runner_commit="a" * 40,
                   image_id="sha256:" + "1" * 64, elapsed_seconds=12)
    assert report["status"] == "pass" and report["rows"] == 3
    assert json.loads(output.read_text())["official_sha256"] == report["official_sha256"]
    pd.DataFrame({"prediction": [0.1, 0.4, 0.8]}, index=["b", "a", "c"]).to_csv(official)
    with pytest.raises(ValueError, match="IDs or row order"):
        audit(expected, official, tmp_path / "bad.json", runner_commit="a" * 40,
              image_id="sha256:" + "1" * 64, elapsed_seconds=12)


def test_official_container_audit_fails_runtime_or_prediction_drift(tmp_path):
    expected = tmp_path / "expected.csv"
    official = tmp_path / "official.csv"
    pd.DataFrame({"prediction": [0.1, 0.9]}, index=["a", "b"]).to_csv(expected)
    pd.DataFrame({"prediction": [0.2, 0.8]}, index=["a", "b"]).to_csv(official)
    with pytest.raises(RuntimeError, match="compatibility failed"):
        audit(expected, official, tmp_path / "audit.json", runner_commit="b" * 40,
              image_id="sha256:" + "4" * 64, elapsed_seconds=601)
    assert json.loads((tmp_path / "audit.json").read_text())["status"] == "fail"
