import json

from numerai_competitive.leaderboard_snapshot import create_snapshot, summarize


class FakeAPI:
    def get_leaderboard(self, limit, offset):
        assert limit == 3 and offset == 0
        return [
            {"rank": 1, "username": "first", "corr20V2Rep": 0.03, "mmcRep": 0.01,
             "bmcRep": 0.02, "corj60Rep": 0.04},
            {"rank": 2, "username": "second", "corr20V2Rep": 0.02, "mmcRep": 0.00,
             "bmcRep": 0.01, "corj60Rep": 0.03},
            {"rank": 3, "username": "third", "corr20V2Rep": 0.01, "mmcRep": -0.01,
             "bmcRep": 0.00, "corj60Rep": 0.02},
        ]

    def get_current_round(self):
        return 99


def test_snapshot_preserves_raw_response_and_distinguishes_live_scale(tmp_path):
    report = create_snapshot(
        tmp_path, limit=3, api=FakeAPI(), retrieved_at="2026-01-01T00:00:00+00:00",
    )
    assert report["round"] == 99
    assert report["summary"]["metrics"]["corr20V2Rep"]["median"] == 0.02
    assert len(report["raw_sha256"]) == 64
    assert json.loads((tmp_path / "leaderboard-raw.json").read_text())[0]["rank"] == 1
    assert "cannot be converted" in (tmp_path / "leaderboard-summary.md").read_text()


def test_summary_allows_tied_ranks_but_rejects_duplicate_models():
    rows = FakeAPI().get_leaderboard(3, 0)
    rows[1]["rank"] = 1
    assert summarize(rows)["top_rank_models"] == ["first", "second"]
    rows[1]["username"] = "first"
    try:
        summarize(rows)
    except ValueError as error:
        assert "duplicate model" in str(error)
    else:
        raise AssertionError("duplicate models were accepted")


def test_summary_records_metric_specific_missing_rows():
    rows = FakeAPI().get_leaderboard(3, 0)
    rows[-1]["corj60Rep"] = None
    metric = summarize(rows)["metrics"]["corj60Rep"]
    assert metric["finite_rows"] == 2 and metric["missing_rows"] == 1
