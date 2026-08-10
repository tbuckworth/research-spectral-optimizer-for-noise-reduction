import pytest

from numerai_competitive.run_task import resolve_split


def test_resolve_split_finds_nested_and_outer_folds():
    assert resolve_split("outer_1").valid_eras[0] == "0313"
    assert resolve_split("outer_3_inner_4").valid_eras[0] == "0391"


def test_resolve_split_rejects_unknown_name():
    with pytest.raises(ValueError, match="unknown split"):
        resolve_split("future")
