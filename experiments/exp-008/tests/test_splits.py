from numerai_competitive.splits import assert_no_future_leakage, inner_splits_20d, outer_splits_20d

ERAS = [f"{i:04d}" for i in range(1, 575)]


def test_outer_and_inner_are_purged_and_disjoint():
    outer = outer_splits_20d(ERAS)
    assert [(x.valid_eras[0], x.valid_eras[-1]) for x in outer] == [
        ("0313", "0390"), ("0391", "0468"), ("0469", "0574")
    ]
    for split in outer:
        assert_no_future_leakage(split, 8)
        assert not (set(split.train_eras) & set(split.valid_eras))
        for inner in inner_splits_20d(split):
            assert_no_future_leakage(inner, 8)
            assert max(map(int, inner.valid_eras)) < min(map(int, split.valid_eras))
