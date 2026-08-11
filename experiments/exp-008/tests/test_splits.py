from numerai_competitive.splits import (
    PURGE_ERAS_60D,
    assert_no_future_leakage,
    inner_splits_60d,
    outer_splits_60d,
    resolve_split_60d,
    sealed_validation_refit_60d,
)

ERAS = [f"{i:04d}" for i in range(1, 575)]


def test_outer_and_inner_are_purged_and_disjoint():
    outer = outer_splits_60d(ERAS)
    assert [(x.valid_eras[0], x.valid_eras[-1]) for x in outer] == [
        ("0313", "0390"), ("0391", "0468"), ("0469", "0574")
    ]
    for split in outer:
        assert_no_future_leakage(split, PURGE_ERAS_60D)
        assert not (set(split.train_eras) & set(split.valid_eras))
        for inner in inner_splits_60d(split):
            assert_no_future_leakage(inner, PURGE_ERAS_60D)
            assert max(map(int, inner.valid_eras)) < min(map(int, split.valid_eras))


def test_named_resolver_uses_sixteen_era_purge():
    split = resolve_split_60d("outer_1_inner_1")
    assert split.train_eras[-1] == "0140"
    assert split.purged_eras == tuple(f"{era:04d}" for era in range(141, 157))
    assert split.valid_eras[0] == "0157"


def test_sealed_validation_refit_purges_last_sixteen_train_eras():
    split = sealed_validation_refit_60d(ERAS)
    assert split.name == "sealed_validation_refit_60d"
    assert split.train_eras[-1] == "0558"
    assert split.purged_eras == tuple(f"{era:04d}" for era in range(559, 575))
    assert not split.valid_eras
