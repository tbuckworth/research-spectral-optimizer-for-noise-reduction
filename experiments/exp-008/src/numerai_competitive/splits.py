"""Deterministic nested expanding era splits with target-horizon purges."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass

PURGE_ERAS_60D = 16


def era_key(era: str | int) -> int:
    return int(era)


@dataclass(frozen=True)
class EraSplit:
    name: str
    train_eras: tuple[str, ...]
    valid_eras: tuple[str, ...]
    purged_eras: tuple[str, ...]

    def to_dict(self) -> dict:
        value = asdict(self)
        return {k: list(v) if isinstance(v, tuple) else v for k, v in value.items()}


def expanding_split(
    eras: Sequence[str], *, name: str, valid_start: int, valid_end: int, purge: int
) -> EraSplit:
    ordered = tuple(sorted({str(e) for e in eras}, key=era_key))
    valid = tuple(e for e in ordered if valid_start <= era_key(e) <= valid_end)
    if not valid:
        raise ValueError(f"{name}: empty validation interval")
    train_cut = valid_start - purge - 1
    train = tuple(e for e in ordered if era_key(e) <= train_cut)
    purged = tuple(e for e in ordered if train_cut < era_key(e) < valid_start)
    if len(purged) != purge:
        raise ValueError(f"{name}: expected {purge} purge eras, got {len(purged)}")
    if set(train) & set(valid) or set(purged) & (set(train) | set(valid)):
        raise AssertionError(f"{name}: era overlap")
    return EraSplit(name, train, valid, purged)


def outer_splits_60d(eras: Sequence[str]) -> list[EraSplit]:
    """Three untouched development blocks; latest block absorbs eras through 574."""
    return [
        expanding_split(
            eras, name="outer_1", valid_start=313, valid_end=390, purge=PURGE_ERAS_60D,
        ),
        expanding_split(
            eras, name="outer_2", valid_start=391, valid_end=468, purge=PURGE_ERAS_60D,
        ),
        expanding_split(
            eras, name="outer_3", valid_start=469, valid_end=574, purge=PURGE_ERAS_60D,
        ),
    ]


def inner_splits_60d(outer: EraSplit) -> list[EraSplit]:
    """Two or three inner blocks wholly before an outer validation block."""
    max_train = max(map(era_key, outer.train_eras))
    candidates = [(157, 234), (235, 312), (313, 390), (391, 468)]
    usable = [(a, min(b, max_train)) for a, b in candidates if a <= max_train]
    usable = [(a, b) for a, b in usable if b - a + 1 >= 48]
    if len(usable) < 2:
        raise ValueError(f"{outer.name}: fewer than two usable inner folds")
    return [
        expanding_split(outer.train_eras, name=f"{outer.name}_inner_{i+1}",
                        valid_start=a, valid_end=b, purge=PURGE_ERAS_60D)
        for i, (a, b) in enumerate(usable)
    ]


def resolve_split_60d(name: str, eras: Sequence[str] | None = None) -> EraSplit:
    """Resolve one protocol split with the frozen 60-day purge."""
    universe = list(eras) if eras is not None else [f"{i:04d}" for i in range(1, 575)]
    outer = {split.name: split for split in outer_splits_60d(universe)}
    all_splits = dict(outer)
    for split in outer.values():
        all_splits.update({inner.name: inner for inner in inner_splits_60d(split)})
    try:
        return all_splits[name]
    except KeyError as exc:
        raise ValueError(f"unknown split {name!r}; choices={sorted(all_splits)}") from exc


def sealed_validation_refit_60d(eras: Sequence[str]) -> EraSplit:
    """Use all train eras except the 60-day overlap with official validation."""
    ordered = tuple(sorted({str(era) for era in eras}, key=era_key))
    if len(ordered) <= PURGE_ERAS_60D:
        raise ValueError("not enough eras for a sealed-validation refit")
    return EraSplit(
        "sealed_validation_refit_60d", ordered[:-PURGE_ERAS_60D], (),
        ordered[-PURGE_ERAS_60D:],
    )


def assert_no_future_leakage(split: EraSplit, purge: int) -> None:
    train_max = max(map(era_key, split.train_eras))
    valid_min = min(map(era_key, split.valid_eras))
    if valid_min - train_max - 1 != purge:
        raise AssertionError(
            f"{split.name}: gap={valid_min - train_max - 1}, expected purge={purge}"
        )
