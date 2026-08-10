"""Deterministic nested expanding era splits with target-horizon purges."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence


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


def outer_splits_20d(eras: Sequence[str]) -> list[EraSplit]:
    """Three untouched development blocks; latest block absorbs eras through 574."""
    return [
        expanding_split(eras, name="outer_1", valid_start=313, valid_end=390, purge=8),
        expanding_split(eras, name="outer_2", valid_start=391, valid_end=468, purge=8),
        expanding_split(eras, name="outer_3", valid_start=469, valid_end=574, purge=8),
    ]


def inner_splits_20d(outer: EraSplit) -> list[EraSplit]:
    """Two or three inner blocks wholly before an outer validation block."""
    max_train = max(map(era_key, outer.train_eras))
    candidates = [(157, 234), (235, 312), (313, 390), (391, 468)]
    usable = [(a, min(b, max_train)) for a, b in candidates if a <= max_train]
    usable = [(a, b) for a, b in usable if b - a + 1 >= 48]
    if len(usable) < 2:
        raise ValueError(f"{outer.name}: fewer than two usable inner folds")
    return [
        expanding_split(outer.train_eras, name=f"{outer.name}_inner_{i+1}",
                        valid_start=a, valid_end=b, purge=8)
        for i, (a, b) in enumerate(usable)
    ]


def assert_no_future_leakage(split: EraSplit, purge: int) -> None:
    train_max = max(map(era_key, split.train_eras))
    valid_min = min(map(era_key, split.valid_eras))
    if valid_min - train_max - 1 != purge:
        raise AssertionError(
            f"{split.name}: gap={valid_min - train_max - 1}, expected purge={purge}"
        )
