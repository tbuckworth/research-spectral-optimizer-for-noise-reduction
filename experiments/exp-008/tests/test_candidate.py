from __future__ import annotations

import numpy as np
import pytest

from numerai_competitive.candidate import select_candidate


def _oof(path, *, unstable: bool = False):
    rng = np.random.default_rng(8)
    eras = np.repeat(np.arange(1, 9), 20)
    rows = np.arange(len(eras))
    target = np.tile(np.linspace(0.02, 0.98, 20), 8)
    benchmark = target + rng.normal(0, 0.35, len(rows))
    adamw = target + rng.normal(0, 0.25, len(rows))
    spectral = target + rng.normal(0, 0.15, len(rows))
    if unstable:
        adamw[eras >= 5] *= -1
        spectral[eras >= 5] *= -1
    np.savez_compressed(
        path, row_index=rows, era=eras, target=target, benchmark=benchmark,
        adamw=adamw, spectral=spectral,
        split=np.where(eras <= 4, "outer_1", "outer_2"),
    )


def test_candidate_selection_uses_only_hashed_nested_outer_artifact(tmp_path):
    source = tmp_path / "oof.npz"
    _oof(source)
    payload = select_candidate(source, tmp_path / "candidate.json")
    assert payload["status"] == "frozen_train_only_selection"
    assert payload["selected"]["arm"] in {"adamw", "spectral"}
    assert payload["selected"]["model_weight"] in payload["model_weights"]
    assert len(payload["source_oof_sha256"]) == 64
    assert (tmp_path / "candidate.json").is_file()


def test_candidate_selection_rejects_arms_unstable_across_outer_folds(tmp_path):
    source = tmp_path / "oof.npz"
    _oof(source, unstable=True)
    with pytest.raises(ValueError, match="positive standalone CORR in every outer fold"):
        select_candidate(source, tmp_path / "candidate.json")
