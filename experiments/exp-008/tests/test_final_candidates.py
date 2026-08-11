import json

import pytest

from numerai_competitive.final_candidates import collect_candidates


def _inputs(tmp_path):
    selections, audits = [], []
    for number in range(1, 4):
        selection = tmp_path / f"selection-{number}.json"
        selection.write_text(json.dumps({
            "selected": {"adamw": [number], "spectral": [number + 2]},
        }))
        audit = tmp_path / f"audit-{number}.json"
        audit.write_text(json.dumps({
            "status": "audit_complete",
            "split": {"name": f"outer_{number}"},
            "updates": 100000, "seeds": [0, 1, 2], "cells": 6,
            "selected": {"adamw": number, "spectral": number + 2},
        }))
        selections.append(selection)
        audits.append(audit)
    return selections, audits


def test_collect_candidates_builds_paired_outer_winner_union(tmp_path):
    selections, audits = _inputs(tmp_path)
    payload = collect_candidates(selections, audits)
    assert payload["status"] == "outer_winners_audited"
    assert payload["selected"] == {
        "adamw": [1, 2, 3], "spectral": [3, 4, 5], "paired_union": [1, 2, 3, 4, 5],
    }
    assert len(payload["outer_folds"]) == 3


def test_collect_candidates_rejects_audit_selection_disagreement(tmp_path):
    selections, audits = _inputs(tmp_path)
    value = json.loads(audits[1].read_text())
    value["selected"]["adamw"] = 9
    audits[1].write_text(json.dumps(value))
    with pytest.raises(ValueError, match="differ"):
        collect_candidates(selections, audits)


def test_collect_candidates_rejects_missing_outer_fold(tmp_path):
    selections, audits = _inputs(tmp_path)
    with pytest.raises(ValueError, match="exactly three"):
        collect_candidates(selections[:2], audits[:2])
