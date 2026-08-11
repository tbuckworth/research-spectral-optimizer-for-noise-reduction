"""Freeze the paired union of independently selected nested-outer winners."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import atomic_json, sha256


def collect_candidates(selections: list[Path], audits: list[Path]) -> dict:
    if len(selections) != 3 or len(audits) != 3:
        raise ValueError("final selection requires exactly three outer selections and audits")
    winners: dict[str, list[int]] = {"adamw": [], "spectral": []}
    outer_folds = []
    for expected_number, (selection_path, audit_path) in enumerate(
        zip(selections, audits, strict=True), start=1,
    ):
        selection = json.loads(selection_path.read_text())
        audit = json.loads(audit_path.read_text())
        expected_split = f"outer_{expected_number}"
        if (audit.get("status") != "audit_complete"
                or audit.get("split", {}).get("name") != expected_split
                or audit.get("updates") != 100000
                or audit.get("seeds") != [0, 1, 2]
                or audit.get("cells") != 6):
            raise ValueError(f"{audit_path}: incomplete or misordered outer audit")
        selected = selection.get("selected", {})
        selected_winners = {}
        for arm in ("adamw", "spectral"):
            ids = selected.get(arm, [])
            if len(ids) != 1 or not isinstance(ids[0], int) or ids[0] < 0:
                raise ValueError(f"{selection_path}: {arm} must contain one config ID")
            selected_winners[arm] = ids[0]
            winners[arm].append(ids[0])
        if audit.get("selected") != selected_winners:
            raise ValueError(f"{audit_path}: audited winners differ from selection")
        outer_folds.append({
            "split": expected_split,
            "selection": str(selection_path),
            "selection_sha256": sha256(selection_path),
            "audit": str(audit_path),
            "audit_sha256": sha256(audit_path),
            "selected": selected_winners,
        })
    selected = {
        arm: sorted(set(ids)) for arm, ids in winners.items()
    }
    selected["paired_union"] = sorted(set(selected["adamw"]) | set(selected["spectral"]))
    return {
        "status": "outer_winners_audited",
        "criterion": (
            "paired union of one independently inner-selected winner per optimizer and outer fold"
        ),
        "outer_folds": outer_folds,
        "selected": selected,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, action="append", required=True)
    parser.add_argument("--audit", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = collect_candidates(args.selection, args.audit)
    atomic_json(args.output, payload)
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
