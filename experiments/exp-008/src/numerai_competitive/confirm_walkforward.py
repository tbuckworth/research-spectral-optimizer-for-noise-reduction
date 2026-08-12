"""Confirm a pre-outer winner without reselecting on walk-forward outcomes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import atomic_json, sha256


def confirm(selection_path: Path, audit_paths: list[Path], output: Path) -> dict:
    source = json.loads(selection_path.read_text())
    selected = source.get("selected", {})
    updates = source.get("selected_updates", {})
    if any(len(selected.get(arm, [])) != 1 or len(updates.get(arm, [])) != 1
           for arm in ("adamw", "spectral")):
        raise ValueError("development selection must contain one config/budget per arm")
    if len(audit_paths) != 3:
        raise ValueError("all three walk-forward audits are required")
    audit_hashes = {}
    for number, path in enumerate(audit_paths, 1):
        audit = json.loads(path.read_text())
        if (audit.get("status") != "audit_complete"
                or audit.get("split", {}).get("name") != f"outer_{number}"
                or any(audit.get("selected", {}).get(arm) != selected[arm][0]
                       or audit.get("updates", {}).get(arm) != updates[arm][0]
                       for arm in ("adamw", "spectral"))):
            raise ValueError(f"outer_{number} audit differs from the fixed selection")
        audit_hashes[f"outer_{number}"] = sha256(path)
    report = dict(source)
    report.update({
        "status": "fixed_config_walkforward_confirmed",
        "selection_origin_status": source.get("status"),
        "source_selection_sha256": sha256(selection_path),
        "walkforward_audit_sha256": audit_hashes,
        "reselected_on_outer_outcomes": False,
    })
    atomic_json(output, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--audit", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(confirm(args.selection, args.audit, args.output), sort_keys=True))


if __name__ == "__main__":
    main()
