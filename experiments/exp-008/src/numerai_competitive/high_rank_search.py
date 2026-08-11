"""Build the executable search manifest after audited high-rank GPU probes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import atomic_json, sha256


def build_augmented_search(search_path: Path, extension_path: Path,
                           probe_audit_path: Path, output: Path) -> dict:
    search = json.loads(search_path.read_text())
    extension = json.loads(extension_path.read_text())
    audit = json.loads(probe_audit_path.read_text())
    if (extension.get("status") != "development_only_high_rank_extension"
            or extension.get("source_search_sha256") != sha256(search_path)
            or audit.get("status") != "probe_audit_complete"
            or audit.get("extension_sha256") != sha256(extension_path)):
        raise ValueError("high-rank search inputs fail provenance or completion checks")
    eligible = set(audit.get("eligible_ranks", []))
    if not eligible:
        raise ValueError("no GPU-probed high rank is eligible")
    additions = [config for config in extension.get("configs", [])
                 if config.get("rank") in eligible
                 or (config.get("arm") == "adamw"
                     and any(other.get("config_id") == config.get("config_id")
                             and other.get("rank") in eligible
                             for other in extension.get("configs", [])))]
    expected_ids = {config["config_id"] for config in additions}
    if any(sum(config.get("config_id") == config_id and config.get("arm") == arm
               for config in additions) != 1
           for config_id in expected_ids for arm in ("adamw", "spectral")):
        raise ValueError("eligible extension is not exactly paired by arm")
    configs = [*search.get("configs", []), *additions]
    keys = [(config.get("arm"), config.get("config_id")) for config in configs]
    if len(keys) != len(set(keys)):
        raise ValueError("augmented search has duplicate arm/config IDs")
    report = {key: value for key, value in search.items() if key != "configs"}
    report.update({
        "status": "development_only_augmented_search",
        "base_search_sha256": sha256(search_path),
        "extension_sha256": sha256(extension_path),
        "probe_audit_sha256": sha256(probe_audit_path),
        "high_rank_config_ids": sorted(expected_ids),
        "configs": configs,
    })
    atomic_json(output, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--search", type=Path, required=True)
    parser.add_argument("--extension", type=Path, required=True)
    parser.add_argument("--probe-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build_augmented_search(
        args.search, args.extension, args.probe_audit, args.output
    ), sort_keys=True))


if __name__ == "__main__":
    main()
