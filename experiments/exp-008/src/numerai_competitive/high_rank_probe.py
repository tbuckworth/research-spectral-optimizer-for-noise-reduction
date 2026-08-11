"""Audit near-full-rank GPU probes before expensive high-rank training."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import atomic_json, sha256

MAX_PROBE_BYTES = 45 * 1024**3


def audit_probes(extension_path: Path, result_paths: list[Path], output: Path,
                 max_probe_bytes: int = MAX_PROBE_BYTES) -> dict:
    extension = json.loads(extension_path.read_text())
    if extension.get("status") != "development_only_high_rank_extension":
        raise ValueError("high-rank extension manifest is incomplete")
    expected = {
        config["config_id"]: config for config in extension.get("configs", [])
        if config.get("arm") == "spectral"
        and config["memory_screen"]["analytically_feasible_48gb"]
    }
    if not expected:
        report = {
            "status": "complete_no_analytically_feasible_rank",
            "extension_sha256": sha256(extension_path), "probes": [], "eligible_ranks": [],
        }
        atomic_json(output, report)
        return report
    if len(result_paths) != len(expected):
        raise ValueError("probe results do not cover every analytically feasible rank")
    rows, seen = [], set()
    for path in result_paths:
        result_path = path if path.name == "result.json" else path / "result.json"
        result = json.loads(result_path.read_text())
        config_id = result.get("config", {}).get("search_config_id")
        if config_id not in expected or config_id in seen:
            raise ValueError("probe result has unexpected or duplicate configuration")
        seen.add(config_id)
        config = expected[config_id]
        rank = config["rank"]
        updates = result.get("updates", 0)
        final_filter = (result.get("logs") or [{}])[-1].get("filter", {})
        required_updates = config.get("required_probe_updates")
        if not isinstance(required_updates, int) or required_updates < 2 * rank:
            raise ValueError("extension lacks a valid cadence-aware probe budget")
        if (result.get("status") != "complete"
                or result.get("parameter_count") != config["memory_screen"]["parameter_count"]
                or result.get("config", {}).get("arm") != "spectral"
                or result.get("config", {}).get("filter", {}).get("rank") != rank
                or updates != required_updates
                or result.get("peak_cuda_memory_bytes", max_probe_bytes) > max_probe_bytes
                or not final_filter.get("filtering_active", False)
                or final_filter.get("basis_rank", 0) < rank
                or final_filter.get("orthogonality_error", float("inf")) > 1e-3):
            raise ValueError(f"rank {rank} probe did not realize a safe full-rank filter")
        rows.append({
            "rank": rank, "config_id": config_id, "updates": updates,
            "basis_rank": final_filter["basis_rank"],
            "peak_cuda_memory_bytes": result["peak_cuda_memory_bytes"],
            "result_sha256": sha256(result_path),
        })
    rows.sort(key=lambda row: row["rank"])
    report = {
        "status": "probe_audit_complete", "extension_sha256": sha256(extension_path),
        "max_probe_bytes": max_probe_bytes, "probes": rows,
        "eligible_ranks": [row["rank"] for row in rows],
    }
    atomic_json(output, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--extension", type=Path, required=True)
    parser.add_argument("--result", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-probe-bytes", type=int, default=MAX_PROBE_BYTES)
    args = parser.parse_args()
    print(json.dumps(audit_probes(
        args.extension, args.result, args.output, args.max_probe_bytes
    ), sort_keys=True))


if __name__ == "__main__":
    main()
