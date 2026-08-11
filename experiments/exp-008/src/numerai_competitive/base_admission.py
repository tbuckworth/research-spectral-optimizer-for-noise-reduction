"""Outcome-independent memory admission for the paired base search."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import atomic_json, sha256
from .high_rank import L40_BYTES, memory_estimate, required_probe_updates


def _paired_draws(search: dict) -> dict[int, dict[str, dict]]:
    pairs: dict[int, dict[str, dict]] = {}
    for draw in search.get("configs", []):
        arm, config_id = draw.get("arm"), draw.get("config_id")
        if arm not in {"adamw", "spectral"} or not isinstance(config_id, int):
            raise ValueError("search contains an invalid arm or configuration ID")
        if arm in pairs.setdefault(config_id, {}):
            raise ValueError("search contains a duplicate arm/configuration ID")
        pairs[config_id][arm] = draw
    if not pairs or any(set(pair) != {"adamw", "spectral"} for pair in pairs.values()):
        raise ValueError("search must contain exactly one AdamW/spectral pair per ID")
    for config_id, pair in pairs.items():
        adamw, spectral = pair["adamw"], pair["spectral"]
        shared = set(adamw) - {"arm"}
        if any(spectral.get(key) != adamw[key] for key in shared):
            raise ValueError(f"configuration {config_id} is not a paired shared draw")
    return pairs


def _probe_evidence(path: Path, draw: dict, estimate: dict) -> dict:
    result = json.loads(path.read_text())
    config = result.get("config", {})
    expected_updates = required_probe_updates(draw, int(draw["rank"]))
    checks = {
        "status_complete": result.get("status") == "complete",
        "arm_matches": config.get("arm") == "spectral",
        "config_id_matches": config.get("search_config_id") == draw["config_id"],
        "rank_matches": config.get("filter", {}).get("rank") == draw["rank"],
        "parameter_count_matches": result.get("parameter_count") == estimate["parameter_count"],
        "updates_exercise_filter": result.get("updates", 0) >= expected_updates,
        "peak_below_device_capacity": (
            isinstance(result.get("peak_cuda_memory_bytes"), int)
            and result["peak_cuda_memory_bytes"] < L40_BYTES
        ),
    }
    return {
        "path": str(path), "sha256": sha256(path), "checks": checks,
        "required_probe_updates": expected_updates,
        "observed_updates": result.get("updates"),
        "observed_peak_cuda_bytes": result.get("peak_cuda_memory_bytes"),
        "passed": all(checks.values()),
    }


def create_admission(search_path: Path, results: Path, output: Path, *, final: bool) -> dict:
    """Admit paired IDs by a frozen static rule or a successful GPU probe."""
    search = json.loads(search_path.read_text())
    pairs = _paired_draws(search)
    rows = []
    for config_id, pair in sorted(pairs.items()):
        draw = pair["spectral"]
        estimate = memory_estimate(draw, int(draw["rank"]))
        static_safe = estimate["analytically_feasible_48gb"]
        probe_path = results / (
            f"stage-outer_1_inner_1-u5000-s0-spectral-c{config_id}/result.json"
        )
        evidence = _probe_evidence(probe_path, draw, estimate) if probe_path.is_file() else None
        admitted = static_safe or bool(evidence and evidence["passed"])
        state = ("static_safe" if static_safe else "empirical_probe_passed" if admitted
                 else "excluded_no_valid_probe" if final else "pending_empirical_probe")
        rows.append({
            "config_id": config_id, "rank": draw["rank"], "state": state,
            "admitted": admitted, "memory_screen": estimate, "probe": evidence,
        })
    pending = [row["config_id"] for row in rows if row["state"] == "pending_empirical_probe"]
    payload = {
        "status": "complete" if not pending else "pending_probe",
        "policy": (
            "admit both arms when the spectral draw uses at most 85% of 48 GiB under the "
            "frozen conservative estimate, or when a provenance-checked run exercises the "
            "full requested rank/filter and records peak CUDA memory below 48 GiB; otherwise "
            "exclude both arms before metric-based selection"
        ),
        "search_sha256": sha256(search_path), "final": final,
        "admitted_config_ids": [row["config_id"] for row in rows if row["admitted"]],
        "excluded_config_ids": [row["config_id"] for row in rows
                                if row["state"] == "excluded_no_valid_probe"],
        "pending_probe_config_ids": pending, "rows": rows,
    }
    atomic_json(output, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--search", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--final", action="store_true")
    args = parser.parse_args()
    print(json.dumps(create_admission(
        args.search, args.results, args.output, final=args.final,
    ), sort_keys=True))


if __name__ == "__main__":
    main()
