"""Create a development-only global-rank extension from a selected spectral draw."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import atomic_json, sha256

FEATURE_DIMENSIONS = {"medium": 780, "all": 3555}
L40_BYTES = 48 * 1024**3
EXTENSION_ID_BASE = 1_000_000
EXTENSION_ID_STRIDE = 10_000


def parameter_count(config: dict) -> int:
    """Exact parameter count for the configured ResidualMLP without allocating it."""
    width, depth = int(config["width"]), int(config["depth"])
    try:
        input_dim = FEATURE_DIMENSIONS[config["feature_set"]]
    except KeyError as exc:
        raise ValueError("unknown frozen v5.3 feature set") from exc
    normalization = 2 * width if config["normalization"] == "layer" else 0
    return ((input_dim * width + width)
            + (depth - 1) * (normalization + width * width + width)
            + normalization + width + 1)


def memory_estimate(config: dict, rank: int) -> dict:
    """Conservative screen; a successful near-full-rank GPU probe remains mandatory."""
    parameters = parameter_count(config)
    basis = parameters * rank * 4
    # Four p×r copies conservatively cover retained basis plus update/QR temporaries.
    # Sixteen bytes/parameter cover fp32 weights, gradients and two Adam moments.
    peak = 4 * basis + 16 * parameters + rank * rank * 8 * 4
    return {
        "parameter_count": parameters, "basis_bytes": basis,
        "conservative_peak_bytes": peak,
        "analytically_feasible_48gb": peak <= int(0.85 * L40_BYTES),
        "probe_required": True,
    }


def required_probe_updates(config: dict, rank: int) -> int:
    """Updates needed to realize rank and exercise the active filter.

    The first covariance observation initializes the running mean.  Thereafter
    at most one basis direction is added per covariance update, and covariance
    updates occur every ``filter_update_every`` gradient steps.
    """
    cadence = int(config["filter_update_every"])
    warmup = int(config["filter_warmup"])
    if cadence < 1 or warmup < 0 or rank < 1:
        raise ValueError("invalid rank, filter cadence or filter warmup")
    return max(2 * rank, 1 + rank * cadence, warmup + 1)


def extension_config_id(source_id: int, rank: int) -> int:
    """Stable collision-free ID for one source draw and rank below 10,000."""
    if not 0 <= source_id < 10_000 or not 0 < rank < EXTENSION_ID_STRIDE:
        raise ValueError("source ID or extension rank is outside the supported range")
    return EXTENSION_ID_BASE + source_id * EXTENSION_ID_STRIDE + rank


def create_extension(search_path: Path, selection_path: Path, ranks: list[int],
                     output: Path) -> dict:
    search = json.loads(search_path.read_text())
    selection = json.loads(selection_path.read_text())
    selected = selection.get("selected", {}).get("spectral", [])
    if len(selected) != 1:
        raise ValueError("high-rank extension requires one selected spectral config")
    source_id = selected[0]
    matches = [draw for draw in search.get("configs", [])
               if draw.get("config_id") == source_id]
    spectral_matches = [draw for draw in matches if draw.get("arm") == "spectral"]
    adamw_matches = [draw for draw in matches if draw.get("arm") == "adamw"]
    if len(spectral_matches) != 1 or len(adamw_matches) != 1:
        raise ValueError("selected config lacks one AdamW/spectral pair")
    source, adamw_source = spectral_matches[0], adamw_matches[0]
    for key, value in adamw_source.items():
        if key not in {"arm", "config_id"} and source.get(key) != value:
            raise ValueError("selected AdamW/spectral pair changes non-filter settings")
    if (not ranks or len(ranks) != len(set(ranks))
            or any(not isinstance(rank, int) or rank <= source["rank"] for rank in ranks)):
        raise ValueError("extension ranks must be unique integers above the selected rank")
    configs = []
    for rank in sorted(ranks):
        config_id = extension_config_id(source_id, rank)
        adamw = dict(adamw_source)
        adamw.update({"config_id": config_id, "source_config_id": source_id})
        spectral = dict(source)
        spectral.update({"config_id": config_id, "rank": rank,
                         "source_config_id": source_id,
                         "memory_screen": memory_estimate(source, rank),
                         "required_probe_updates": required_probe_updates(source, rank)})
        configs.extend([adamw, spectral])
    report = {
        "status": "development_only_high_rank_extension",
        "selection_role": "eligible before freeze; official validation remains sealed",
        "primary_target": search.get("primary_target"),
        "primary_metric": search.get("primary_metric"),
        "source_search_sha256": sha256(search_path),
        "source_selection_sha256": sha256(selection_path),
        "source_config_id": source_id,
        "original_rank": source["rank"],
        "ranks": sorted(ranks),
        "memory_screen_rule": (
            "four fp32 p-by-rank basis-sized allocations plus 16 bytes per parameter and "
            "four float64 rank-squared workspaces must use no more than 85% of 48 GiB; "
            "successful near-full-rank GPU probe still required"
        ),
        "configs": configs,
    }
    atomic_json(output, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--search", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--rank", type=int, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(create_extension(
        args.search, args.selection, args.rank, args.output
    ), sort_keys=True))


if __name__ == "__main__":
    main()
