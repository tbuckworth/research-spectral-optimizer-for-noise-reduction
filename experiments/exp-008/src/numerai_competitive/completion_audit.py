"""Cross-check every final artifact before declaring the Numerai study complete."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from .data import atomic_json, sha256


def _json(path: Path, status: tuple[str, ...]) -> dict:
    value = json.loads(path.read_text())
    if value.get("status") not in status:
        raise ValueError(f"{path}: unexpected or incomplete status")
    return value


def _verify_named_hashes(directory: Path, hashes: dict[str, str]) -> None:
    for name, expected in hashes.items():
        path = directory / name
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"{path}: missing or hash mismatch")


def audit(results: Path, leaderboard_path: Path, output: Path) -> dict:
    evidence = {}
    outer_selected = {}
    for number in range(1, 4):
        path = results / f"audit-outer_{number}-u100000" / "outer-audit.json"
        value = _json(path, ("audit_complete",))
        if (value.get("split", {}).get("name") != f"outer_{number}"
                or value.get("updates") != 100_000 or value.get("seeds") != [0, 1, 2]
                or value.get("cells") != 6):
            raise ValueError(f"outer_{number} audit has wrong split, updates or seeds")
        if set(value.get("selected", {})) != {"adamw", "spectral"}:
            raise ValueError(f"outer_{number} audit lacks both selected arms")
        outer_selected[str(number)] = value["selected"]
        evidence[f"outer_{number}_audit"] = sha256(path)

    nested_path = results / "nested-outer" / "nested-outer-report.json"
    nested = _json(nested_path, ("complete",))
    if (nested.get("outer_splits") != ["outer_1", "outer_2", "outer_3"]
            or nested.get("expected_seeds") != [0, 1, 2]
            or "spectral_minus_adamw" not in nested):
        raise ValueError("nested-outer report lacks exact folds, seeds or paired inference")
    candidate_path = results / "candidate-plan.json"
    candidate = _json(candidate_path, ("frozen_train_only_selection",))
    evidence.update({"nested_outer": sha256(nested_path), "candidate_plan": sha256(candidate_path)})

    union_path = results / "selection-final-outer-winner-union.json"
    union = _json(union_path, ("outer_winners_audited",))
    expected_by_arm = {
        arm: sorted({outer_selected[str(number)][arm] for number in range(1, 4)})
        for arm in ("adamw", "spectral")
    }
    expected_union = sorted(set(expected_by_arm["adamw"]) | set(expected_by_arm["spectral"]))
    if (union.get("selected", {}).get("adamw") != expected_by_arm["adamw"]
            or union.get("selected", {}).get("spectral") != expected_by_arm["spectral"]
            or union.get("selected", {}).get("paired_union") != expected_union):
        raise ValueError("final candidate union differs from audited outer winners")
    evidence["outer_winner_union"] = sha256(union_path)

    refit_path = results / "audit-final-refits-u100000" / "refit-audit.json"
    refit = _json(refit_path, ("audit_complete",))
    if (refit.get("cells") != 6 or refit.get("updates") != 100_000
            or refit.get("seeds") != [0, 1, 2]
            or set(refit.get("selected", {})) != {"adamw", "spectral"}):
        raise ValueError("final refit audit is not the exact two-arm, three-seed procedure")
    final_selection_path = results / "selection-final-top1.json"
    final_selection = json.loads(final_selection_path.read_text())
    if any(final_selection.get("selected", {}).get(arm) != [refit["selected"][arm]]
           for arm in ("adamw", "spectral")):
        raise ValueError("final refit audit differs from final canonical-fold selection")
    if any(refit["selected"][arm] not in expected_union for arm in ("adamw", "spectral")):
        raise ValueError("final selected config was not an audited outer winner")
    evidence["final_refit_audit"] = sha256(refit_path)
    evidence["final_selection"] = sha256(final_selection_path)

    freeze_path = results / "freeze.json"
    freeze = _json(freeze_path, ("frozen",))
    if (not re.fullmatch(r"[0-9a-f]{40}", freeze.get("code_commit", ""))
            or freeze.get("primary_target") != "target_cyrusd_20"
            or freeze.get("candidate_plan_sha256") != sha256(candidate_path)
            or freeze.get("candidate_transform") != candidate.get("selected")):
        raise ValueError("freeze provenance or candidate transformation is inconsistent")
    for arm in ("adamw", "spectral"):
        selected = freeze["selected"][arm]
        if (selected.get("config_id") != refit["selected"][arm]
                or selected.get("seeds") != [0, 1, 2]
                or selected.get("updates") != 100_000):
            raise ValueError(f"{arm} freeze differs from final refit audit")
        paths = [results / f"final-refit-u100000-s{seed}-{arm}-c{selected['config_id']}"
                 / "model.pt" for seed in range(3)]
        if [sha256(path) for path in paths] != selected.get("model_sha256"):
            raise ValueError(f"{arm} frozen model hashes differ from final refits")
    evidence["freeze"] = sha256(freeze_path)

    validation_dir = results / "official-validation"
    validation_marker = _json(validation_dir / "evaluation-complete.json", ("complete",))
    _verify_named_hashes(validation_dir, validation_marker.get("artifacts", {}))
    validation_path = validation_dir / "official-validation-report.json"
    validation = _json(validation_path, ("complete",))
    if (validation.get("target") != "target_cyrusd_20"
            or validation.get("freeze_manifest_sha256") != sha256(freeze_path)
            or "spectral_minus_adamw" not in validation
            or "candidate_minus_ender20" not in validation):
        raise ValueError("sealed validation report has inconsistent provenance or endpoints")
    evidence["official_validation"] = sha256(validation_path)

    bundle = results / "live-bundle"
    fixture = bundle / "live-fixture"
    download_path = fixture / "download-complete.json"
    download = _json(download_path, ("complete",))
    if download.get("freeze_manifest_sha256") != sha256(freeze_path):
        raise ValueError("live fixture was not downloaded for the final freeze")
    for name, metadata in download.get("artifacts", {}).items():
        path = fixture / name
        if sha256(path) != metadata.get("sha256") or path.stat().st_size != metadata.get("bytes"):
            raise ValueError(f"{path}: live fixture hash or size mismatch")
    runtime_path = bundle / "runtime-audit.json"
    runtime = _json(runtime_path, ("pass",))
    predictor, predictions = bundle / "predictor.pkl", bundle / "live_predictions.csv"
    if (runtime.get("artifact_sha256") != sha256(predictor)
            or runtime.get("prediction_sha256") != sha256(predictions)
            or runtime.get("max_bytes") != 4_000_000_000
            or runtime.get("max_seconds") != 600):
        raise ValueError("live runtime audit differs from the candidate bundle or limits")
    official_path = bundle / "official-container" / "official-container-audit.json"
    official = _json(official_path, ("pass",))
    if (official.get("expected_sha256") != sha256(predictions)
            or official.get("cpu_limit") != 1
            or official.get("memory_limit_bytes") != 4_000_000_000
            or official.get("max_seconds") != 600
            or not re.fullmatch(r"[0-9a-f]{40}", official.get("runner_commit", ""))):
        raise ValueError("official container audit differs from live predictions or limits")
    evidence.update({"live_download": sha256(download_path), "live_runtime": sha256(runtime_path),
                     "official_container": sha256(official_path)})

    leaderboard = _json(leaderboard_path, ("complete",))
    if not leaderboard.get("summary", {}).get("rows"):
        raise ValueError("public leaderboard snapshot is empty")
    report_dir = results / "final-report"
    report_manifest_path = report_dir / "report-manifest.json"
    report_manifest = _json(report_manifest_path, ("complete",))
    if report_manifest.get("comparability") != "historical-direct_live-context-only":
        raise ValueError("final report does not preserve the live comparability boundary")
    expected_report_inputs = {sha256(nested_path), sha256(validation_path),
                              sha256(leaderboard_path)}
    if set(report_manifest.get("inputs", {}).values()) != expected_report_inputs:
        raise ValueError("final report was not built from the audited result inputs")
    _verify_named_hashes(report_dir, report_manifest.get("artifacts", {}))
    evidence.update({"leaderboard": sha256(leaderboard_path),
                     "final_report": sha256(report_manifest_path)})

    result = {
        "status": "audit_complete", "primary_target": "target_cyrusd_20",
        "outer_selected": outer_selected, "final_selected": refit["selected"],
        "code_commit": freeze["code_commit"], "evidence_sha256": evidence,
    }
    atomic_json(output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--leaderboard", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.results, args.leaderboard, args.output), sort_keys=True))


if __name__ == "__main__":
    main()
