"""Cross-check every final artifact before declaring the Numerai study complete."""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

from .code_snapshot import verify_snapshot
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


def _verify_score_hashes(hashes: dict[str, str], expected_paths: list[Path]) -> None:
    expected = {path.resolve() for path in expected_paths}
    actual = {Path(path).resolve() for path in hashes}
    if actual != expected:
        raise ValueError("selection references unexpected development score files")
    for raw_path, expected_hash in hashes.items():
        path = Path(raw_path)
        if not path.is_file() or sha256(path) != expected_hash:
            raise ValueError(f"{path}: selected score file is missing or changed")


def _score_arm_ids(path: Path) -> dict[str, set[int]]:
    values = {"adamw": set(), "spectral": set()}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            arm = row.get("arm")
            if arm not in values:
                raise ValueError(f"{path}: unexpected score arm")
            values[arm].add(int(row["config_id"]))
    return values


def audit(results: Path, leaderboard_path: Path, output: Path,
          procedure_code_root: Path | None = None,
          production_code_root: Path | None = None) -> dict:
    evidence = {}
    procedure_root = (results.parent if procedure_code_root is None
                      else procedure_code_root.resolve())
    production_root = (results.parent if production_code_root is None
                       else production_code_root.resolve())
    base_search_path = procedure_root / "configs" / "search-v1.json"
    admission_path = results / "base-search-memory-admission.json"
    admission = _json(admission_path, ("complete",))
    admission_rows = admission.get("rows", [])
    admitted_ids = set(admission.get("admitted_config_ids", []))
    excluded_ids = set(admission.get("excluded_config_ids", []))
    if (admission.get("search_sha256") != sha256(base_search_path)
            or admission.get("pending_probe_config_ids") != []
            or admitted_ids & excluded_ids
            or admitted_ids | excluded_ids != set(range(40))
            or {row.get("config_id") for row in admission_rows} != set(range(40))):
        raise ValueError("base-search memory admission is incomplete or inconsistent")
    for row in admission_rows:
        if row["config_id"] in admitted_ids:
            if row.get("state") not in {"static_safe", "empirical_probe_passed"}:
                raise ValueError("admitted base-search row has invalid evidence state")
            if row.get("state") == "empirical_probe_passed":
                probe = row.get("probe", {})
                probe_path = Path(probe.get("path", ""))
                if (not probe.get("passed") or not probe.get("checks")
                        or not all(probe["checks"].values()) or not probe_path.is_file()
                        or probe.get("sha256") != sha256(probe_path)):
                    raise ValueError("empirically admitted row lacks valid frozen probe evidence")
        elif row.get("state") != "excluded_no_valid_probe":
            raise ValueError("excluded base-search row has invalid evidence state")
    evidence["base_search_memory_admission"] = sha256(admission_path)
    source_path = results / "selection-high-rank-source-r2048.json"
    extension_path = results / "search-v1-high-rank-extension.json"
    probe_path = results / "audit-high-rank-probes.json"
    search_path = results / "search-v1-high-rank.json"
    source = _json(source_path, ("development_only_high_rank_source_selection",))
    extension = _json(extension_path, ("development_only_high_rank_extension",))
    probes = _json(probe_path, ("probe_audit_complete",))
    search = _json(search_path, ("development_only_augmented_search",))
    spectral_source = source.get("selected", {}).get("spectral", [])
    source_scores = [
        results / f"summary-outer_1_inner_{inner}-u20000-s0" / "scores.csv"
        for inner in (1, 2)
    ]
    _verify_score_hashes(source.get("score_sha256", {}), source_scores)
    eligible_ranks = probes.get("eligible_ranks", [])
    extension_ids = sorted(
        config["config_id"] for config in extension.get("configs", [])
        if config.get("arm") == "spectral" and config.get("rank") in eligible_ranks
    )
    if (source.get("requested_rank") != 2048 or len(spectral_source) != 1
            or extension.get("source_config_id") != spectral_source[0]
            or extension.get("source_search_sha256") != sha256(base_search_path)
            or extension.get("source_selection_sha256") != sha256(source_path)
            or probes.get("extension_sha256") != sha256(extension_path)
            or not eligible_ranks or search.get("high_rank_config_ids") != extension_ids
            or search.get("base_search_sha256") != sha256(base_search_path)
            or search.get("extension_sha256") != sha256(extension_path)
            or search.get("probe_audit_sha256") != sha256(probe_path)):
        raise ValueError("high-rank amendment provenance or eligible ranks are inconsistent")
    evidence.update({"high_rank_source": sha256(source_path),
                     "high_rank_extension": sha256(extension_path),
                     "high_rank_probe_audit": sha256(probe_path),
                     "augmented_search": sha256(search_path)})
    outer_selected = {}
    for number in range(1, 4):
        inner_count = number + 1
        f0_score = results / f"summary-outer_{number}_inner_1-u5000-s0" / "scores.csv"
        if _score_arm_ids(f0_score) != {
            "adamw": admitted_ids, "spectral": admitted_ids,
        }:
            raise ValueError(f"outer_{number} F0 coverage differs from memory admission")
        f0_path = results / f"selection-outer_{number}-f0-top12.json"
        f0 = json.loads(f0_path.read_text())
        _verify_score_hashes(f0.get("score_sha256", {}), [f0_score])
        f0_selected = f0.get("selected", {})
        if (f0.get("top") != 12
                or any(len(f0_selected.get(arm, [])) != 12
                       for arm in ("adamw", "spectral"))
                or not 12 <= len(f0_selected.get("paired_union", [])) <= 24
                or set(f0_selected.get("paired_union", []))
                != set(f0_selected["adamw"]) | set(f0_selected["spectral"])
                or not set(f0_selected["paired_union"]) <= admitted_ids):
            raise ValueError(f"outer_{number} F0 selection is inconsistent")
        f1_scores = [
            results / f"summary-outer_{number}_inner_{inner}-u20000-s0" / "scores.csv"
            for inner in range(1, inner_count + 1)
        ]
        f1_path = results / f"selection-outer_{number}-f1-top4.json"
        f1 = _json(f1_path, ("f1_selection_augmented_with_gpu_audited_high_ranks",))
        _verify_score_hashes(f1.get("score_sha256", {}), [f0_score, *f1_scores])
        ordinary = set(f1.get("selected", {}).get("paired_union", []))
        for score_path in f1_scores:
            if _score_arm_ids(score_path) != {
                "adamw": set(f0_selected["paired_union"]),
                "spectral": set(f0_selected["paired_union"]),
            }:
                raise ValueError(f"{score_path}: F1 coverage differs from F0 promotion")
        fidelity = f1.get("fidelity_selections", {})
        if (f1.get("status") != "f1_selection_augmented_with_gpu_audited_high_ranks"
                or f1.get("top_per_arm_per_fidelity") != 4
                or set(fidelity) != {"5000", "20000"}
                or any(len(fidelity[budget].get(arm, [])) != 4
                       for budget in fidelity for arm in ("adamw", "spectral"))
                or any(set(fidelity[budget].get("paired_union", []))
                       != set(fidelity[budget]["adamw"]) | set(fidelity[budget]["spectral"])
                       for budget in fidelity)
                or any(set(f1.get("selected", {}).get(arm, []))
                       != set(fidelity["5000"][arm]) | set(fidelity["20000"][arm])
                       for arm in ("adamw", "spectral"))
                or f1.get("augmented_search_sha256") != sha256(search_path)
                or f1.get("selected", {}).get("high_rank_spectral") != extension_ids
                or not 4 <= len(ordinary) <= 16):
            raise ValueError(f"outer_{number} F1 selection omits audited candidates")
        f2_scores = [
            results / f"summary-f2-outer_{number}_inner_{inner}-u{budget}-s{seed}"
            / "scores.csv"
            for inner in range(1, inner_count + 1)
            for budget in (5000, 20000, 100000) for seed in range(3)
        ]
        f2_path = results / f"selection-outer_{number}-f2-budget-top1.json"
        f2 = json.loads(f2_path.read_text())
        _verify_score_hashes(f2.get("score_sha256", {}), f2_scores)
        for score_path in f2_scores:
            coverage = _score_arm_ids(score_path)
            is_long_budget = "-u100000-" in score_path.parent.name
            expected_spectral = ordinary | set(extension_ids) if is_long_budget else ordinary
            if coverage["adamw"] != ordinary or coverage["spectral"] != expected_spectral:
                raise ValueError(f"{score_path}: F2 arm/config coverage is inconsistent")
        if (f2.get("top") != 1 or f2.get("allow_asymmetric") is not True
                or any(len(f2.get("selected", {}).get(arm, [])) != 1
                       for arm in ("adamw", "spectral"))):
            raise ValueError(f"outer_{number} F2 selection is not exact asymmetric top-1")
        if any(
            len(f2.get("selected_updates", {}).get(arm, [])) != 1
            or f2["selected_updates"][arm][0] not in {5000, 20000, 100000}
            for arm in ("adamw", "spectral")
        ):
            raise ValueError(f"outer_{number} F2 selection lacks exact update budgets")
        evidence[f"outer_{number}_f1_selection"] = sha256(f1_path)
        evidence[f"outer_{number}_f0_selection"] = sha256(f0_path)
        evidence[f"outer_{number}_f2_selection"] = sha256(f2_path)
        path = results / f"audit-outer_{number}-budgeted" / "outer-audit.json"
        value = _json(path, ("audit_complete",))
        if (value.get("split", {}).get("name") != f"outer_{number}"
                or value.get("updates") != {
                    arm: f2["selected_updates"][arm][0] for arm in ("adamw", "spectral")
                } or value.get("seeds") != [0, 1, 2]
                or value.get("cells") != 6):
            raise ValueError(f"outer_{number} audit has wrong split, updates or seeds")
        if set(value.get("selected", {})) != {"adamw", "spectral"}:
            raise ValueError(f"outer_{number} audit lacks both selected arms")
        if any(value["selected"][arm] != f2["selected"][arm][0]
               for arm in ("adamw", "spectral")):
            raise ValueError(f"outer_{number} audit differs from its F2 selection")
        outer_selected[str(number)] = {
            arm: {
                "config_id": value["selected"][arm],
                "updates": value["updates"][arm],
            }
            for arm in ("adamw", "spectral")
        }
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
        arm: sorted({
            outer_selected[str(number)][arm]["config_id"] for number in range(1, 4)
        })
        for arm in ("adamw", "spectral")
    }
    expected_union = sorted(set(expected_by_arm["adamw"]) | set(expected_by_arm["spectral"]))
    if (union.get("selected", {}).get("adamw") != expected_by_arm["adamw"]
            or union.get("selected", {}).get("spectral") != expected_by_arm["spectral"]
            or union.get("selected", {}).get("paired_union") != expected_union):
        raise ValueError("final candidate union differs from audited outer winners")
    expected_budgeted = sorted({
        (
            outer_selected[str(number)][arm]["config_id"],
            outer_selected[str(number)][arm]["updates"],
        )
        for number in range(1, 4) for arm in ("adamw", "spectral")
    })
    if union.get("budgeted_candidates") != [
        {"config_id": config_id, "updates": updates}
        for config_id, updates in expected_budgeted
    ]:
        raise ValueError("final budgeted candidate union differs from outer winners")
    evidence["outer_winner_union"] = sha256(union_path)

    refit_path = results / "audit-final-refits-budgeted" / "refit-audit.json"
    refit = _json(refit_path, ("audit_complete",))
    if (refit.get("cells") != 6
            or set(refit.get("updates", {})) != {"adamw", "spectral"}
            or any(value not in {5000, 20000, 100000}
                   for value in refit.get("updates", {}).values())
            or refit.get("seeds") != [0, 1, 2]
            or set(refit.get("selected", {})) != {"adamw", "spectral"}):
        raise ValueError("final refit audit is not the exact two-arm, three-seed procedure")
    final_selection_path = results / "selection-final-top1.json"
    final_selection = json.loads(final_selection_path.read_text())
    if any(final_selection.get("selected", {}).get(arm) != [refit["selected"][arm]]
           for arm in ("adamw", "spectral")):
        raise ValueError("final refit audit differs from final canonical-fold selection")
    if any(final_selection.get("selected_updates", {}).get(arm)
           != [refit["updates"][arm]] for arm in ("adamw", "spectral")):
        raise ValueError("final refit budgets differ from final canonical-fold selection")
    if any(refit["selected"][arm] not in expected_union for arm in ("adamw", "spectral")):
        raise ValueError("final selected config was not an audited outer winner")
    if any(
        (refit["selected"][arm], refit["updates"][arm]) not in expected_budgeted
        for arm in ("adamw", "spectral")
    ):
        raise ValueError("final selected config/budget was not an audited outer winner")
    evidence["final_refit_audit"] = sha256(refit_path)
    evidence["final_selection"] = sha256(final_selection_path)

    freeze_path = results / "freeze.json"
    freeze = _json(freeze_path, ("frozen",))
    code_snapshot_path = procedure_root / "code-snapshot.json"
    protocol_path = procedure_root / "fidelity-protocol.md"
    if (not re.fullmatch(r"[0-9a-f]{40}", freeze.get("code_commit", ""))
            or freeze.get("primary_target") != "target"
            or freeze.get("primary_benchmark") != "v53_lgbm_ender60"
            or freeze.get("code_snapshot_sha256") != sha256(code_snapshot_path)
            or freeze.get("search_sha256") != sha256(search_path)
            or freeze.get("fidelity_protocol_sha256") != sha256(protocol_path)
            or freeze.get("candidate_plan_sha256") != sha256(candidate_path)
            or freeze.get("candidate_transform") != candidate.get("selected")):
        raise ValueError("freeze provenance or candidate transformation is inconsistent")
    verify_snapshot(procedure_root, code_snapshot_path, freeze["code_commit"])
    for arm in ("adamw", "spectral"):
        selected = freeze["selected"][arm]
        if (selected.get("config_id") != refit["selected"][arm]
                or selected.get("seeds") != [0, 1, 2]
                or selected.get("updates") != refit["updates"][arm]):
            raise ValueError(f"{arm} freeze differs from final refit audit")
        paths = [results / (
            f"final-refit-u{selected['updates']}-s{seed}-{arm}-c{selected['config_id']}"
        )
                 / "model.pt" for seed in range(3)]
        if [sha256(path) for path in paths] != selected.get("model_sha256"):
            raise ValueError(f"{arm} frozen model hashes differ from final refits")
    evidence.update({"freeze": sha256(freeze_path), "code_snapshot": sha256(code_snapshot_path)})

    validation_dir = results / "official-validation"
    validation_marker = _json(validation_dir / "evaluation-complete.json", ("complete",))
    _verify_named_hashes(validation_dir, validation_marker.get("artifacts", {}))
    validation_path = validation_dir / "official-validation-report.json"
    validation = _json(validation_path, ("complete",))
    if (validation.get("target") != "target"
            or validation.get("benchmark") != "v53_lgbm_ender60"
            or validation.get("target_alias_audit") != {
                "target_equals_target_ender_60": True,
                "live_corr20v2_target": "target_cyrus_20",
                "live_target_released_in_v5_3": False,
            }
            or validation.get("freeze_manifest_sha256") != sha256(freeze_path)
            or "spectral_minus_adamw" not in validation
            or "candidate_minus_ender60" not in validation
            or validation.get("candidate_transform", {}).get("benchmark")
            != "v53_lgbm_ender60"):
        raise ValueError("sealed validation report has inconsistent provenance or endpoints")
    evidence["official_validation"] = sha256(validation_path)

    production_audit_path = results / "production-refit-audit.json"
    production = _json(production_audit_path, ("audit_complete",))
    production_snapshot_path = production_root / "production-code-snapshot.json"
    candidate_arm = freeze["candidate_transform"]["arm"]
    selected_candidate = freeze["selected"][candidate_arm]
    production_manifest = production.get("production_manifest", {})
    if (production.get("purpose") != "unstaked_forward_live_candidate"
            or production.get("arm") != candidate_arm
            or production.get("config_id") != selected_candidate["config_id"]
            or production.get("updates") != selected_candidate["updates"]
            or production.get("seeds") != selected_candidate["seeds"]
            or production.get("freeze_manifest_sha256") != sha256(freeze_path)
            or not re.fullmatch(r"[0-9a-f]{40}", production.get("production_code_commit", ""))
            or production.get("procedure_code_commit") != freeze["code_commit"]
            or production.get("production_code_snapshot_sha256")
            != sha256(production_snapshot_path)
            or production.get("sealed_evaluation_sha256")
            != sha256(validation_dir / "evaluation-complete.json")
            or production_manifest.get("split") != "production_train"
            or production_manifest.get("training_data_sha256")
            != production.get("training_data_sha256")
            or production_manifest.get("resolved_validation_rows", 0) <= 0):
        raise ValueError("production-live refit audit differs from freeze/evaluation")
    verify_snapshot(
        production_root, production_snapshot_path, production["production_code_commit"],
    )
    production_paths = [results / (
        f"production-refit-s{seed}-{candidate_arm}-c{selected_candidate['config_id']}"
    ) / "model.pt" for seed in selected_candidate["seeds"]]
    if [sha256(path) for path in production_paths] != production.get("model_sha256"):
        raise ValueError("production-live model hashes differ from production audit")
    evidence["production_refits"] = sha256(production_audit_path)
    evidence["production_code_snapshot"] = sha256(production_snapshot_path)

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
    official_predictions = list(official_path.parent.glob("live_predictions-*.csv"))
    if (official.get("expected_sha256") != sha256(predictions)
            or len(official_predictions) != 1
            or official.get("official_sha256") != sha256(official_predictions[0])
            or official.get("cpu_limit") != 1
            or official.get("memory_limit_bytes") != 4_000_000_000
            or official.get("max_seconds") != 600
            or official.get("elapsed_seconds", 600) >= official.get("max_seconds", 600)
            or official.get("max_abs_difference", float("inf"))
            > official.get("allowed_max_abs_difference", -1)
            or not re.fullmatch(r"[0-9a-f]{40}", official.get("runner_commit", ""))
            or not re.fullmatch(r"sha256:[0-9a-f]{64}", official.get("image_id", ""))):
        raise ValueError("official container audit differs from live predictions or limits")
    evidence.update({"live_download": sha256(download_path), "live_runtime": sha256(runtime_path),
                     "official_container": sha256(official_path)})

    leaderboard = _json(leaderboard_path, ("complete",))
    leaderboard_raw = leaderboard_path.parent / "leaderboard-raw.json"
    if (not leaderboard.get("summary", {}).get("rows")
            or not isinstance(leaderboard.get("round"), int)
            or not leaderboard_raw.is_file()
            or leaderboard.get("raw_sha256") != sha256(leaderboard_raw)):
        raise ValueError("public leaderboard snapshot is empty or raw response is inconsistent")
    report_dir = results / "final-report"
    report_manifest_path = report_dir / "report-manifest.json"
    report_manifest = _json(report_manifest_path, ("complete",))
    if report_manifest.get("comparability") != "historical-direct_live-context-only":
        raise ValueError("final report does not preserve the live comparability boundary")
    expected_report_inputs = {sha256(nested_path), sha256(validation_path),
                              sha256(leaderboard_path), sha256(freeze_path),
                              sha256(search_path), sha256(admission_path)}
    if set(report_manifest.get("inputs", {}).values()) != expected_report_inputs:
        raise ValueError("final report was not built from the audited result inputs")
    if report_manifest.get("selected_configs", {}).keys() != {"adamw", "spectral"}:
        raise ValueError("final report does not expose both selected configurations")
    _verify_named_hashes(report_dir, report_manifest.get("artifacts", {}))
    evidence.update({"leaderboard": sha256(leaderboard_path),
                     "leaderboard_raw": sha256(leaderboard_raw),
                     "final_report": sha256(report_manifest_path)})

    result = {
        "status": "audit_complete", "primary_target": "target",
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
    parser.add_argument("--procedure-code-root", type=Path)
    parser.add_argument("--production-code-root", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(
        args.results, args.leaderboard, args.output,
        args.procedure_code_root, args.production_code_root,
    ), sort_keys=True))


if __name__ == "__main__":
    main()
