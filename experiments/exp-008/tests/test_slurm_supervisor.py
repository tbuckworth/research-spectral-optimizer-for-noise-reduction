import json
import os
import subprocess
from pathlib import Path

SCRIPT = Path(__file__).parents[1] / "slurm" / "supervise-resumable-stage.sh"
OUTER_SUBMIT = Path(__file__).parents[1] / "slurm" / "submit-outer-eval.sh"
OUTER_AUDIT = Path(__file__).parents[1] / "slurm" / "audit-outer-when-ready.sh"
SUBMIT_STAGE = Path(__file__).parents[1] / "slurm" / "submit-stage.sh"
SUBMIT_SELECTED = Path(__file__).parents[1] / "slurm" / "submit-selected.sh"
SUBMIT_BUDGETED = Path(__file__).parents[1] / "slurm" / "submit-budgeted-selected.sh"
SUBMIT_HIGH_RANK = Path(__file__).parents[1] / "slurm" / "submit-high-rank-spectral.sh"
SUBMIT_SUCCESSIVE = Path(__file__).parents[1] / "slurm" / "submit-successive-plan.sh"
SUBMIT_FINALISTS = Path(__file__).parents[1] / "slurm" / "submit-successive-finalists.sh"
RESUME_STAGE = Path(__file__).parents[1] / "slurm" / "resume-checkpointed-stage.sh"
LAUNCH_OUTER = Path(__file__).parents[1] / "slurm" / "launch-outer.sh"
CONTINUE_NESTED = Path(__file__).parents[1] / "slurm" / "continue-nested-pipeline.sh"
SUBMIT_REFIT = Path(__file__).parents[1] / "slurm" / "submit-refit.sh"
BUILD_OOF = Path(__file__).parents[1] / "slurm" / "build-oof-candidate.sh"
SUBMIT_SEALED = Path(__file__).parents[1] / "slurm" / "submit-sealed-evaluation.sh"
SUBMIT_LIVE = Path(__file__).parents[1] / "slurm" / "submit-live-bundle.sh"
MAIN_TARGET_SMOKE = Path(__file__).parents[1] / "slurm" / "run-main-target-smoke.sbatch"
PROMOTERS = [
    Path(__file__).parents[1] / "slurm" / name
    for name in (
        "promote-f0-when-ready.sh",
        "promote-f1-when-ready.sh",
        "promote-f2-when-ready.sh",
    )
]


def test_f2_uses_separate_summary_namespace_from_f0_and_f1():
    f1_promoter = PROMOTERS[1].read_text()
    scout_promoter = (
        Path(__file__).parents[1] / "slurm" / "promote-successive-scout-when-ready.sh"
    ).read_text()
    assert "NUMERAI_SUMMARY_PREFIX=f2a-" in f1_promoter
    assert "NUMERAI_SUMMARY_PREFIX=f2b-" in scout_promoter


def test_f1_promotion_preserves_5k_and_20k_fidelity_winners():
    script = PROMOTERS[1].read_text()
    assert "numerai_competitive.select_successive_halving" in script
    assert '--score-group "$F0_SUMMARY" --score-group "${F1_SUMMARIES[@]}"' in script
    assert "--confirmation-top 2 --long-scout-top 1" in script


def test_f0_promotion_uses_exact_resumable_f1_supervision_and_promoter():
    script = PROMOTERS[0].read_text()
    assert "supervise-resumable-stage.sh' '$FOLD_MANIFEST' --selection '$SELECTION'" in script
    assert "submission-${OUTER_SPLIT}-f1-${SPLIT}-u20000-s0.tsv" in script
    assert "promote-f1-when-ready.sh' '$LAST_JOB' '$OUTER_SPLIT'" in script
    assert "monitor-stage.sh" not in script
    assert '-v target_split="$SPLIT"' in script
    assert "$2 == target_split" in script


def test_f1_fold_manifest_awk_expression_is_portable(tmp_path):
    manifest = tmp_path / "manifest.tsv"
    manifest.write_text(
        "1\touter_1_inner_1\t20000\t0\tadamw\t1\n"
        "2\touter_1_inner_2\t20000\t0\tadamw\t1\n"
    )
    completed = subprocess.run(
        ["awk", "-F", "\t", "-v", "target_split=outer_1_inner_1",
         "$2 == target_split", manifest],
        check=True, capture_output=True, text=True,
    )
    assert completed.stdout == "1\touter_1_inner_1\t20000\t0\tadamw\t1\n"


def test_main_target_smoke_gates_matching_ender60_benchmark():
    script = MAIN_TARGET_SMOKE.read_text()
    assert 'result["config"]["target"] == "target"' in script
    assert 'result["config"]["benchmark"] == "v53_lgbm_ender60"' in script
    assert 'result["split"]["train_eras"][-1] == "0140"' in script


def _executable(path: Path, body: str) -> None:
    path.write_text("#!/bin/bash\nset -euo pipefail\n" + body)
    path.chmod(0o755)


def _project(tmp_path: Path, *, complete: bool) -> tuple[Path, Path, dict[str, str]]:
    project = tmp_path / "project"
    (project / "results").mkdir(parents=True)
    (project / "configs").mkdir()
    (project / "slurm").mkdir()
    manifest = project / "results" / "submission.tsv"
    manifest.write_text(
        "1\tfold\t100\t0\tadamw\t7\n2\tfold\t100\t0\tspectral\t7\n"
    )
    for arm in ("adamw", "spectral"):
        result_dir = project / "results" / f"stage-fold-u100-s0-{arm}-c7"
        result_dir.mkdir()
        if complete:
            (result_dir / "result.json").write_text("{}")
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    _executable(fake_bin / "squeue", "exit 0\n")
    _executable(project / "uv", 'printf "%s\\n" "$*" >> "$NUMERAI_PROJECT/uv-calls"\n')
    env = os.environ | {
        "NUMERAI_PROJECT": str(project),
        "NUMERAI_QUEUE_USER": "test-user",
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
    }
    return project, manifest, env


def test_supervisor_summarizes_only_after_complete_manifest(tmp_path):
    project, manifest, env = _project(tmp_path, complete=True)
    subprocess.run(["bash", SCRIPT, manifest], check=True, env=env)
    assert "stage complete and all cells summarized" in manifest.with_name(
        "submission-supervisor.log"
    ).read_text()
    assert "numerai_competitive.summarize" in (project / "uv-calls").read_text()


def test_supervisor_uses_explicit_augmented_search(tmp_path):
    project, manifest, env = _project(tmp_path, complete=True)
    search = project / "configs" / "augmented.json"
    search.write_text("{}")
    env["NUMERAI_SEARCH_CONFIG"] = str(search)
    subprocess.run(["bash", SCRIPT, manifest], check=True, env=env)
    assert f"--search {search}" in (project / "uv-calls").read_text()


def test_supervisor_runs_from_project_directory(tmp_path):
    project, manifest, env = _project(tmp_path, complete=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    subprocess.run(["bash", SCRIPT, manifest], check=True, env=env, cwd=outside)
    assert (project / "uv-calls").exists()


def test_supervisor_uses_cell_specific_expected_pairs_for_mixed_budgets(tmp_path):
    project, manifest, env = _project(tmp_path, complete=True)
    second = project / "results" / "stage-fold-u200-s0-adamw-c8"
    second.mkdir()
    (second / "result.json").write_text("{}")
    third = project / "results" / "stage-fold-u200-s0-spectral-c8"
    third.mkdir()
    (third / "result.json").write_text("{}")
    manifest.write_text(
        "1\tfold\t100\t0\tadamw\t7\n"
        "2\tfold\t100\t0\tspectral\t7\n"
        "3\tfold\t200\t0\tadamw\t8\n"
        "4\tfold\t200\t0\tspectral\t8\n"
    )
    subprocess.run(["bash", SCRIPT, manifest], check=True, env=env)
    calls = (project / "uv-calls").read_text().splitlines()
    summaries = [line for line in calls if "numerai_competitive.summarize" in line]
    assert len(summaries) == 2
    assert "adamw:7" in summaries[0] and "adamw:8" not in summaries[0]
    assert "adamw:8" in summaries[1] and "adamw:7" not in summaries[1]


def test_supervisor_resubmits_exact_checkpoint_before_summary(tmp_path):
    project, manifest, env = _project(tmp_path, complete=False)
    result_dir = project / "results" / "stage-fold-u100-s0-adamw-c7"
    (result_dir / "checkpoint.pt").write_text("checkpoint")
    (result_dir / "checkpoint-status.json").write_text("{}")
    spectral_dir = project / "results" / "stage-fold-u100-s0-spectral-c7"
    (spectral_dir / "checkpoint.pt").write_text("checkpoint")
    (spectral_dir / "checkpoint-status.json").write_text("{}")
    _executable(
        project / "slurm" / "resume-checkpointed-stage.sh",
        'touch "$NUMERAI_PROJECT/results/stage-fold-u100-s0-adamw-c7/result.json"\n'
        'touch "$NUMERAI_PROJECT/results/stage-fold-u100-s0-spectral-c7/result.json"\n'
        'printf "2\\tfold\\t100\\t0\\tadamw\\t7\\n3\\tfold\\t100\\t0\\tspectral\\t7\\n"\n',
    )
    subprocess.run(["bash", SCRIPT, manifest], check=True, env=env)
    retry = manifest.with_name("submission.retry-1.tsv")
    assert retry.read_text() == (
        "2\tfold\t100\t0\tadamw\t7\n3\tfold\t100\t0\tspectral\t7\n"
    )
    log = manifest.with_name("submission-supervisor.log").read_text()
    assert "resubmitted 2 checkpointed tasks" in log
    assert "stage complete and all cells summarized (retries=1)" in log


def test_checkpoint_retry_preserves_high_rank_probe_runner(tmp_path):
    project = tmp_path / "project"
    result = project / "results" / "stage-outer_1_inner_1-u1024-s0-spectral-c1000512"
    result.mkdir(parents=True)
    (project / "slurm").mkdir()
    (project / "slurm" / "run-high-rank-probe.sbatch").write_text("")
    (result / "checkpoint.pt").write_text("checkpoint")
    (result / "checkpoint-status.json").write_text("{}")
    manifest = project / "manifest.tsv"
    manifest.write_text("1\touter_1_inner_1\t1024\t0\tspectral\t1000512\n")
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    calls = tmp_path / "sbatch-calls"
    _executable(fake_bin / "sbatch", f'printf "%s\\n" "$*" >> "{calls}"\necho 9001\n')
    env = os.environ | {
        "NUMERAI_PROJECT": str(project), "NUMERAI_PROBE_MODE": "1",
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
    }
    subprocess.run(["bash", RESUME_STAGE, manifest, "7"], check=True, env=env)
    assert str(project / "slurm" / "run-high-rank-probe.sbatch") in calls.read_text()


def test_stage_retry_restarts_failed_task_without_checkpoint(tmp_path):
    project = tmp_path / "project"
    (project / "slurm").mkdir(parents=True)
    (project / "slurm" / "run-one.sbatch").write_text("")
    manifest = project / "manifest.tsv"
    manifest.write_text("1\touter_1_inner_1\t5000\t0\tspectral\t14\n")
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    calls = tmp_path / "sbatch-calls"
    _executable(fake_bin / "sbatch", f'printf "%s\\n" "$*" >> "{calls}"\necho 9002\n')
    env = os.environ | {
        "NUMERAI_PROJECT": str(project),
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
    }
    completed = subprocess.run(
        ["bash", RESUME_STAGE, manifest, "7"], check=True, env=env,
        capture_output=True, text=True,
    )
    assert completed.stdout == "9002\touter_1_inner_1\t5000\t0\tspectral\t14\n"
    assert "--dependency=afterok:7" in calls.read_text()


def test_stage_retry_rejects_partial_checkpoint(tmp_path):
    project = tmp_path / "project"
    result = project / "results" / "stage-outer_1_inner_1-u5000-s0-spectral-c14"
    result.mkdir(parents=True)
    (result / "checkpoint.pt").write_text("orphaned")
    (project / "slurm").mkdir()
    (project / "slurm" / "run-one.sbatch").write_text("")
    manifest = project / "manifest.tsv"
    manifest.write_text("1\touter_1_inner_1\t5000\t0\tspectral\t14\n")
    completed = subprocess.run(
        ["bash", RESUME_STAGE, manifest, "7"],
        env=os.environ | {"NUMERAI_PROJECT": str(project)},
        capture_output=True, text=True, check=False,
    )
    assert completed.returncode != 0
    assert "partial restart checkpoint" in completed.stderr


def test_supervisor_can_defer_nonpaired_outer_summary(tmp_path):
    project, manifest, env = _project(tmp_path, complete=True)
    subprocess.run(["bash", SCRIPT, manifest, "--skip-summary"], check=True, env=env)
    assert not (project / "uv-calls").exists()
    log = manifest.with_name("submission-supervisor.log").read_text()
    assert "stage complete; downstream audit required (retries=0)" in log


def test_supervisor_summarizes_only_exact_selection_ids(tmp_path):
    project, manifest, env = _project(tmp_path, complete=True)
    env["NUMERAI_SUMMARY_PREFIX"] = "final-"
    selection = project / "selection.json"
    selection.write_text('{"selected":{"paired_union":[7]}}')
    subprocess.run(
        ["bash", SCRIPT, manifest, "--selection", selection], check=True, env=env,
    )
    call = (project / "uv-calls").read_text()
    assert "--expected-arm-config-id adamw:7" in call
    assert "--expected-arm-config-id spectral:7" in call
    assert "results/summary-final-fold-u100-s0" in call
    assert "all exact selected cells summarized" in manifest.with_name(
        "submission-supervisor.log"
    ).read_text()


def test_outer_submit_validates_both_winners_before_any_sbatch(tmp_path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    calls = tmp_path / "sbatch-calls"
    _executable(fake_bin / "sbatch", f'printf "%s\\n" "$*" >> "{calls}"\necho 9000\n')
    env = os.environ | {"PATH": f"{fake_bin}:{os.environ['PATH']}"}
    selection = tmp_path / "selection.json"
    selection.write_text('{"selected":{"adamw":[7],"spectral":[]}}')
    failed = subprocess.run(
        ["bash", OUTER_SUBMIT, selection, "outer_1", "100000", "1", "0,1,2"],
        env=env, check=False, capture_output=True, text=True,
    )
    assert failed.returncode != 0 and not calls.exists()
    selection.write_text('{"selected":{"adamw":[7],"spectral":[8]}}')
    completed = subprocess.run(
        ["bash", OUTER_SUBMIT, selection, "outer_1", "100000", "1", "0,1,2"],
        env=env, check=True, capture_output=True, text=True,
    )
    assert len(completed.stdout.splitlines()) == 6
    assert len(calls.read_text().splitlines()) == 6


def test_outer_submit_uses_arm_specific_selected_budgets(tmp_path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    calls = tmp_path / "sbatch-calls"
    _executable(fake_bin / "sbatch", f'printf "%s\\n" "$*" >> "{calls}"\necho 9000\n')
    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps({
        "selected": {"adamw": [7], "spectral": [8]},
        "selected_updates": {"adamw": [5000], "spectral": [100000]},
    }))
    completed = subprocess.run(
        ["bash", OUTER_SUBMIT, selection, "outer_1", "selected", "1", "0,1,2"],
        env=os.environ | {"PATH": f"{fake_bin}:{os.environ['PATH']}"},
        check=True, capture_output=True, text=True,
    )
    rows = [line.split("\t") for line in completed.stdout.splitlines()]
    assert {row[2] for row in rows if row[4] == "adamw"} == {"5000"}
    assert {row[2] for row in rows if row[4] == "spectral"} == {"100000"}


def test_refit_submit_validates_both_winners_before_any_sbatch(tmp_path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    calls = tmp_path / "sbatch-calls"
    _executable(fake_bin / "sbatch", f'printf "%s\\n" "$*" >> "{calls}"\necho 9000\n')
    env = os.environ | {"PATH": f"{fake_bin}:{os.environ['PATH']}"}
    selection = tmp_path / "selection.json"
    selection.write_text('{"selected":{"adamw":[7],"spectral":[]}}')
    failed = subprocess.run(
        ["bash", SUBMIT_REFIT, selection, "100000", "1", "0,1,2"],
        env=env, check=False, capture_output=True, text=True,
    )
    assert failed.returncode != 0 and not calls.exists()
    selection.write_text('{"selected":{"adamw":[7],"spectral":[8]}}')
    completed = subprocess.run(
        ["bash", SUBMIT_REFIT, selection, "100000", "1", "0,1,2"],
        env=env, check=True, capture_output=True, text=True,
    )
    assert len(completed.stdout.splitlines()) == 6
    assert len(calls.read_text().splitlines()) == 6


def test_refit_submit_uses_arm_specific_selected_budgets(tmp_path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    _executable(fake_bin / "sbatch", "echo 9000\n")
    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps({
        "selected": {"adamw": [7], "spectral": [8]},
        "selected_updates": {"adamw": [5000], "spectral": [20000]},
    }))
    completed = subprocess.run(
        ["bash", SUBMIT_REFIT, selection, "selected", "1", "0,1,2"],
        env=os.environ | {"PATH": f"{fake_bin}:{os.environ['PATH']}"},
        check=True, capture_output=True, text=True,
    )
    rows = [line.split("\t") for line in completed.stdout.splitlines()]
    assert {row[3] for row in rows if row[1] == "adamw"} == {"5000"}
    assert {row[3] for row in rows if row[1] == "spectral"} == {"20000"}


def test_supervisor_refuses_to_treat_squeue_failure_as_completion(tmp_path):
    project, manifest, env = _project(tmp_path, complete=True)
    _executable(Path(env["PATH"].split(":", 1)[0]) / "squeue", "exit 7\n")
    failed = subprocess.run(
        ["bash", SCRIPT, manifest], env=env, check=False, capture_output=True, text=True,
    )
    assert failed.returncode != 0
    assert "refusing to infer" in failed.stderr
    assert not (project / "uv-calls").exists()


def test_outer_2_audit_watches_its_own_supervisor(tmp_path):
    project = tmp_path / "project"
    (project / "results").mkdir(parents=True)
    manifest = project / "results" / "outer.tsv"
    manifest.write_text("1\touter_2\t100000\t0\tadamw\t7\n")
    selection = project / "selection.json"
    selection.write_text('{}')
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    calls = tmp_path / "tmux-calls"
    _executable(fake_bin / "tmux", f'printf "%s\\n" "$*" >> "{calls}"\nexit 1\n')
    env = os.environ | {
        "NUMERAI_PROJECT": str(project),
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
    }
    failed = subprocess.run(
        ["bash", OUTER_AUDIT, manifest, selection, "outer_2", project / "audit"],
        env=env, check=False, capture_output=True, text=True,
    )
    assert failed.returncode != 0
    assert "outer supervisor exited" in failed.stderr
    assert "has-session -t numerai-outer2-supervisor" in calls.read_text()


def test_outer_promoters_accept_only_named_outer_splits(tmp_path):
    for script in (PROMOTERS[0], PROMOTERS[2]):
        failed = subprocess.run(
            ["bash", script, "1", "outer_4"], check=False, capture_output=True, text=True,
        )
        assert failed.returncode != 0
        assert "outer split must be" in failed.stderr


def test_submit_stage_emits_exact_paired_manifest(tmp_path):
    project = tmp_path / "project"
    (project / "slurm").mkdir(parents=True)
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    _executable(fake_bin / "sbatch", 'echo "$((9000 + RANDOM))"\n')
    env = os.environ | {
        "NUMERAI_PROJECT": str(project),
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
    }
    completed = subprocess.run(
        ["bash", SUBMIT_STAGE, "outer_2_inner_1", "5000", "0", "7"],
        env=env, check=True, capture_output=True, text=True,
    )
    rows = [line.split("\t") for line in completed.stdout.splitlines()]
    assert len(rows) == 80 and all(len(row) == 6 for row in rows)
    assert {(row[4], int(row[5])) for row in rows} == {
        (arm, config_id) for arm in ("adamw", "spectral") for config_id in range(40)
    }
    assert {row[1] for row in rows} == {"outer_2_inner_1"}


def test_submit_selected_can_reuse_only_existing_exact_result(tmp_path):
    project = tmp_path / "project"
    (project / "slurm").mkdir(parents=True)
    result = project / "results" / "stage-outer_2_inner_1-u100000-s0-adamw-c7"
    result.mkdir(parents=True)
    (result / "result.json").write_text("{}")
    selection = project / "selection.json"
    selection.write_text('{"selected":{"paired_union":[7]}}')
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    calls = tmp_path / "sbatch-calls"
    _executable(fake_bin / "sbatch", f'printf "%s\\n" "$*" >> "{calls}"\necho 9001\n')
    env = os.environ | {
        "NUMERAI_PROJECT": str(project),
        "NUMERAI_REUSE_COMPLETE": "1",
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
    }
    completed = subprocess.run(
        ["bash", SUBMIT_SELECTED, selection, "100000", "0", "5", "outer_2_inner_1"],
        env=env, check=True, capture_output=True, text=True,
    )
    rows = [line.split("\t") for line in completed.stdout.splitlines()]
    assert rows[0][0] == "0" and rows[0][4:] == ["adamw", "7"]
    assert rows[1][0] == "9001" and rows[1][4:] == ["spectral", "7"]
    assert len(calls.read_text().splitlines()) == 1


def test_submit_successive_plan_uses_budget_specific_and_long_scout_ids(tmp_path):
    project = tmp_path / "project"
    (project / "slurm").mkdir(parents=True)
    plan = project / "plan.json"
    plan.write_text(json.dumps({
        "confirmation_selections": {
            "5000": {"paired_union": [1, 2]},
            "20000": {"paired_union": [2, 3]},
        },
        "long_scout_paired_union": [3, 7],
    }))
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    _executable(fake_bin / "sbatch", "echo 9001\n")
    env = os.environ | {
        "NUMERAI_PROJECT": str(project),
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
    }
    confirmed = subprocess.run(
        ["bash", SUBMIT_SUCCESSIVE, plan, "confirmation", "20000", "1", "4", "fold"],
        env=env, check=True, capture_output=True, text=True,
    )
    scouted = subprocess.run(
        ["bash", SUBMIT_SUCCESSIVE, plan, "long-scout", "100000", "0", "4", "fold"],
        env=env, check=True, capture_output=True, text=True,
    )
    assert {(row.split("\t")[4], int(row.split("\t")[5]))
            for row in confirmed.stdout.splitlines()} == {
        (arm, config_id) for arm in ("adamw", "spectral") for config_id in (2, 3)
    }
    assert {(row.split("\t")[4], int(row.split("\t")[5]))
            for row in scouted.stdout.splitlines()} == {
        (arm, config_id) for arm in ("adamw", "spectral") for config_id in (3, 7)
    }


def test_submit_successive_plan_rejects_100k_confirmation(tmp_path):
    plan = tmp_path / "plan.json"
    plan.write_text("{}")
    failed = subprocess.run(
        ["bash", SUBMIT_SUCCESSIVE, plan, "confirmation", "100000", "0", "4", "fold"],
        check=False, capture_output=True, text=True,
    )
    assert failed.returncode != 0
    assert "invalid successive-plan" in failed.stderr


def test_submit_successive_finalists_preserves_ordinary_pairing_and_rank_asymmetry(tmp_path):
    project = tmp_path / "project"
    (project / "slurm").mkdir(parents=True)
    selection = project / "finalists.json"
    selection.write_text(json.dumps({
        "ordinary_confirmation_paired_union": [1, 7],
        "high_rank_spectral": [101, 102],
    }))
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    _executable(fake_bin / "sbatch", "echo 9001\n")
    env = os.environ | {
        "NUMERAI_PROJECT": str(project),
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
    }
    ordinary = subprocess.run(
        ["bash", SUBMIT_FINALISTS, selection, "ordinary", "100000", "0", "4", "fold"],
        env=env, check=True, capture_output=True, text=True,
    )
    ranks = subprocess.run(
        ["bash", SUBMIT_FINALISTS, selection, "high-rank", "20000", "1", "4", "fold"],
        env=env, check=True, capture_output=True, text=True,
    )
    assert {(row.split("\t")[4], int(row.split("\t")[5]))
            for row in ordinary.stdout.splitlines()} == {
        (arm, config_id) for arm in ("adamw", "spectral") for config_id in (1, 7)
    }
    assert {(row.split("\t")[4], int(row.split("\t")[5]))
            for row in ranks.stdout.splitlines()} == {
        ("spectral", 101), ("spectral", 102),
    }


def test_submit_budgeted_selected_pairs_every_candidate_across_arms(tmp_path):
    project = tmp_path / "project"
    (project / "slurm").mkdir(parents=True)
    selection = project / "selection.json"
    selection.write_text(json.dumps({
        "budgeted_candidates": [
            {"config_id": 7, "updates": 5000},
            {"config_id": 8, "updates": 100000},
        ],
    }))
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    _executable(fake_bin / "sbatch", "echo 9001\n")
    completed = subprocess.run(
        ["bash", SUBMIT_BUDGETED, selection, "0", "4", "outer_3_inner_1"],
        env=os.environ | {
            "NUMERAI_PROJECT": str(project),
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
        },
        check=True, capture_output=True, text=True,
    )
    rows = [line.split("\t") for line in completed.stdout.splitlines()]
    assert len(rows) == 4
    assert {(row[4], int(row[5]), int(row[2])) for row in rows} == {
        (arm, config_id, updates)
        for arm in ("adamw", "spectral")
        for config_id, updates in ((7, 5000), (8, 100000))
    }


def test_high_rank_submit_launches_only_spectral_rank_variants(tmp_path):
    project = tmp_path / "project"
    (project / "slurm").mkdir(parents=True)
    selection = project / "selection.json"
    selection.write_text(json.dumps({
        "selected": {"high_rank_spectral": [1000512, 1001024]},
    }))
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    calls = tmp_path / "sbatch-calls"
    _executable(fake_bin / "sbatch", f'printf "%s\\n" "$*" >> "{calls}"\necho 9001\n')
    env = os.environ | {
        "NUMERAI_PROJECT": str(project),
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
    }
    completed = subprocess.run(
        ["bash", SUBMIT_HIGH_RANK, selection, "100000", "0", "7",
         "outer_1_inner_1", "outer_1_inner_2"],
        env=env, check=True, capture_output=True, text=True,
    )
    rows = [line.split("\t") for line in completed.stdout.splitlines()]
    assert len(rows) == 4 and {row[4] for row in rows} == {"spectral"}
    assert {int(row[5]) for row in rows} == {1000512, 1001024}


def test_launch_outer_builds_atomic_f0_manifest_and_controllers(tmp_path):
    project = tmp_path / "project"
    (project / "results").mkdir(parents=True)
    (project / "slurm").mkdir()
    submit = project / "slurm" / "submit-stage.sh"
    _executable(
        submit,
        'job=9000\nfor config in $(seq 0 39); do for arm in adamw spectral; do '
        'job=$((job+1)); printf "%s\\touter_3_inner_1\\t5000\\t0\\t%s\\t%s\\n" '
        '"$job" "$arm" "$config"; done; done\n',
    )
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    calls = tmp_path / "tmux-calls"
    _executable(
        fake_bin / "tmux",
        f'if [[ $1 == has-session ]]; then exit 1; fi\nprintf "%s\\n" "$*" >> "{calls}"\n',
    )
    env = os.environ | {
        "NUMERAI_PROJECT": str(project),
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
    }
    subprocess.run(["bash", LAUNCH_OUTER, "outer_3", "7"], check=True, env=env)
    manifest = project / "results" / "submission-outer_3-f0-u5000-s0.tsv"
    assert len(manifest.read_text().splitlines()) == 80
    assert not manifest.with_suffix(".tsv.tmp").exists()
    tmux_calls = calls.read_text()
    assert "new-session -d -s numerai-outer3-f0-monitor" in tmux_calls
    assert "new-session -d -s numerai-outer3-f0-promote" in tmux_calls


def test_launch_outer_rejects_duplicate_manifest_coverage(tmp_path):
    project = tmp_path / "project"
    (project / "results").mkdir(parents=True)
    (project / "slurm").mkdir()
    _executable(
        project / "slurm" / "submit-stage.sh",
        'for i in $(seq 1 80); do printf "%s\\touter_2_inner_1\\t5000\\t0\\tadamw\\t0\\n" '
        '"$((9000+i))"; done\n',
    )
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    _executable(fake_bin / "tmux", "exit 1\n")
    env = os.environ | {
        "NUMERAI_PROJECT": str(project),
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
    }
    failed = subprocess.run(
        ["bash", LAUNCH_OUTER, "outer_2", "7"],
        env=env, check=False, capture_output=True, text=True,
    )
    assert failed.returncode != 0
    assert "40 configs x two arms" in failed.stderr
    assert not (project / "results" / "submission-outer_2-f0-u5000-s0.tsv").exists()


def test_nested_controller_confirms_fixed_walkforward_and_submits_refits(tmp_path):
    project = tmp_path / "project"
    results = project / "results"
    (project / "slurm").mkdir(parents=True)
    selection = {
        "selected": {"adamw": [1], "spectral": [2]},
        "selected_updates": {"adamw": [5000], "spectral": [100000]},
    }
    results.mkdir()
    (results / "selection-outer_1-f2-budget-top1.json").write_text(json.dumps(selection))
    for number in range(1, 4):
        audit = results / f"audit-outer_{number}-budgeted"
        audit.mkdir()
        (audit / "outer-audit.json").write_text(json.dumps({
            "status": "audit_complete", "split": {"name": f"outer_{number}"},
            "selected": {"adamw": 1, "spectral": 2},
            "updates": {"adamw": 5000, "spectral": 100000},
        }))
    _executable(project / "slurm" / "sync-env.sbatch", "exit 0\n")
    _executable(
        project / "uv",
        'if [[ "$*" == *confirm_walkforward* ]]; then cp '
        '"$NUMERAI_PROJECT/results/selection-outer_1-f2-budget-top1.json" '
        '"$NUMERAI_PROJECT/results/selection-final-top1.json"; fi\n',
    )
    _executable(
        project / "slurm" / "build-oof-candidate.sh",
        'mkdir -p "$NUMERAI_PROJECT/results/nested-outer"\n'
        'printf \'{"status":"complete"}\\n\' > '
        '"$NUMERAI_PROJECT/results/nested-outer/nested-outer-report.json"\n'
        'printf \'{"status":"frozen_train_only_selection"}\\n\' > '
        '"$NUMERAI_PROJECT/results/candidate-plan.json"\n',
    )
    _executable(
        project / "slurm" / "submit-refit.sh",
        'for arm in adamw spectral; do for seed in 0 1 2; do '
        'printf "9001\\t%s\\t%s\\t%s\\t%s\\n" "$arm" '
        '"$([[ $arm == adamw ]] && echo 1 || echo 2)" '
        '"$([[ $arm == adamw ]] && echo 5000 || echo 100000)" "$seed"; done; done\n',
    )
    _executable(
        project / "slurm" / "supervise-refits.sh",
        'mkdir -p "$NUMERAI_PROJECT/results/audit-final-refits-budgeted"\n'
        'printf \'{"status":"audit_complete","cells":6,'
        '"updates":{"adamw":5000,"spectral":100000},'
        '"seeds":[0,1,2]}\\n\' > '
        '"$NUMERAI_PROJECT/results/audit-final-refits-budgeted/refit-audit.json"\n',
    )
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    sbatch_calls = tmp_path / "sbatch-calls"
    _executable(fake_bin / "sbatch", f'printf "%s\\n" "$*" >> "{sbatch_calls}"\necho 9001\n')
    _executable(
        fake_bin / "tmux",
        'if [[ $1 == new-session ]]; then bash -c "${@: -1}"; else exit 1; fi\n',
    )
    env = os.environ | {
        "NUMERAI_PROJECT": str(project), "NUMERAI_POLL_SECONDS": "0",
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
    }
    subprocess.run(["bash", CONTINUE_NESTED], check=True, env=env)
    assert len(sbatch_calls.read_text().splitlines()) == 1
    assert (results / "selection-final-top1.json").is_file()
    assert "walk-forward pipeline ready" in (
        results / "continue-nested-pipeline.log"
    ).read_text()


def test_build_oof_resolves_three_audited_folds_and_three_seeds(tmp_path):
    project = tmp_path / "project"
    results = project / "results"
    results.mkdir(parents=True)
    selection = {
        "selected": {"adamw": [1], "spectral": [4]},
        "selected_updates": {"adamw": [100000], "spectral": [100000]},
    }
    (results / "selection-final-top1.json").write_text(json.dumps(selection))
    for number in range(1, 4):
        audit_dir = results / f"audit-outer_{number}-budgeted"
        audit_dir.mkdir()
        (audit_dir / "outer-audit.json").write_text(json.dumps({
                "status": "audit_complete", "split": {"name": f"outer_{number}"},
                "selected": {"adamw": 1, "spectral": 4},
                "updates": {"adamw": 100000, "spectral": 100000},
        }))
        for arm, config_id in (("adamw", 1), ("spectral", 4)):
            for seed in range(3):
                directory = results / (
                    f"stage-outer_{number}-u100000-s{seed}-{arm}-c{config_id}"
                )
                directory.mkdir()
                (directory / "result.json").write_text("{}")
    calls = project / "uv-calls"
    _executable(
        project / "uv",
        f'printf "%s\\n" "$*" >> "{calls}"\n'
        f'if [[ "$*" == *numerai_competitive.oof* ]]; then mkdir -p "{results}/nested-outer"; '
        f'touch "{results}/nested-outer/nested-outer-predictions.npz"; fi\n'
        f'if [[ "$*" == *numerai_competitive.candidate* ]]; then touch '
        f'"{results}/candidate-plan.json"; fi\n',
    )
    env = os.environ | {"NUMERAI_PROJECT": str(project)}
    subprocess.run(["bash", BUILD_OOF], check=True, env=env)
    text = calls.read_text()
    assert text.count("--adamw-result") == 9
    assert text.count("--spectral-result") == 9
    assert "numerai_competitive.candidate" in text


def test_sealed_evaluation_submits_only_after_refit_audit_and_candidate(tmp_path):
    project = tmp_path / "project"
    results = project / "results"
    (results / "audit-final-refits-budgeted").mkdir(parents=True)
    (project / "slurm").mkdir()
    (results / "search-v1-high-rank.json").write_text("{}")
    (results / "selection-final-top1.json").write_text(json.dumps({
        "selected": {"adamw": [7], "spectral": [8]},
        "selected_updates": {"adamw": [100000], "spectral": [100000]},
    }))
    (results / "audit-final-refits-budgeted" / "refit-audit.json").write_text(json.dumps({
        "status": "audit_complete", "cells": 6,
        "updates": {"adamw": 100000, "spectral": 100000},
        "seeds": [0, 1, 2], "selected": {"adamw": 7, "spectral": 8},
    }))
    (results / "candidate-plan.json").write_text(json.dumps({
        "status": "frozen_train_only_selection",
    }))
    commit = "a" * 40
    (project / "code-snapshot.json").write_text(json.dumps({
        "status": "complete", "code_commit": commit,
    }))
    for arm, config in (("adamw", 7), ("spectral", 8)):
        for seed in range(3):
            directory = results / f"final-refit-u100000-s{seed}-{arm}-c{config}"
            directory.mkdir()
            (directory / "model.pt").write_text("model")
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    calls = tmp_path / "sbatch-calls"
    _executable(fake_bin / "sbatch", f'printf "%s\\n" "$*" >> "{calls}"\necho 9001\n')
    env = os.environ | {
        "NUMERAI_PROJECT": str(project),
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
    }
    completed = subprocess.run(
        ["bash", SUBMIT_SEALED, commit], env=env, check=True,
        capture_output=True, text=True,
    )
    assert len(calls.read_text().splitlines()) == 2
    assert "--dependency=afterok:9001" in calls.read_text()
    assert "official_validation" in completed.stdout
    assert (results / "submission-sealed-evaluation.tsv").is_file()


def test_sealed_evaluation_refuses_disagreeing_refit_audit_before_sbatch(tmp_path):
    project = tmp_path / "project"
    results = project / "results"
    (results / "audit-final-refits-u100000").mkdir(parents=True)
    (project / "slurm").mkdir()
    (results / "search-v1-high-rank.json").write_text("{}")
    (results / "selection-final-top1.json").write_text(json.dumps({
        "selected": {"adamw": [7], "spectral": [8]},
    }))
    (results / "audit-final-refits-u100000" / "refit-audit.json").write_text(json.dumps({
        "status": "audit_complete", "cells": 6, "updates": 100000,
        "seeds": [0, 1, 2], "selected": {"adamw": 9, "spectral": 8},
    }))
    (results / "candidate-plan.json").write_text(json.dumps({
        "status": "frozen_train_only_selection",
    }))
    (project / "code-snapshot.json").write_text(json.dumps({
        "status": "complete", "code_commit": "a" * 40,
    }))
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    calls = tmp_path / "sbatch-calls"
    _executable(fake_bin / "sbatch", f'printf "%s\\n" "$*" >> "{calls}"\necho 9001\n')
    env = os.environ | {
        "NUMERAI_PROJECT": str(project),
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
    }
    failed = subprocess.run(
        ["bash", SUBMIT_SEALED, "a" * 40], env=env, check=False,
        capture_output=True, text=True,
    )
    assert failed.returncode != 0 and not calls.exists()


def test_live_bundle_uses_frozen_candidate_after_sealed_evaluation(tmp_path):
    project = tmp_path / "project"
    results = project / "results"
    (results / "official-validation").mkdir(parents=True)
    (project / "slurm").mkdir()
    commit = "b" * 40
    (results / "freeze.json").write_text(json.dumps({
        "status": "frozen", "code_commit": commit,
        "candidate_transform": {"arm": "spectral"},
        "selected": {"spectral": {"config_id": 8, "updates": 100000}},
    }))
    (results / "official-validation" / "evaluation-complete.json").write_text(json.dumps({
        "status": "complete",
    }))
    for seed in range(3):
        directory = results / f"production-refit-s{seed}-spectral-c8"
        directory.mkdir()
        (directory / "model.pt").write_text("model")
    (results / "production-refit-audit.json").write_text(json.dumps({
        "status": "audit_complete", "arm": "spectral", "production_code_commit": commit,
    }))
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    calls = tmp_path / "sbatch-calls"
    _executable(fake_bin / "sbatch", f'printf "%s\\n" "$*" >> "{calls}"\necho 9002\n')
    env = os.environ | {
        "NUMERAI_PROJECT": str(project),
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
    }
    completed = subprocess.run(
        ["bash", SUBMIT_LIVE, commit], env=env, check=True,
        capture_output=True, text=True,
    )
    assert len(calls.read_text().splitlines()) == 2
    assert "CANDIDATE_ARM=spectral" in calls.read_text()
    assert "--dependency=afterok:9002" in calls.read_text()
    assert "validate_live_bundle" in completed.stdout
