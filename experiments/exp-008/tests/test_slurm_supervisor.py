import json
import os
import subprocess
from pathlib import Path

SCRIPT = Path(__file__).parents[1] / "slurm" / "supervise-resumable-stage.sh"
OUTER_SUBMIT = Path(__file__).parents[1] / "slurm" / "submit-outer-eval.sh"
OUTER_AUDIT = Path(__file__).parents[1] / "slurm" / "audit-outer-when-ready.sh"
SUBMIT_STAGE = Path(__file__).parents[1] / "slurm" / "submit-stage.sh"
SUBMIT_SELECTED = Path(__file__).parents[1] / "slurm" / "submit-selected.sh"
RESUME_STAGE = Path(__file__).parents[1] / "slurm" / "resume-checkpointed-stage.sh"
LAUNCH_OUTER = Path(__file__).parents[1] / "slurm" / "launch-outer.sh"
CONTINUE_NESTED = Path(__file__).parents[1] / "slurm" / "continue-nested-pipeline.sh"
SUBMIT_REFIT = Path(__file__).parents[1] / "slurm" / "submit-refit.sh"
BUILD_OOF = Path(__file__).parents[1] / "slurm" / "build-oof-candidate.sh"
SUBMIT_SEALED = Path(__file__).parents[1] / "slurm" / "submit-sealed-evaluation.sh"
SUBMIT_LIVE = Path(__file__).parents[1] / "slurm" / "submit-live-bundle.sh"
PROMOTERS = [
    Path(__file__).parents[1] / "slurm" / name
    for name in (
        "promote-f0-when-ready.sh",
        "promote-f1-when-ready.sh",
        "promote-f2-when-ready.sh",
    )
]


def _executable(path: Path, body: str) -> None:
    path.write_text("#!/bin/bash\nset -euo pipefail\n" + body)
    path.chmod(0o755)


def _project(tmp_path: Path, *, complete: bool) -> tuple[Path, Path, dict[str, str]]:
    project = tmp_path / "project"
    (project / "results").mkdir(parents=True)
    (project / "configs").mkdir()
    (project / "slurm").mkdir()
    manifest = project / "results" / "submission.tsv"
    manifest.write_text("1\tfold\t100\t0\tadamw\t7\n")
    result_dir = project / "results" / "stage-fold-u100-s0-adamw-c7"
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


def test_supervisor_resubmits_exact_checkpoint_before_summary(tmp_path):
    project, manifest, env = _project(tmp_path, complete=False)
    result_dir = project / "results" / "stage-fold-u100-s0-adamw-c7"
    (result_dir / "checkpoint.pt").write_text("checkpoint")
    (result_dir / "checkpoint-status.json").write_text("{}")
    _executable(
        project / "slurm" / "resume-checkpointed-stage.sh",
        'touch "$NUMERAI_PROJECT/results/stage-fold-u100-s0-adamw-c7/result.json"\n'
        'printf "2\\tfold\\t100\\t0\\tadamw\\t7\\n"\n',
    )
    subprocess.run(["bash", SCRIPT, manifest], check=True, env=env)
    retry = manifest.with_name("submission.retry-1.tsv")
    assert retry.read_text() == "2\tfold\t100\t0\tadamw\t7\n"
    log = manifest.with_name("submission-supervisor.log").read_text()
    assert "resubmitted 1 checkpointed tasks" in log
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
    assert "--expected-configs 1 --expected-config-id 7" in call
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
    for script in PROMOTERS:
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


def test_nested_controller_launches_remaining_outers_and_final_stage(tmp_path):
    project = tmp_path / "project"
    results = project / "results"
    (project / "slurm").mkdir(parents=True)
    first = results / "audit-outer_1-u100000"
    first.mkdir(parents=True)
    (first / "outer-audit.json").write_text(json.dumps({
        "status": "audit_complete", "split": {"name": "outer_1"},
    }))
    _executable(project / "slurm" / "sync-env.sbatch", "exit 0\n")
    launches = project / "launches"
    _executable(
        project / "slurm" / "launch-outer.sh",
        f'printf "%s\\n" "$*" >> "{launches}"\n'
        'number=${1#outer_}\n'
        'mkdir -p "$NUMERAI_PROJECT/results/audit-outer_${number}-u100000"\n'
        'printf \'{"status":"audit_complete","split":{"name":"%s"}}\\n\' "$1" > '
        '"$NUMERAI_PROJECT/results/audit-outer_${number}-u100000/outer-audit.json"\n',
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
        project / "slurm" / "launch-final-selection.sh",
        f'printf "%s\\n" "$*" >> "{launches}"\n'
        'touch "$NUMERAI_PROJECT/results/submission-final-selection-u100000.tsv"\n'
        'mkdir -p "$NUMERAI_PROJECT/results/audit-final-refits-u100000"\n'
        'printf \'{"status":"audit_complete","cells":6,"updates":100000,'
        '"seeds":[0,1,2]}\\n\' > '
        '"$NUMERAI_PROJECT/results/audit-final-refits-u100000/refit-audit.json"\n',
    )
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    sbatch_calls = tmp_path / "sbatch-calls"
    _executable(fake_bin / "sbatch", f'printf "%s\\n" "$*" >> "{sbatch_calls}"\necho 9001\n')
    env = os.environ | {
        "NUMERAI_PROJECT": str(project), "NUMERAI_POLL_SECONDS": "0",
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
    }
    subprocess.run(["bash", CONTINUE_NESTED], check=True, env=env)
    assert launches.read_text().splitlines() == ["outer_2 9001", "outer_3 9001", "9001"]
    assert len(sbatch_calls.read_text().splitlines()) == 3
    assert "ready for immutable freeze" in (
        results / "continue-nested-pipeline.log"
    ).read_text()


def test_build_oof_resolves_three_audited_folds_and_three_seeds(tmp_path):
    project = tmp_path / "project"
    results = project / "results"
    results.mkdir(parents=True)
    for number in range(1, 4):
        selection = {"selected": {"adamw": [number], "spectral": [number + 3]}}
        (results / f"selection-outer_{number}-f2-top1.json").write_text(
            json.dumps(selection)
        )
        audit_dir = results / f"audit-outer_{number}-u100000"
        audit_dir.mkdir()
        (audit_dir / "outer-audit.json").write_text(json.dumps({
            "status": "audit_complete", "split": {"name": f"outer_{number}"},
            "selected": {"adamw": number, "spectral": number + 3},
        }))
        for arm, config_id in (("adamw", number), ("spectral", number + 3)):
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
    (results / "audit-final-refits-u100000").mkdir(parents=True)
    (project / "slurm").mkdir()
    (results / "search-v1-high-rank.json").write_text("{}")
    (results / "selection-final-top1.json").write_text(json.dumps({
        "selected": {"adamw": [7], "spectral": [8]},
    }))
    (results / "audit-final-refits-u100000" / "refit-audit.json").write_text(json.dumps({
        "status": "audit_complete", "cells": 6, "updates": 100000,
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
        "selected": {"spectral": {"config_id": 8}},
    }))
    (results / "official-validation" / "evaluation-complete.json").write_text(json.dumps({
        "status": "complete",
    }))
    for seed in range(3):
        directory = results / f"final-refit-u100000-s{seed}-spectral-c8"
        directory.mkdir()
        (directory / "model.pt").write_text("model")
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
