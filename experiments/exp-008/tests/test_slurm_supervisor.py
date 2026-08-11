import os
import subprocess
from pathlib import Path

SCRIPT = Path(__file__).parents[1] / "slurm" / "supervise-resumable-stage.sh"
OUTER_SUBMIT = Path(__file__).parents[1] / "slurm" / "submit-outer-eval.sh"
OUTER_AUDIT = Path(__file__).parents[1] / "slurm" / "audit-outer-when-ready.sh"
SUBMIT_STAGE = Path(__file__).parents[1] / "slurm" / "submit-stage.sh"
LAUNCH_OUTER = Path(__file__).parents[1] / "slurm" / "launch-outer.sh"
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


def test_supervisor_can_defer_nonpaired_outer_summary(tmp_path):
    project, manifest, env = _project(tmp_path, complete=True)
    subprocess.run(["bash", SCRIPT, manifest, "--skip-summary"], check=True, env=env)
    assert not (project / "uv-calls").exists()
    log = manifest.with_name("submission-supervisor.log").read_text()
    assert "stage complete; downstream audit required (retries=0)" in log


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
