import os
import subprocess
from pathlib import Path

SCRIPT = Path(__file__).parents[1] / "slurm" / "supervise-resumable-stage.sh"


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
