import subprocess

import pytest

from numerai_competitive.code_snapshot import create_snapshot, verify_snapshot


def _git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *args], check=True,
                          capture_output=True, text=True).stdout.strip()


def test_snapshot_binds_rsynced_execution_tree_to_git_commit(tmp_path):
    repo = tmp_path / "repo"
    source = repo / "experiments" / "exp-008"
    for relative in ("pyproject.toml", "uv.lock", "fidelity-protocol.md",
                     "src/numerai_competitive/code_snapshot.py",
                     "src/numerai_competitive/freeze.py", "slurm/run.sbatch"):
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative)
    _git(repo, "init")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "snapshot")
    commit = _git(repo, "rev-parse", "HEAD")
    snapshot = tmp_path / "snapshot.json"
    report = create_snapshot(repo, "experiments/exp-008", commit, snapshot)
    assert report["code_commit"] == commit and report["file_count"] == 6
    assert verify_snapshot(source, snapshot, commit)["file_map_sha256"] == report[
        "file_map_sha256"
    ]
    (source / "src/numerai_competitive/freeze.py").write_text("mutated")
    with pytest.raises(ValueError, match="differs from commit"):
        verify_snapshot(source, snapshot, commit)


def test_snapshot_rejects_untracked_execution_file(tmp_path):
    repo = tmp_path / "repo"
    source = repo / "experiments" / "exp-008"
    for relative in ("pyproject.toml", "uv.lock", "fidelity-protocol.md",
                     "src/numerai_competitive/code_snapshot.py",
                     "src/numerai_competitive/freeze.py"):
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative)
    _git(repo, "init")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "snapshot")
    commit = _git(repo, "rev-parse", "HEAD")
    snapshot = tmp_path / "snapshot.json"
    create_snapshot(repo, "experiments/exp-008", commit, snapshot)
    extra = source / "slurm" / "untracked.sh"
    extra.parent.mkdir()
    extra.write_text("exit 0")
    with pytest.raises(ValueError, match="coverage differs"):
        verify_snapshot(source, snapshot, commit)


def test_snapshot_ignores_nfs_open_file_tombstones(tmp_path):
    repo = tmp_path / "repo"
    source = repo / "experiments" / "exp-008"
    for relative in ("pyproject.toml", "uv.lock", "fidelity-protocol.md",
                     "src/numerai_competitive/code_snapshot.py",
                     "src/numerai_competitive/freeze.py", "slurm/run.sbatch"):
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative)
    _git(repo, "init")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "snapshot")
    commit = _git(repo, "rev-parse", "HEAD")
    snapshot = tmp_path / "snapshot.json"
    create_snapshot(repo, "experiments/exp-008", commit, snapshot)
    (source / "slurm" / ".nfs0000000000000001").write_text("open tombstone")
    assert verify_snapshot(source, snapshot, commit)["code_commit"] == commit
