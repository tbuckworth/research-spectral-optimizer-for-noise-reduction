"""Bind an rsynced execution tree to bytes from a specific Git commit."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from .data import atomic_json, sha256

FIXED_FILES = {"pyproject.toml", "uv.lock", "fidelity-protocol.md"}
REQUIRED_FILES = FIXED_FILES | {
    "src/numerai_competitive/code_snapshot.py", "src/numerai_competitive/freeze.py",
}
PREFIXES = ("src/", "configs/", "slurm/")


def _included(relative: str) -> bool:
    return relative in FIXED_FILES or relative.startswith(PREFIXES)


def _git(repo: Path, *args: str, text: bool = True):
    return subprocess.run(["git", "-C", str(repo), *args], check=True,
                          capture_output=True, text=text).stdout


def create_snapshot(repo: Path, source_prefix: str, commit: str, output: Path) -> dict:
    full_commit = _git(repo, "rev-parse", f"{commit}^{{commit}}").strip()
    paths = _git(repo, "ls-tree", "-r", "--name-only", full_commit, "--", source_prefix)
    prefix = source_prefix.rstrip("/") + "/"
    relative_paths = sorted(path.removeprefix(prefix) for path in paths.splitlines()
                            if path.startswith(prefix) and _included(path.removeprefix(prefix)))
    if not relative_paths or not REQUIRED_FILES <= set(relative_paths):
        raise ValueError("commit lacks the required execution tree")
    files = {}
    for relative in relative_paths:
        content = _git(repo, "show", f"{full_commit}:{prefix}{relative}", text=False)
        files[relative] = hashlib.sha256(content).hexdigest()
    digest = hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode())
    report = {
        "status": "complete", "code_commit": full_commit,
        "source_prefix": source_prefix.rstrip("/"), "files": files,
        "file_count": len(files), "file_map_sha256": digest.hexdigest(),
    }
    atomic_json(output, report)
    return report


def verify_snapshot(root: Path, snapshot_path: Path, expected_commit: str) -> dict:
    snapshot = json.loads(snapshot_path.read_text())
    files = snapshot.get("files", {})
    discovered = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*") if path.is_file()
        and _included(path.relative_to(root).as_posix())
        and "__pycache__" not in path.parts
    }
    digest = hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode())
    if (snapshot.get("status") != "complete" or snapshot.get("code_commit") != expected_commit
            or not REQUIRED_FILES <= set(files) or snapshot.get("file_count") != len(files)
            or snapshot.get("file_map_sha256") != digest.hexdigest()
            or set(files) != discovered):
        raise ValueError("code snapshot metadata or execution-file coverage differs")
    mismatches = [relative for relative, expected in files.items()
                  if sha256(root / relative) != expected]
    if mismatches:
        raise ValueError(f"rsynced execution tree differs from commit: {mismatches[:5]}")
    return snapshot


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    create = subparsers.add_parser("create")
    create.add_argument("--repo", type=Path, required=True)
    create.add_argument("--source-prefix", required=True)
    create.add_argument("--commit", required=True)
    create.add_argument("--output", type=Path, required=True)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--root", type=Path, required=True)
    verify.add_argument("--snapshot", type=Path, required=True)
    verify.add_argument("--commit", required=True)
    args = parser.parse_args()
    result = (create_snapshot(args.repo, args.source_prefix, args.commit, args.output)
              if args.command == "create"
              else verify_snapshot(args.root, args.snapshot, args.commit))
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
