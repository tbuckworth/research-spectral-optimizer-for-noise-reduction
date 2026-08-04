#!/usr/bin/env python3
"""Durable per-trial result persistence for the fold jobs (B2 deliverable).

The parent exp-003 sweep.py accumulated trial results in an in-memory list and
wrote sweep_results.json once at the end of main() -- a SIGKILL (Slurm --time
kill, node failure) loses every completed trial. This module replaces that
pattern with an append-only JSON-lines store where each trial's record is
flushed AND fsync'd the moment the trial completes, so a kill costs at most
the in-flight trial.

Guarantees:
  - append(record): one JSON line, written + flush + os.fsync before return.
    The line is written as a single os.write on an O_APPEND fd, so concurrent
    or interrupted writers cannot interleave partial lines (a kill mid-write
    can leave at most one torn FINAL line, which load() detects and skips).
  - load(): returns all complete records; a torn trailing line (no newline /
    unparseable) is dropped with a warning, matching crash semantics.
  - completed_ids(): the set of trial_ids already durably recorded -- the
    resume-on-restart scan.
  - assert_no_duplicates(): hard check that no trial_id appears twice.
"""
import json
import os


class TrialStore:
    def __init__(self, path):
        self.path = str(path)

    def append(self, record):
        assert "trial_id" in record, "every record needs a trial_id"
        line = json.dumps(record, sort_keys=True, default=float) + "\n"
        # O_APPEND + single os.write: atomic append on POSIX local/NFS-local
        fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        try:
            os.write(fd, line.encode("utf-8"))
            os.fsync(fd)          # durable before we return
        finally:
            os.close(fd)
        # fsync the directory entry too (first create must survive a crash)
        dfd = os.open(os.path.dirname(os.path.abspath(self.path)) or ".",
                      os.O_RDONLY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)

    def load(self):
        if not os.path.exists(self.path):
            return []
        records = []
        with open(self.path, "rb") as f:
            raw = f.read()
        for i, line in enumerate(raw.split(b"\n")):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                # torn final line from a mid-write kill: drop it (that trial
                # never durably completed); anywhere else it is corruption.
                if i == raw.count(b"\n"):
                    print(f"[trial_store] dropping torn trailing line in "
                          f"{self.path}", flush=True)
                else:
                    raise
        return records

    def completed_ids(self):
        return {r["trial_id"] for r in self.load()}

    def assert_no_duplicates(self):
        ids = [r["trial_id"] for r in self.load()]
        dupes = {i for i in ids if ids.count(i) > 1}
        assert not dupes, f"duplicate trial records for ids {sorted(dupes)}"
        return len(ids)
