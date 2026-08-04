#!/usr/bin/env python3
"""Supervisor for the B2 kill-test (exp-f03 sub-check 1 and 3).

1. Launch harness.py --phase sweep (6 synthetic trials).
2. When out/state/trial_4.started appears, wait 0.5 s (mid-trial-4) and send
   an ACTUAL SIGKILL to the harness process.
3. Verify: process died from SIGKILL (rc == -9); trials.jsonl holds exactly
   trials {1,2,3}, every line parseable.
4. Restart the harness; verify from its log that it SKIPs 1-3 and RUNs only
   4-6, and that the final store holds exactly ids 1..6 with no duplicates.
5. Refit kill: launch --phase refit, SIGKILL after the first checkpoint
   lands, restart, verify it resumes from a step > 0 and completes.

Exit code 0 <=> every check passed. All checks print CHECK lines.
"""
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

SRC = Path(__file__).resolve().parent
OUT = SRC.parent / "out" / "kill_test"
PY = sys.executable

ENV = dict(os.environ, CUDA_VISIBLE_DEVICES="")   # zero-GPU, belt and braces

FAILURES = []


def check(name, ok, detail=""):
    print(f"CHECK [{name}]: {'PASS' if ok else 'FAIL'} {detail}", flush=True)
    if not ok:
        FAILURES.append(name)


def launch(phase, logname, extra=()):
    log = open(OUT / logname, "w")
    p = subprocess.Popen(
        [PY, str(SRC / "harness.py"), "--outdir", str(OUT), "--phase", phase,
         *extra],
        stdout=log, stderr=subprocess.STDOUT, env=ENV, cwd=str(SRC))
    return p, log


def wait_for(path, timeout=60):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if path.exists():
            return True
        time.sleep(0.05)
    return False


def load_jsonl(path):
    recs, torn = [], 0
    if not path.exists():
        return recs, torn
    with open(path, "rb") as f:
        for line in f.read().split(b"\n"):
            if not line.strip():
                continue
            try:
                recs.append(json.loads(line))
            except json.JSONDecodeError:
                torn += 1
    return recs, torn


def main():
    if OUT.exists():
        import shutil
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    store = OUT / "trials.jsonl"

    # ---------- phase 1: sweep + SIGKILL mid-trial-4 ----------
    print("=== sweep kill-test ===", flush=True)
    p, log = launch("sweep", "run1_sweep.log")
    ok = wait_for(OUT / "state" / "trial_4.started")
    check("trial4-started-marker", ok)
    time.sleep(0.5)                                   # land inside trial 4
    os.kill(p.pid, signal.SIGKILL)                    # ACTUAL SIGKILL
    rc = p.wait()
    log.close()
    check("killed-by-sigkill", rc == -signal.SIGKILL, f"rc={rc}")

    print("\n--- state on disk immediately after SIGKILL ---", flush=True)
    for f in sorted(OUT.rglob("*")):
        if f.is_file():
            print(f"  {f.relative_to(OUT)}  {f.stat().st_size}B", flush=True)
    recs, torn = load_jsonl(store)
    ids_after_kill = sorted(r["trial_id"] for r in recs)
    print(f"trials.jsonl after kill: ids={ids_after_kill}, "
          f"torn_lines={torn}", flush=True)
    check("survivors-1-2-3", ids_after_kill == [1, 2, 3],
          f"ids={ids_after_kill}")
    check("no-torn-lines", torn == 0, f"torn={torn}")

    # ---------- phase 2: restart, verify resume runs exactly 4-6 ----------
    print("\n=== sweep resume ===", flush=True)
    p, log = launch("sweep", "run2_sweep.log")
    rc = p.wait(timeout=120)
    log.close()
    check("resume-exit-ok", rc == 0, f"rc={rc}")
    run2 = (OUT / "run2_sweep.log").read_text()
    print(run2, flush=True)
    skipped = sorted(int(l.split("trial")[1].split("(")[0])
                     for l in run2.splitlines() if "SKIP trial" in l)
    ran = sorted(int(l.split("trial")[1].split("(")[0])
                 for l in run2.splitlines() if "RUN  trial" in l)
    check("resume-skips-1-2-3", skipped == [1, 2, 3], f"skipped={skipped}")
    check("resume-runs-4-5-6", ran == [4, 5, 6], f"ran={ran}")

    recs, torn = load_jsonl(store)
    ids_final = sorted(r["trial_id"] for r in recs)
    check("final-ids-1..6-no-dupes", ids_final == [1, 2, 3, 4, 5, 6],
          f"ids={ids_final}")
    check("final-no-torn-lines", torn == 0, f"torn={torn}")
    # no completed trial ran twice: RUN markers across both logs
    run1 = (OUT / "run1_sweep.log").read_text()
    ran1 = sorted(int(l.split("trial")[1].split("(")[0])
                  for l in run1.splitlines() if "RUN  trial" in l)
    all_runs = ran1 + ran
    completed_runs = [t for t in all_runs if all_runs.count(t) > 1
                      and t in ids_final]
    # trial 4 STARTED in run1 (killed mid-flight, never completed) and ran
    # again in run2 -- that is exactly the "one trial lost" semantics.
    twice_completed = [t for t in set(all_runs)
                       if all_runs.count(t) > 1 and t in set(ids_after_kill)]
    check("no-completed-trial-rerun", twice_completed == [],
          f"started-in-both={sorted(set(ran1) & set(ran))} "
          f"(allowed only for the killed in-flight trial)")

    # ---------- phase 3: refit kill + resume ----------
    print("\n=== refit kill-test ===", flush=True)
    p, log = launch("refit", "run3_refit.log", extra=("--ckpt-every", "10"))
    ok = wait_for(OUT / "refit_ckpt.pt")
    check("refit-ckpt-appears", ok)
    time.sleep(0.3)                                   # past ckpt, mid-refit
    os.kill(p.pid, signal.SIGKILL)
    rc = p.wait()
    log.close()
    check("refit-killed-by-sigkill", rc == -signal.SIGKILL, f"rc={rc}")
    check("refit-not-done-after-kill", not (OUT / "refit_done.json").exists())

    p, log = launch("refit", "run4_refit.log", extra=("--ckpt-every", "10"))
    rc = p.wait(timeout=120)
    log.close()
    run4 = (OUT / "run4_refit.log").read_text()
    print(run4, flush=True)
    check("refit-resume-exit-ok", rc == 0, f"rc={rc}")
    check("refit-resumed-log", "RESUMED from checkpoint at step" in run4)
    done = json.loads((OUT / "refit_done.json").read_text())
    check("refit-resumed-from>0", done.get("resumed_from", 0) > 0,
          f"resumed_from={done.get('resumed_from')}")
    check("refit-completed-all-steps", done.get("steps") == 60,
          f"steps={done.get('steps')}")

    print(f"\nKILL-TEST RESULT: "
          f"{'ALL PASS' if not FAILURES else 'FAILURES: ' + str(FAILURES)}",
          flush=True)
    sys.exit(0 if not FAILURES else 1)


if __name__ == "__main__":
    main()
