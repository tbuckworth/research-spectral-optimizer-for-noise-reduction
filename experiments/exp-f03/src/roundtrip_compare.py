#!/usr/bin/env python3
"""Driver for the refit checkpoint roundtrip (exp-f03 sub-check 2).

Runs refit_worker.py three times, each as a FRESH python process:
  1. reference (uninterrupted N+M steps)
  2. phase1    (N steps -> checkpoint)
  3. phase2    (fresh process, restore, M steps)

then compares the reference's last-M-step trajectory against phase2's
trajectory: bitwise equality (fp64) of the full flat parameter vector at every
step, loss equality, and bitwise equality of the final filter state (V, S,
grad_mean, step_count). Exit 0 <=> bit-continuous.
"""
import os
import subprocess
import sys
from pathlib import Path

import torch

SRC = Path(__file__).resolve().parent
OUT = SRC.parent / "out" / "roundtrip"
PY = sys.executable
ENV = dict(os.environ, CUDA_VISIBLE_DEVICES="")

FAILURES = []


def check(name, ok, detail=""):
    print(f"CHECK [{name}]: {'PASS' if ok else 'FAIL'} {detail}", flush=True)
    if not ok:
        FAILURES.append(name)


def run(mode):
    r = subprocess.run([PY, str(SRC / "refit_worker.py"), "--mode", mode,
                        "--outdir", str(OUT)],
                       env=ENV, cwd=str(SRC), capture_output=True, text=True,
                       timeout=300)
    print(r.stdout, end="", flush=True)
    if r.stderr:
        print(r.stderr, end="", flush=True)
    assert r.returncode == 0, f"{mode} failed rc={r.returncode}"


def bits_equal(a, b):
    return (a.shape == b.shape and a.dtype == b.dtype
            and a.numpy().tobytes() == b.numpy().tobytes())


def main():
    if OUT.exists():
        import shutil
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)

    print("=== roundtrip: reference run (uninterrupted) ===", flush=True)
    run("reference")
    print("=== roundtrip: phase1 (train to N, checkpoint) ===", flush=True)
    run("phase1")
    print("=== roundtrip: phase2 (fresh process, restore, continue) ===",
          flush=True)
    run("phase2")

    ref = torch.load(OUT / "ref_traj.pt", weights_only=False)
    res = torch.load(OUT / "resume_traj.pt", weights_only=False)

    print("\n--- per-step comparison (fp64, full flat parameter vector) ---",
          flush=True)
    all_bit = True
    for i, (a, b) in enumerate(zip(ref["params"], res["params"])):
        be = bits_equal(a, b)
        mad = (a - b).abs().max().item()
        la, lb = ref["losses"][i], res["losses"][i]
        print(f"  step N+{i + 1:2d}: bitwise_equal={be}  max|dparam|={mad:.3e}"
              f"  loss_ref={la:.17g} loss_res={lb:.17g} "
              f"loss_equal={la == lb}", flush=True)
        all_bit = all_bit and be and (la == lb)
    check("trajectory-bit-continuous-10-steps", all_bit)
    check("same-number-of-steps",
          len(ref["params"]) == len(res["params"]) == 10,
          f"ref={len(ref['params'])} res={len(res['params'])}")

    check("filter-V-bitwise", bits_equal(ref["V"], res["V"]),
          f"shape ref={tuple(ref['V'].shape)} res={tuple(res['V'].shape)}")
    check("filter-S-bitwise", bits_equal(ref["S"], res["S"]),
          f"S_ref={ref['S'].tolist()}")
    check("filter-grad_mean-bitwise",
          bits_equal(ref["grad_mean"], res["grad_mean"]))
    check("filter-step_count", ref["step_count"] == res["step_count"],
          f"ref={ref['step_count']} res={res['step_count']}")

    print(f"\nROUNDTRIP RESULT: "
          f"{'ALL PASS' if not FAILURES else 'FAILURES: ' + str(FAILURES)}",
          flush=True)
    sys.exit(0 if not FAILURES else 1)


if __name__ == "__main__":
    main()
