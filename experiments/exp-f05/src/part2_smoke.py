"""Quick smoke of part2_descent.py code paths (STEPS=120, WARMUP=40).

Purpose: exercise every arm's post-warmup branch (norm replay, steady-state
flags, rotation logging every 25 steps, Cb config parse) BEFORE the real run,
without writing part2's output JSON. Run with 2 threads so it does not
contend with the concurrently running part1 full sim.
"""
import sys

import torch

torch.set_num_threads(2)

import part2_descent as p2

p2.STEPS, p2.WARMUP = 120, 40

res = {}
res["A"] = p2.run_arm("A")
res["B_k8"] = p2.run_arm("B", k=8)
blog8 = {"ratio": res["B_k8"]["ratio_log"], "k": res["B_k8"]["k_log"]}
theta8 = None
if res["B_k8"]["rot_mean_deg_per25"] is not None:
    import math
    theta8 = math.radians(res["B_k8"]["rot_mean_deg_per25"] / 25.0)
res["Ca_k8"] = p2.run_arm("Ca", k=8, blog=blog8)
res["Cb_k8"] = p2.run_arm("Cb", k=8, r=32, blog=blog8, eta_b=0.005,
                          decay_b=0.999)
res["Cc_k8"] = p2.run_arm("Cc", k=8, blog=blog8, theta_per_step=theta8)

# config-parse check against the smoke-run part1 output naming
name = "k8_eta0.005_d0.999"
parts = name.split("_")
assert float(parts[1][3:]) == 0.005 and float(parts[2][1:]) == 0.999

for n, r in res.items():
    print(f"[smoke] {n:6s} final={r['final_loss']:.4f} "
          f"rot25={r.get('rot_mean_deg_per25')} "
          f"steady={r.get('n_steady_steps')}/{r.get('n_ctrl_steps')} "
          f"amp_med={r.get('amplification_median')}")
print("[smoke] all arm code paths executed OK")
