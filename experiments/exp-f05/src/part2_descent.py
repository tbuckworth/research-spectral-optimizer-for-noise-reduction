"""Part 2 — descent check (criterion 4) + real-space verification, p = 590,337.

A real Numerai-shaped MLP (2304 features -> 256 hidden -> 1, p = 590,337
parameters) trained with AdamW on a synthetic low-SNR tabular regression
task (quantized 5-level features, 64 informative features with power-law
weights, signal var 0.5 / noise var 0.5). CPU only.

Arms (all same model init, same seeded data stream; control-specific RNG
from a separate generator so pairing semantics mirror the fold jobs):
  A      : plain AdamW.
  B      : real SpectralGradientFilter, hard top-k, normalize="none",
           k in {8, 128}; logs kept-ratio(t) + k(t) + basis-rotation rate.
  Ca     : candidate (a) — own rank-k tracker, basis rotated by an exact
           SRHT-style orthogonal R (signed perms + block Hadamard);
           norm-ratio replay of B's logged trajectory.
  Cb     : candidate (b) — rank-r tracker (r > k), random k-of-span
           projection; norm-ratio replay.  r = 4k at k=8; r = 2k at k=128
           (local CPU budget; part 1 measures r=4k at k=512 exactly).
  Cc     : candidate (c) — drifting random k-basis in R^p at B's measured
           rotation rate; norm-ratio replay.  k=8 (k=128 analytically = Ca).

Outputs: out/part2_descent.json
"""
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import TORCH_THREADS, SRHTRotation, qr_orth, angle_stats
from spectral_filter import SpectralGradientFilter

torch.set_num_threads(TORCH_THREADS)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "out")

N_FEAT, HIDDEN, BATCH, STEPS, WARMUP = 2304, 256, 256, 500, 100
LR, WD = 1e-3, 1e-4
NOISE_VAR = 0.5
DATA_SEED, MODEL_SEED, CTRL_SEED = 777, 0, 4242


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(N_FEAT, HIDDEN), nn.ReLU(),
                                 nn.Linear(HIDDEN, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def make_wstar():
    g = torch.Generator().manual_seed(11)
    w = torch.zeros(N_FEAT)
    mag = torch.arange(1, 65, dtype=torch.float32) ** (-0.6)
    idx = torch.randperm(N_FEAT, generator=g)[:64]
    w[idx] = mag * (torch.randint(0, 2, (64,), generator=g) * 2 - 1)
    w *= math.sqrt((1.0 - NOISE_VAR) / float((w * w).sum()))
    return w


WSTAR = make_wstar()


def batch(gen):
    X = torch.randint(0, 5, (BATCH, N_FEAT), generator=gen).float()
    X = (X - 2.0) / math.sqrt(2.0)
    y = X @ WSTAR + math.sqrt(NOISE_VAR) * torch.randn(BATCH, generator=gen)
    return X, y


def flat_grad(model):
    return torch.cat([p.grad.reshape(-1) for p in model.parameters()])


def set_flat_grad(model, flat):
    off = 0
    for p in model.parameters():
        n = p.numel()
        p.grad = flat[off:off + n].reshape(p.shape)
        off += n


def run_arm(kind, k=None, r=None, blog=None, theta_per_step=None, eta_b=0.0,
            decay_b=0.99):
    label = f"{kind}" + (f"_k{k}" if k else "") + (f"_r{r}" if r else "")
    t0 = time.time()
    torch.manual_seed(MODEL_SEED)
    model = MLP()
    p_total = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    data_gen = torch.Generator().manual_seed(DATA_SEED)
    ctrl_gen = torch.Generator().manual_seed(CTRL_SEED)

    filt = None
    if kind == "B" or kind == "Ca":
        filt = SpectralGradientFilter(model, opt, rank=k, warmup=WARMUP,
                                      weighting="hard", normalize="none")
    elif kind == "Cb":
        # decay_b: span-tracker decay chosen from part 1's rotation match (D2
        # larger-history-span variant); B itself stays at 0.99 (H7 pin).
        filt = SpectralGradientFilter(model, opt, rank=r, decay=decay_b,
                                      warmup=WARMUP,
                                      weighting="hard", normalize="none")
    rot = SRHTRotation(p_total, ctrl_gen) if kind == "Ca" else None
    G_b = torch.randn(r, k, generator=ctrl_gen) if kind == "Cb" else None
    Wb = None
    U_c = None
    if kind == "Cc":
        U_c = qr_orth(torch.randn(p_total, k, generator=ctrl_gen))

    losses, ratios, kts = [], [], []
    amps, ces, steadys = [], [], []      # aligned per C-measurement
    prevQ, rots = None, []               # rotation log (B and C arms)
    for step in range(1, STEPS + 1):
        X, y = batch(data_gen)
        opt.zero_grad()
        loss = nn.functional.mse_loss(model(X), y)
        loss.backward()
        losses.append(float(loss))

        if kind == "A":
            pass
        elif kind == "B":
            g = flat_grad(model)
            gn = float(g.norm())
            diag = filt.filter_grad()
            kts.append(int(diag.get("basis_rank", 0)))
            if step > WARMUP:
                ratios.append(float(flat_grad(model).norm()) / max(gn, 1e-30))
            else:
                ratios.append(None)
            if step > WARMUP and step % 25 == 0:
                Q = qr_orth(filt.V)
                if prevQ is not None and prevQ.shape[1] == Q.shape[1]:
                    rots.append(angle_stats(prevQ, Q)["mean_deg"])
                prevQ = Q
        else:
            g = flat_grad(model)
            gn = float(g.norm())
            if filt is not None:
                filt._update_svd(g)
            if kind == "Cc" and theta_per_step is not None:
                eta = theta_per_step / math.sqrt(p_total)
                U_c = qr_orth(U_c + eta * torch.randn(p_total, k,
                                                      generator=ctrl_gen))
            if step > WARMUP and blog["ratio"][step - 1] is not None:
                k_t = blog["k"][step - 1]
                if kind == "Ca":
                    V = filt.V
                    k_use = min(k_t, V.shape[1])
                    steady = V.shape[1] >= k
                    z = rot.apply_T(rot._pad(g[None]))[0, :p_total]
                    c = V[:, :k_use].T @ z
                    u_raw = rot.apply((V[:, :k_use] @ c)[None])[0, :p_total]
                elif kind == "Cb":
                    V = filt.V
                    r_cur = V.shape[1]
                    if Wb is None or Wb.shape[0] != r_cur:
                        Wb = qr_orth(G_b[:r_cur])
                    elif eta_b > 0.0:
                        Wb = qr_orth(Wb + eta_b * torch.randn(
                            r_cur, k, generator=ctrl_gen))
                    k_use = min(k_t, Wb.shape[1])
                    # steady only once the r-dim span tracker is FULL; during
                    # ramp-up C spans ~the whole tracked span (clone-like) and
                    # those steps must not enter the reported medians.
                    steady = (r_cur == r and Wb.shape[1] == k)
                    c = Wb[:, :k_use].T @ (V.T @ g)
                    u_raw = V @ (Wb[:, :k_use] @ c)
                else:  # Cc
                    k_use = min(k_t, U_c.shape[1])
                    steady = True
                    c = U_c[:, :k_use].T @ g
                    u_raw = U_c[:, :k_use] @ c
                ce = float(c.square().sum()) / max(gn * gn, 1e-30)
                ces.append(ce)
                steadys.append(steady)
                un = float(u_raw.norm())
                target = blog["ratio"][step - 1]
                amp = target / max(un / max(gn, 1e-30), 1e-30)
                amps.append(amp)
                u = u_raw * (target * gn / max(un, 1e-30))
                set_flat_grad(model, u)
            # C-arm realized-basis rotation rate (criterion 3, real p-space)
            if step > WARMUP and step % 25 == 0:
                if kind == "Ca":
                    # R is fixed orthogonal: angles(R V(t1), R V(t2)) ==
                    # angles(V(t1), V(t2)); use V directly.
                    Q = qr_orth(filt.V)
                elif kind == "Cb":
                    Q = (qr_orth(filt.V @ Wb[:, :k])
                         if Wb is not None else None)
                else:  # Cc
                    Q = U_c[:, :k].clone()
                if Q is not None:
                    if prevQ is not None and prevQ.shape[1] == Q.shape[1]:
                        rots.append(angle_stats(prevQ, Q)["mean_deg"])
                    prevQ = Q
        opt.step()
        if step % 100 == 0:
            print(f"  [{label}] step {step:4d} loss={np.mean(losses[-20:]):.4f} "
                  f"wall={time.time()-t0:6.1f}s", flush=True)

    final = float(np.mean(losses[-50:]))
    res = {"label": label, "kind": kind, "k": k, "r": r, "p": p_total,
           "final_loss": final, "excess_loss": final - NOISE_VAR,
           "loss_curve_every5": losses[::5],
           "wall_s": time.time() - t0}
    if kind == "B":
        res["ratio_log"] = ratios
        res["k_log"] = kts
        res["kept_ratio_median"] = float(np.median([x for x in ratios
                                                    if x is not None]))
    if rots:
        res["rot_mean_deg_per25"] = float(np.mean(rots))
    elif kind != "A":
        res["rot_mean_deg_per25"] = None
    if kind == "Cb":
        res["decay_b"], res["eta_b"] = decay_b, eta_b
    if amps:
        # steady-state medians (span tracker full); ramp-up steps are
        # clone-degenerate for Cb and excluded from the reported numbers
        st_amps = [a for a, s in zip(amps, steadys) if s]
        st_ces = [c for c, s in zip(ces, steadys) if s]
        use_amps, use_ces = (st_amps, st_ces) if st_amps else (amps, ces)
        res["n_ctrl_steps"] = len(amps)
        res["n_steady_steps"] = len(st_amps)
        res["steady_fallback_all"] = not bool(st_amps)
        res["amplification_median"] = float(np.median(use_amps))
        res["amplification_p90"] = float(np.percentile(use_amps, 90))
        res["captured_energy_median"] = float(np.median(use_ces))
    print(f"  [{label}] final_loss={final:.4f} excess={final-NOISE_VAR:.4f} "
          f"wall={time.time()-t0:.1f}s")
    return res


def select_cb_config(part1, k_part1):
    """Pick the part-1 Cb (eta, decay) whose realized rotation rate best
    matches B's at the same k (min |log ratio|). Returns (eta, dec, name)."""
    b_rot = (part1["B"][str(k_part1)].get("rot") or {}).get("mean_deg")
    best, best_score = None, None
    for name, rec in part1["Cb"].items():
        if rec is None or not name.startswith(f"k{k_part1}_"):
            continue
        rot = (rec.get("rot") or {}).get("mean_deg")
        if rot is None or b_rot is None or b_rot <= 0:
            continue
        score = abs(math.log(max(rot, 1e-9) / b_rot))
        if best_score is None or score < best_score:
            best, best_score = name, score
    if best is None:
        return 0.0, 0.99, None
    parts = best.split("_")            # "k8_eta0.005_d0.999"
    return float(parts[1][3:]), float(parts[2][1:]), best


def main():
    print(f"[part2] MLP {N_FEAT}-{HIDDEN}-1, p=590337 expected; "
          f"batch={BATCH} steps={STEPS} warmup={WARMUP} lr={LR} wd={WD} "
          f"signal/noise var = {1-NOISE_VAR}/{NOISE_VAR}")
    p1_path = os.path.join(OUT, "part1_main.json")
    eta8, dec8, eta128, dec128 = 0.0, 0.99, 0.0, 0.99
    if os.path.exists(p1_path):
        with open(p1_path) as fh:
            part1 = json.load(fh)
        eta8, dec8, n8 = select_cb_config(part1, 8)
        eta128, dec128, n512 = select_cb_config(part1, 512)
        print(f"[part2] Cb config from part1 rotation match: "
              f"k8 -> {n8} (eta={eta8}, decay={dec8}); "
              f"k128 -> from k512 winner {n512} (eta={eta128}, decay={dec128})")
    else:
        print("[part2] WARNING: part1_main.json not found; Cb defaults "
              "eta=0, decay=0.99")
    # SRHT sanity check
    gen = torch.Generator().manual_seed(1)
    rt = SRHTRotation(590337, gen)
    v = torch.randn(1, 590337)
    Rv = rt.apply(v)
    err_norm = abs(float(Rv.norm()) - float(v.norm())) / float(v.norm())
    back = rt.apply_T(Rv)[0, :590337]
    err_inv = float((back - v[0]).norm()) / float(v.norm())
    print(f"[part2] SRHT sanity: |norm err|={err_norm:.2e}, "
          f"|R^T R v - v|/|v|={err_inv:.2e}")

    results = {}
    results["A"] = run_arm("A")
    results["B_k8"] = run_arm("B", k=8)
    results["B_k128"] = run_arm("B", k=128)
    blog8 = {"ratio": results["B_k8"]["ratio_log"], "k": results["B_k8"]["k_log"]}
    blog128 = {"ratio": results["B_k128"]["ratio_log"],
               "k": results["B_k128"]["k_log"]}
    theta8 = None
    if results["B_k8"]["rot_mean_deg_per25"] is not None:
        theta8 = math.radians(results["B_k8"]["rot_mean_deg_per25"] / 25.0)
    results["Ca_k8"] = run_arm("Ca", k=8, blog=blog8)
    results["Cb_k8"] = run_arm("Cb", k=8, r=32, blog=blog8,
                               eta_b=eta8, decay_b=dec8)
    results["Cc_k8"] = run_arm("Cc", k=8, blog=blog8, theta_per_step=theta8)
    results["Ca_k128"] = run_arm("Ca", k=128, blog=blog128)
    results["Cb_k128"] = run_arm("Cb", k=128, r=256, blog=blog128,
                                 eta_b=eta128, decay_b=dec128)

    a_final, a_excess = results["A"]["final_loss"], results["A"]["excess_loss"]
    print("\n[part2] descent summary (vs arm A "
          f"final={a_final:.4f} excess={a_excess:.4f}, floor={NOISE_VAR}):")
    for name, r in results.items():
        if name == "A":
            continue
        rr = r["final_loss"] / a_final
        re = r["excess_loss"] / max(a_excess, 1e-9)
        amp = r.get("amplification_median")
        ce = r.get("captured_energy_median")
        rot = r.get("rot_mean_deg_per25")
        print(f"  {name:9s} final={r['final_loss']:.4f} raw_ratio={rr:5.2f} "
              f"excess_ratio={re:6.2f} "
              f"amp={amp if amp is None else f'{amp:8.2f}'} "
              f"ce={ce if ce is None else f'{ce:.6f}'} "
              f"rot25={rot if rot is None else f'{rot:6.2f}'} "
              f"steady={r.get('n_steady_steps')}/{r.get('n_ctrl_steps')}")
    with open(os.path.join(OUT, "part2_descent.json"), "w") as fh:
        json.dump(results, fh, indent=1)
    print("[part2] wrote out/part2_descent.json")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"[part2] total wall {time.time()-t0:.1f}s")
