"""Part 1 — arm-C candidate invariant measurement at p = 600,000 (exact).

Key device: SpectralGradientFilter is exactly equivariant under a fixed
orthogonal change of basis (every operation it performs — norms, inner
products, linear combinations of observed vectors, the small Gram eigh — is
basis-independent). A synthetic gradient stream

    g_t = U_sig(t) b_t + eps_t,   eps_t ~ N(0, sigma^2 I_p),  p = 600,000

therefore drives the REAL filter code identically whether represented in
R^p or in the incremental orthonormal coordinate system of the stream's own
span. The signal lives in a fixed r_amb-dim coordinate block (with a slowly
rotating r_true-dim frame inside it); the i.i.d. noise coordinates follow
the exact Bartlett/Wishart decomposition: N(0, sigma^2) on every already-
spanned coordinate plus a new-direction magnitude sigma * sqrt(chi2_df),
df = p - (#dims already spanned). Nothing is approximated: this IS the
p = 600k stream, in coordinates. Candidate (b) (random k-of-r within the
tracked span) is measured directly here. Candidates (a)/(c) place their
subspace Haar-uniformly in R^p, which leaves the stream's span — their
captured energy k/p and ~90 deg angles are exact expectations (verified
empirically in real p-space in part 2) and are tabulated in finalize.py.

Outputs: out/part1_main.json, out/part1_planted.json
"""
import json
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (P_NOMINAL, TORCH_THREADS, qr_orth, angle_stats, cosine,
                    make_filter)

torch.set_num_threads(TORCH_THREADS)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "out")


class CoordStream:
    """Exact coordinate-space representation of the p=600k gradient stream."""

    def __init__(self, p, r_amb, r_true, T, sig_energy, noise_energy,
                 drift, powerlaw, seed, planted=False):
        self.p, self.r_amb, self.T = p, r_amb, T
        self.d = r_amb + T
        self.t = 0
        self.rng = np.random.default_rng(seed)
        self.gen = torch.Generator().manual_seed(seed)
        self.drift = drift
        self.planted = planted
        self.sigma = math.sqrt(noise_energy / p)
        if planted:
            # one dominant direction (mean 3, sd 1.5 amplitude) + 16 weak dirs
            self.r_true = 17
            lam = np.full(17, 0.05)
            self.sqrt_lam = torch.tensor(np.sqrt(lam), dtype=torch.float32)
            self.M = torch.eye(r_amb, 17)     # fixed frame; e0 = planted dir
        else:
            self.r_true = r_true
            lam = np.arange(1, r_true + 1, dtype=np.float64) ** (-powerlaw)
            lam = lam / lam.sum() * sig_energy
            self.sqrt_lam = torch.tensor(np.sqrt(lam), dtype=torch.float32)
            M = torch.randn(r_amb, r_true, generator=self.gen)
            self.M = qr_orth(M)

    def next(self):
        t = self.t
        d, r_amb = self.d, self.r_amb
        g = torch.zeros(d)
        # --- signal ---
        if not self.planted and self.drift > 0:
            self.M = qr_orth(self.M + self.drift *
                             torch.randn(r_amb, self.r_true, generator=self.gen))
        b = self.sqrt_lam * torch.randn(self.r_true, generator=self.gen)
        if self.planted:
            b[0] = 3.0 + 1.5 * float(torch.randn(1, generator=self.gen))
        g[:r_amb] = self.M @ b
        # --- exact isotropic noise in R^p, Bartlett coordinates ---
        n_old = r_amb + t                       # coords already spanned
        g[:n_old] += self.sigma * torch.tensor(
            self.rng.standard_normal(n_old), dtype=torch.float32)
        df = self.p - n_old
        g[n_old] = self.sigma * math.sqrt(self.rng.chisquare(df))
        self.t += 1
        return g


class CandidateB:
    """Candidate (b): random k-dim subspace of the rank-r tracked span.

    W = orth of the first-r rows of a FIXED Gaussian (deterministic given r);
    optional isotropic drift of W within the span at rate eta per step
    (the knob that tunes C's rotation rate to match B's).
    """

    def __init__(self, r_max, k, eta, gen):
        self.k, self.eta, self.gen = k, eta, gen
        self.G = torch.randn(r_max, k, generator=gen)
        self.W = None           # drift state, created once span is full

    def basis_coeffs(self, r_cur, k_cur):
        """(r_cur, k_cur) orthonormal coefficient matrix within the span."""
        if self.eta == 0.0 or self.W is None or self.W.shape[0] != r_cur:
            W = qr_orth(self.G[:r_cur])
            if self.eta > 0.0:
                self.W = W
        else:
            W = self.W
        return W[:, :k_cur]

    def step_drift(self, r_cur):
        if self.eta > 0.0 and r_cur == self.G.shape[0]:
            if self.W is None or self.W.shape[0] != r_cur:
                self.W = qr_orth(self.G[:r_cur])
            self.W = qr_orth(self.W + self.eta *
                             torch.randn(r_cur, self.k, generator=self.gen))


def run_main(T=2400, eval_every=25, seed=0):
    print(f"[part1/main] T={T} stream p={P_NOMINAL} r_true=64 r_amb=128 "
          f"sig/noise energy = 1.0/1.0 (SNR 1:1), B decay=0.99, "
          f"C span-tracker decay in {{0.99, 0.999}}")
    stream = CoordStream(P_NOMINAL, r_amb=128, r_true=64, T=T,
                         sig_energy=1.0, noise_energy=1.0,
                         drift=0.01, powerlaw=1.2, seed=seed)
    ctrl_gen = torch.Generator().manual_seed(999)   # separate RNG for controls

    ks = [8, 512, 2048]
    spans = {8: 32, 512: 2048}                      # r = 4k; r=8192 infeasible
    etas = [0.0, 0.005, 0.02, 0.1]
    decays = [0.99, 0.999]

    filters = {k: make_filter(k) for k in ks}       # B arms (decay 0.99, H7)
    # C span trackers: (r, decay). The T=620 smoke showed Cb at decay 0.99
    # rotates ~3.8x FASTER than B (span churn dominates); decay 0.999 = a
    # strictly larger history span (D2) is the slow-rotation variant, with
    # eta as the add-rotation knob.
    span_f = {(32, 0.99): make_filter(32),
              (32, 0.999): make_filter(32, decay=0.999),
              (2048, 0.99): filters[2048],          # shared with B2048
              (2048, 0.999): make_filter(2048, decay=0.999)}

    cands = {(k, eta, dec): CandidateB(spans[k], k, eta, ctrl_gen)
             for k in spans for eta in etas for dec in decays}

    # unique filter instances (the (2048, 0.99) span tracker IS B2048)
    all_filters = list({id(f): f for f in
                        list(filters.values()) + list(span_f.values())}.values())

    prev = {}          # name -> previous orthonormal basis (for rotation rate)
    recs = {}          # name -> list of per-eval metric dicts
    g2_sum, g2_n = 0.0, 0
    t0 = time.time()
    for t in range(T):
        g = stream.next()
        for f in all_filters:
            f._update_svd(g)
        for (k, eta, dec), cb in cands.items():
            sf = span_f[(spans[k], dec)]
            if sf.V is not None:
                cb.step_drift(sf.V.shape[1])
        g2_sum += float(g.dot(g)); g2_n += 1

        if t % eval_every != 0 or t < 200:
            continue
        gnorm2 = float(g.dot(g))
        bases = {}
        for k in ks:
            V = filters[k].V
            bases[k] = qr_orth(V) if V is not None else None
        span_bases = {}
        for key, sf in span_f.items():
            if key == (2048, 0.99):
                span_bases[key] = bases[2048]
            else:
                span_bases[key] = qr_orth(sf.V) if sf.V is not None else None
        for k in ks:
            Qb = bases[k]
            name = f"B{k}"
            ce = float((Qb.T @ g).square().sum()) / gnorm2
            rec = {"t": t, "k_real": Qb.shape[1], "captured_energy": ce,
                   "kept_ratio": math.sqrt(ce)}
            if name in prev and prev[name].shape[1] == Qb.shape[1]:
                rec["rot"] = angle_stats(prev[name], Qb)
            prev[name] = Qb
            recs.setdefault(name, []).append(rec)
        for (k, eta, dec), cb in cands.items():
            Qs = span_bases[(spans[k], dec)]
            if Qs is None or Qs.shape[1] < spans[k]:
                continue                       # only measure with span full
            Qb = bases[k]
            k_cur = Qb.shape[1]
            W = cb.basis_coeffs(Qs.shape[1], k_cur)
            Uc = Qs @ W
            name = f"Cb_k{k}_eta{eta}_d{dec}"
            cs = W.T @ (Qs.T @ g)
            ce_c = float(cs.square().sum()) / gnorm2
            ce_b = float((Qb.T @ g).square().sum()) / gnorm2
            u_c = Uc @ cs
            u_b = Qb @ (Qb.T @ g)
            rec = {"t": t, "k": k_cur, "r": Qs.shape[1],
                   "captured_energy": ce_c,
                   "amplification": math.sqrt(ce_b / max(ce_c, 1e-30)),
                   "pa_vs_B": angle_stats(Uc, Qb),
                   "update_cos": abs(cosine(u_c, u_b))}
            if name in prev and prev[name].shape[1] == k_cur:
                rec["rot"] = angle_stats(prev[name], Uc)
            prev[name] = Uc
            recs.setdefault(name, []).append(rec)
        if t % 200 == 0:
            ks_real = {k: (filters[k].V.shape[1] if filters[k].V is not None else 0)
                       for k in ks}
            print(f"  t={t:5d}  k_real={ks_real}  wall={time.time()-t0:7.1f}s",
                  flush=True)

    print(f"[part1/main] loop done in {time.time()-t0:.1f}s; "
          f"E||g||^2 measured = {g2_sum/g2_n:.4f} (expected ~2.0)")

    # aggregate over the post-fill window (last 6 evals for late-filling spans)
    def agg(name, last_n=None):
        rows = recs.get(name, [])
        if last_n:
            rows = rows[-last_n:]
        if not rows:
            return None
        out = {"n_evals": len(rows), "t_first": rows[0]["t"], "t_last": rows[-1]["t"]}
        for key in ["captured_energy", "amplification", "kept_ratio", "update_cos"]:
            vals = [r[key] for r in rows if key in r]
            if vals:
                out[key + "_median"] = float(np.median(vals))
                out[key + "_mean"] = float(np.mean(vals))
        for key in ["rot", "pa_vs_B"]:
            subs = [r[key] for r in rows if key in r]
            if subs:
                out[key] = {kk: float(np.median([s[kk] for s in subs]))
                            for kk in subs[0]}
        out["k_real_last"] = rows[-1].get("k_real", rows[-1].get("k"))
        return out

    result = {"T": T, "eval_every": eval_every, "p": P_NOMINAL,
              "mean_g2": g2_sum / g2_n,
              "B": {str(k): agg(f"B{k}", last_n=(12 if k == 2048 else 20))
                    for k in ks},
              "Cb": {f"k{k}_eta{eta}_d{dec}": agg(f"Cb_k{k}_eta{eta}_d{dec}",
                                                  last_n=(12 if k == 512 else 20))
                     for k in spans for eta in etas for dec in decays},
              "note_k2048": ("candidate (b) at k=2048 requires span r>2048 "
                             "(r=4k=8192): tracker cost scales as r^2 per step "
                             "and r=k is a degenerate clone (C spans exactly "
                             "B's basis span); measured entries cover k in "
                             "{8,512}; see design doc for the k>512 rule.")}
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "part1_main.json"), "w") as fh:
        json.dump(result, fh, indent=1)
    print("[part1/main] wrote out/part1_main.json")
    return result


def run_planted(T=600, seed=1):
    """Positive-control-for-the-control: planted dominant direction stream.

    B (hard top-k, k=8) must recover the planted direction (cos > 0.9);
    an acceptable C must NOT (criterion 7 / D2).
    """
    print(f"[part1/planted] T={T} planted dominant dir, amp 3.0+-1.5, "
          f"16 weak dirs, noise energy 1.0")
    stream = CoordStream(P_NOMINAL, r_amb=32, r_true=17, T=T,
                         sig_energy=None, noise_energy=1.0,
                         drift=0.0, powerlaw=None, seed=seed, planted=True)
    ctrl_gen = torch.Generator().manual_seed(998)
    fB = make_filter(8)
    fS = make_filter(32)                      # span tracker, decay 0.99
    fS9 = make_filter(32, decay=0.999)        # span tracker, decay 0.999
    cb = CandidateB(32, 8, 0.0, ctrl_gen)
    cb9 = CandidateB(32, 8, 0.0, ctrl_gen)
    e0 = torch.zeros(stream.d); e0[0] = 1.0

    rows = []
    for t in range(T):
        g = stream.next()
        fB._update_svd(g)
        fS._update_svd(g)
        fS9._update_svd(g)
        if t >= 200 and t % 20 == 0:
            Qb = qr_orth(fB.V)
            cos_top1 = abs(float(fB.V[:, 0].dot(e0)) / float(fB.V[:, 0].norm()))
            rec_B = float((Qb.T @ e0).norm())
            row = {"t": t, "B_top1_cos": cos_top1, "B_proj": rec_B}
            if fS.V is not None and fS.V.shape[1] == 32:
                Qs = qr_orth(fS.V)
                W = cb.basis_coeffs(32, 8)
                row["Cb_proj"] = float((W.T @ (Qs.T @ e0)).norm())
            if fS9.V is not None and fS9.V.shape[1] == 32:
                Qs9 = qr_orth(fS9.V)
                W9 = cb9.basis_coeffs(32, 8)
                row["Cb_proj_d999"] = float((W9.T @ (Qs9.T @ e0)).norm())
            rows.append(row)
    res = {
        "B_top1_cos_median": float(np.median([r["B_top1_cos"] for r in rows])),
        "B_proj_median": float(np.median([r["B_proj"] for r in rows])),
        "Cb_proj_median": float(np.median([r["Cb_proj"] for r in rows
                                           if "Cb_proj" in r])),
        "Cb_proj_d999_median": float(np.median([r["Cb_proj_d999"] for r in rows
                                                if "Cb_proj_d999" in r])),
        "Ca_proj_analytic": math.sqrt(8 / P_NOMINAL),
        "Cc_proj_analytic": math.sqrt(8 / P_NOMINAL),
        "note": ("Ca/Cc place their k-subspace Haar-uniformly in R^p: "
                 "E||P u*||^2 = k/p exactly; scaling for k in {512,2048}: "
                 "sqrt(k/p) = 0.029 / 0.058. Cb scaling: sqrt(k/r)=0.5 at r=4k.")}
    print(f"[part1/planted] B top1 cos={res['B_top1_cos_median']:.4f} "
          f"B proj={res['B_proj_median']:.4f} Cb proj={res['Cb_proj_median']:.4f} "
          f"Ca analytic={res['Ca_proj_analytic']:.5f}")
    with open(os.path.join(OUT, "part1_planted.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    print("[part1/planted] wrote out/part1_planted.json")
    return res


if __name__ == "__main__":
    torch.manual_seed(0)
    t_all = time.time()
    run_main()
    run_planted()
    print(f"[part1] total wall {time.time()-t_all:.1f}s")
