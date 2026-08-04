#!/usr/bin/env python3
"""exp-f02 — Component #5: SpectralGradientFilter integration checks.

LOCAL CPU ONLY (authorized light analysis; no GPU, no cluster).

Sub-checks (plan.md):
  (a) filtered training runs 200 steps, loss falls, all A8/B3/D1 diagnostics
      emitted and sane; all sweep knobs exercised at least once
  (b) alpha=0 soft identity vs plain AdamW, fp64, 50 steps, max|diff| <= 1e-12
      (STRUCTURAL failure here is run-level STOP-and-report)
  (c) planted-subspace: hard top-k recovers a planted dominant gradient
      direction (cosine > 0.9)
  (d) forced eigh failure -> CPU-fp64 fallback fires, is counted+logged,
      training continues
  (e) zero/constant predictor evaluates to ~0 numerai_corr on synthetic eras
  (+) RNG separation: arm B's extra draws come from a separate torch.Generator
      so data order at a fixed seed is identical across arms (demonstrated,
      with a negative control showing the failure mode)
"""

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

SRC = Path(__file__).resolve().parent
OUT = SRC.parent / "out"
OUT.mkdir(exist_ok=True)
sys.path.insert(0, str(SRC))

from spectral_filter import SpectralGradientFilter  # run's canonical copy
from diagnostics import FilterDiagnostics
from numerai_eval import per_era_corr

torch.use_deterministic_algorithms(True)
DEVICE = "cpu"  # compute profile: local CPU only for this component

RESULTS = {}


def make_mlp(n_feat=40, hidden=32, dtype=torch.float32, seed=0):
    torch.manual_seed(seed)
    model = nn.Sequential(
        nn.Linear(n_feat, hidden), nn.ReLU(), nn.Linear(hidden, 1),
    ).to(DEVICE).to(dtype)
    return model


def make_data(n_feat=40, n_eras=40, rows_per_era=128, seed=123,
              dtype=torch.float32):
    """Synthetic Numerai-shaped regression: low-SNR linear signal + per-era
    shift + noise, target centered at 0.5 and clipped to [0, 1]."""
    gen = torch.Generator().manual_seed(seed)
    n = n_eras * rows_per_era
    X = torch.randn(n, n_feat, generator=gen)
    w = torch.randn(n_feat, generator=gen)
    w = w / w.norm()
    eras = torch.arange(n_eras).repeat_interleave(rows_per_era)
    era_shift = 0.03 * torch.randn(n_eras, generator=gen)[eras]
    signal = 0.08 * (X @ w)
    noise = 0.15 * torch.randn(n, generator=gen)
    y = (0.5 + signal + era_shift + noise).clamp(0.0, 1.0)
    return (X.to(dtype), y.to(dtype).unsqueeze(1), eras)


# ===================================================================== (a)
def check_a():
    print("\n" + "=" * 72)
    print("(a) 200-step filtered training run + A8/B3/D1 diagnostics + knobs")
    print("=" * 72)
    t0 = time.time()
    X, y, eras = make_data(seed=123)
    n_train = 32 * 128
    Xtr, ytr = X[:n_train], y[:n_train]
    Xva, yva, eva = X[n_train:], y[n_train:], eras[n_train:]

    model = make_mlp(seed=0)
    base_opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    filt = SpectralGradientFilter(model, base_opt, rank=8, decay=0.99,
                                  warmup=20, weighting="hard",
                                  normalize="none")
    diag = FilterDiagnostics(filt, rotation_interval=10)
    data_gen = torch.Generator().manual_seed(7)
    lossfn = nn.MSELoss()
    losses = []
    for step in range(1, 201):
        idx = torch.randint(0, n_train, (128,), generator=data_gen)
        base_opt.zero_grad()
        loss = lossfn(model(Xtr[idx]), ytr[idx])
        loss.backward()
        diag.filter_grad()
        base_opt.step()
        losses.append(loss.item())
        if step % 25 == 0:
            with torch.no_grad():
                preds = model(Xva).squeeze(1).numpy()
            _, mean_corr = per_era_corr(preds, yva.squeeze(1).numpy(),
                                        eva.numpy())
            diag.log_valid_score(step, mean_corr)  # D1 hook
            print(f"  step {step:3d}  loss {loss.item():.5f}  "
                  f"valid numerai_corr {mean_corr:+.4f}")

    first20 = float(np.mean(losses[:20]))
    last20 = float(np.mean(losses[-20:]))
    loss_falls = last20 < first20
    ok_sane, problems = diag.sanity_check()
    summ = diag.summary()
    # required diagnostic fields present on active steps
    required = ["filtered_unfiltered_cosine", "kept_norm_fraction",
                "realized_k", "eigh_fallback_count"]
    active = [r for r in diag.records if r["filtering_active"]]
    fields_ok = all(k in r for r in active for k in required)
    rot_n = summ.get("n_rotation_measurements", 0)
    d1_ok = len(diag.valid_scores) == 8 and all(
        math.isfinite(v["valid_score"]) for v in diag.valid_scores)

    print(f"  loss first20 {first20:.5f} -> last20 {last20:.5f} "
          f"(falls: {loss_falls})")
    print(f"  diagnostics sane: {ok_sane} (problems: {problems[:5]})")
    print(f"  summary: {json.dumps(summ, indent=2)}")

    # --- knob smoke: every sweep knob reachable and exercised at least once
    knob_configs = [
        dict(rank=4, weighting="hard"),
        dict(rank=16, decay=0.999, warmup=5, weighting="hard"),
        dict(rank=8, weighting="soft", alpha=1.0, soft_residual=True),
        dict(rank=8, weighting="soft", alpha=0.5, soft_residual=False),
        dict(rank=16, weighting="hard", energy_threshold=0.90),
        dict(rank=16, weighting="hard", adaptive="effrank"),
        dict(rank=16, weighting="hard", adaptive="gap"),
    ]
    knob_ok = True
    knob_report = []
    for i, cfg in enumerate(knob_configs):
        m = make_mlp(seed=100 + i)
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
        f = SpectralGradientFilter(m, opt, **{"normalize": "none",
                                              "warmup": 10, **cfg})
        d = FilterDiagnostics(f, rotation_interval=10)
        g2 = torch.Generator().manual_seed(50 + i)
        final = None
        try:
            for _ in range(40):
                idx = torch.randint(0, n_train, (64,), generator=g2)
                opt.zero_grad()
                l = lossfn(m(Xtr[idx]), ytr[idx])
                l.backward()
                d.filter_grad()
                opt.step()
                final = l.item()
            s = d.summary()
            fin = math.isfinite(final)
            sane, _ = d.sanity_check()
            knob_report.append({**cfg, "final_loss": final,
                                "realized_k_last": s.get("realized_k_last"),
                                "ok": fin and sane})
            knob_ok &= fin and sane
        except Exception as e:
            knob_report.append({**cfg, "error": repr(e), "ok": False})
            knob_ok = False
    for r in knob_report:
        print(f"  knob cfg {r}")

    passed = (loss_falls and ok_sane and fields_ok and rot_n >= 5
              and d1_ok and knob_ok)
    RESULTS["a"] = {
        "pass": bool(passed), "loss_first20": first20, "loss_last20": last20,
        "diagnostics_sane": ok_sane, "sanity_problems": problems,
        "required_fields_present": fields_ok,
        "n_rotation_measurements": rot_n, "d1_series_ok": d1_ok,
        "d1_valid_scores": diag.valid_scores,
        "knob_smoke_ok": knob_ok, "knob_report": knob_report,
        "summary": summ, "seconds": time.time() - t0,
    }
    with open(OUT / "check_a_diagnostics.json", "w") as fh:
        json.dump({"records": diag.records,
                   "valid_scores": diag.valid_scores}, fh, indent=1)
    print(f"(a) {'PASS' if passed else 'FAIL'}  [{time.time()-t0:.1f}s]")


# ===================================================================== (b)
def train_traj_fp64(seed, use_filter, batches, n_steps=50):
    torch.set_default_dtype(torch.float64)
    try:
        model = make_mlp(dtype=torch.float64, seed=seed)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
        filt = None
        if use_filter:
            filt = SpectralGradientFilter(
                model, opt, rank=8, warmup=5, normalize="none",
                weighting="soft", alpha=0.0, soft_residual=True)
        lossfn = nn.MSELoss()
        for step in range(n_steps):
            Xb, yb = batches[step]
            opt.zero_grad()
            loss = lossfn(model(Xb), yb)
            loss.backward()
            if filt is not None:
                filt.filter_grad()
            opt.step()
        return [p.detach().clone() for p in model.parameters()]
    finally:
        torch.set_default_dtype(torch.float32)


def check_b():
    print("\n" + "=" * 72)
    print("(b) alpha=0 soft identity vs plain AdamW (fp64, 50 steps)")
    print("=" * 72)
    t0 = time.time()
    X, y, _ = make_data(seed=321, dtype=torch.float64)
    gen = torch.Generator().manual_seed(11)
    batches = []
    for _ in range(50):
        idx = torch.randint(0, len(X), (128,), generator=gen)
        batches.append((X[idx], y[idx]))

    plain = train_traj_fp64(seed=42, use_filter=False, batches=batches)
    filt = train_traj_fp64(seed=42, use_filter=True, batches=batches)
    max_diff = max(float((a - b).abs().max().item())
                   for a, b in zip(plain, filt))
    bit_identical = max_diff == 0.0
    passed = max_diff <= 1e-12
    print(f"  max abs param diff after 50 fp64 steps: {max_diff:.3e}")
    print(f"  bit-identical: {bit_identical}")
    RESULTS["b"] = {"pass": bool(passed), "max_abs_diff": max_diff,
                    "bit_identical": bool(bit_identical),
                    "structural_failure": bool(not passed),
                    "seconds": time.time() - t0}
    label = "PASS" if passed else ("FAIL — STRUCTURAL: alpha=0 soft "
                                   "weighting is NOT a no-op vs plain AdamW "
                                   "(STOP-and-report)")
    print(f"(b) {label}  [{time.time()-t0:.1f}s]")


# ===================================================================== (c)
def check_c():
    print("\n" + "=" * 72)
    print("(c) planted-subspace: hard top-k recovers planted direction")
    print("=" * 72)
    t0 = time.time()
    model = make_mlp(seed=5)
    p = sum(q.numel() for q in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    filt = SpectralGradientFilter(model, opt, rank=4, decay=0.99, warmup=10,
                                  weighting="hard", normalize="none")
    diag = FilterDiagnostics(filt, rotation_interval=25)

    plant_gen = torch.Generator().manual_seed(2024)
    u = torch.randn(p, generator=plant_gen)
    u = u / u.norm()
    noise_gen = torch.Generator().manual_seed(77)
    for _ in range(300):
        sign = 1.0 if torch.rand(1, generator=noise_gen).item() < 0.5 else -1.0
        g = sign * 3.0 * u + 0.1 * torch.randn(p, generator=noise_gen)
        filt._set_flat_grad(g)
        diag.filter_grad()
        # no opt.step(): gradient stream is exogenous; only the filter's
        # covariance estimate is under test
    cos_top = float((u @ filt.V[:, 0]).abs().item())

    # filtering-path probe: g = u + orthogonal unit vector; hard top-k
    # projection should keep mostly the u component
    v = torch.randn(p, generator=noise_gen)
    v = v - (v @ u) * u
    v = v / v.norm()
    probe = u + v
    filtered = filt._project_gradient(probe)
    cos_filtered_u = float(
        (filtered @ u / filtered.norm().clamp_min(1e-30)).item())
    kept_frac_probe = float((filtered.norm() / probe.norm()).item())

    passed = cos_top > 0.9
    print(f"  |cos(V[:,0], planted u)| = {cos_top:.6f}  (need > 0.9)")
    print(f"  probe u+v_orth: cos(filtered, u) = {cos_filtered_u:.4f}, "
          f"kept-norm fraction = {kept_frac_probe:.4f}")
    sane, problems = diag.sanity_check()
    print(f"  diagnostics sane during planted run: {sane}")
    RESULTS["c"] = {"pass": bool(passed and sane), "cos_top": cos_top,
                    "cos_filtered_vs_u": cos_filtered_u,
                    "probe_kept_norm_fraction": kept_frac_probe,
                    "diagnostics_sane": sane,
                    "seconds": time.time() - t0}
    print(f"(c) {'PASS' if passed and sane else 'FAIL'}  "
          f"[{time.time()-t0:.1f}s]")


# ===================================================================== (d)
def check_d():
    print("\n" + "=" * 72)
    print("(d) forced eigh failure -> CPU-fp64 fallback fires, training continues")
    print("=" * 72)
    t0 = time.time()
    X, y, _ = make_data(seed=555)
    model = make_mlp(seed=9)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    filt = SpectralGradientFilter(model, opt, rank=8, warmup=20,
                                  weighting="hard", normalize="none")
    diag = FilterDiagnostics(filt, rotation_interval=10)
    lossfn = nn.MSELoss()
    gen = torch.Generator().manual_seed(3)

    fail_steps = {30, 31, 60}
    state = {"step": 0}
    orig_eigh = torch.linalg.eigh

    def flaky_eigh(A, *args, **kwargs):
        # fail the fp32 attempt on chosen steps; the fallback's fp64 retry
        # passes through (mirrors Finding 5: fp32 eigh fails, fp64 succeeds)
        if A.dtype != torch.float64 and state["step"] in fail_steps:
            raise torch.linalg.LinAlgError(
                "forced eigh non-convergence (exp-f02 sub-check d)")
        return orig_eigh(A, *args, **kwargs)

    final_loss, err = None, None
    torch.linalg.eigh = flaky_eigh
    try:
        for step in range(1, 101):
            state["step"] = step
            idx = torch.randint(0, len(X), (128,), generator=gen)
            opt.zero_grad()
            loss = lossfn(model(X[idx]), y[idx])
            loss.backward()
            diag.filter_grad()
            opt.step()
            final_loss = loss.item()
    except Exception as e:
        err = repr(e)
    finally:
        torch.linalg.eigh = orig_eigh

    count = filt.eigh_fallback_count
    logged_in_diag = diag.records[-1]["eigh_fallback_count"] if diag.records \
        else None
    params_finite = all(torch.isfinite(q).all() for q in model.parameters())
    passed = (err is None and count == len(fail_steps)
              and final_loss is not None and math.isfinite(final_loss)
              and params_finite and logged_in_diag == count)
    print(f"  forced failures at steps {sorted(fail_steps)}")
    print(f"  eigh_fallback_count = {count} (expected {len(fail_steps)})")
    print(f"  training completed 100 steps: {err is None}, "
          f"final loss {final_loss}, params finite: {params_finite}")
    print(f"  count surfaced in diagnostics records: {logged_in_diag}")
    RESULTS["d"] = {"pass": bool(passed), "fallback_count": count,
                    "expected_count": len(fail_steps), "error": err,
                    "final_loss": final_loss,
                    "params_finite": bool(params_finite),
                    "count_in_diagnostics": logged_in_diag,
                    "seconds": time.time() - t0}
    print(f"(d) {'PASS' if passed else 'FAIL'}  [{time.time()-t0:.1f}s]")


# ===================================================================== (e)
def check_e():
    print("\n" + "=" * 72)
    print("(e) zero/constant predictor -> ~0 numerai_corr on synthetic eras")
    print("=" * 72)
    t0 = time.time()
    X, y, eras = make_data(seed=999)
    yn = y.squeeze(1).numpy()
    en = eras.numpy()

    zero_preds = np.zeros(len(yn))
    _, mean_zero = per_era_corr(zero_preds, yn, en)

    rng = np.random.default_rng(2026)  # seeded random predictor
    rand_preds = rng.standard_normal(len(yn))
    per, mean_rand = per_era_corr(rand_preds, yn, en)

    # signal predictor (positive control for the plumbing): preds = target
    _, mean_signal = per_era_corr(yn, yn, en)

    passed = (mean_zero == 0.0 and abs(mean_rand) < 0.06
              and mean_signal > 0.5)
    print(f"  zero predictor mean per-era corr:   {mean_zero:+.6f} (expect 0)")
    print(f"  random predictor mean per-era corr: {mean_rand:+.6f} "
          f"(expect |.| < 0.06; {len(per)} eras)")
    print(f"  target-as-preds positive control:   {mean_signal:+.4f} "
          f"(expect > 0.5)")
    RESULTS["e"] = {"pass": bool(passed), "mean_zero": mean_zero,
                    "mean_random": mean_rand, "mean_signal": mean_signal,
                    "n_eras": len(per), "seconds": time.time() - t0}
    print(f"(e) {'PASS' if passed else 'FAIL'}  [{time.time()-t0:.1f}s]")


# ================================================================= RNG sep
def rng_run(data_seed, arm_extra, arm_from_data_gen=False, n_steps=30):
    """Training-shaped loop that records the data index sequence. Arm B/C
    style extra RNG consumption is drawn from a SEPARATE generator (or, for
    the negative control, wrongly from the data generator)."""
    data_gen = torch.Generator().manual_seed(data_seed)
    arm_gen = torch.Generator().manual_seed(data_seed + 900001)
    seq = []
    for _ in range(n_steps):
        idx = torch.randint(0, 4096, (128,), generator=data_gen)
        seq.append(idx.clone())
        if arm_extra:
            g = data_gen if arm_from_data_gen else arm_gen
            _ = torch.randn(64, generator=g)  # basis init / subspace draw
    return seq


def check_rng():
    print("\n" + "=" * 72)
    print("(+) RNG separation: arm-B extra draws from a separate Generator")
    print("=" * 72)
    t0 = time.time()
    a = rng_run(1234, arm_extra=False)
    b = rng_run(1234, arm_extra=True, arm_from_data_gen=False)
    b_bad = rng_run(1234, arm_extra=True, arm_from_data_gen=True)
    same = all(torch.equal(x, y) for x, y in zip(a, b))
    bad_diverges = not all(torch.equal(x, y) for x, y in zip(a, b_bad))
    first_div = next((i for i, (x, y) in enumerate(zip(a, b_bad))
                      if not torch.equal(x, y)), None)
    passed = same and bad_diverges
    print(f"  arm A vs arm B (separate arm generator): data order identical "
          f"over 30 steps: {same}")
    print(f"  negative control (arm draws from the DATA generator): "
          f"diverges: {bad_diverges} (first divergent step: {first_div})")
    RESULTS["rng"] = {"pass": bool(passed), "paired_identical": bool(same),
                      "negative_control_diverges": bool(bad_diverges),
                      "first_divergent_step": first_div,
                      "seconds": time.time() - t0}
    print(f"(+) {'PASS' if passed else 'FAIL'}  [{time.time()-t0:.1f}s]")


def main():
    t0 = time.time()
    print(f"exp-f02 run_checks.py — torch {torch.__version__}, "
          f"numpy {np.__version__}, python {sys.version.split()[0]}, "
          f"device {DEVICE}")
    check_a()
    check_b()
    if RESULTS["b"]["structural_failure"]:
        print("\nSTRUCTURAL alpha=0 IDENTITY FAILURE — STOP-and-report per "
              "plan.md. Remaining checks skipped.")
    else:
        check_c()
        check_d()
        check_e()
        check_rng()

    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    overall = True
    for name in ["a", "b", "c", "d", "e", "rng"]:
        if name in RESULTS:
            p = RESULTS[name]["pass"]
            overall &= p
            print(f"  ({name}) {'PASS' if p else 'FAIL'}")
        else:
            overall = False
            print(f"  ({name}) NOT RUN")
    RESULTS["overall_pass"] = bool(overall)
    RESULTS["total_seconds"] = time.time() - t0
    RESULTS["env"] = {"torch": torch.__version__, "numpy": np.__version__,
                      "python": sys.version.split()[0], "device": DEVICE}
    with open(OUT / "summary.json", "w") as fh:
        json.dump(RESULTS, fh, indent=2)
    print(f"\nOVERALL: {'PASS' if overall else 'FAIL'} "
          f"({RESULTS['total_seconds']:.1f}s total)")


if __name__ == "__main__":
    main()
