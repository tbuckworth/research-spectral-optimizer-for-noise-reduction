#!/usr/bin/env python3
"""A8/B3/D1 diagnostics for SpectralGradientFilter runs (exp-f02 deliverable).

Importable module reused by every later arm-B run in this run
(2026-08-03-followup ... p×p filter walk-forward redo). Per success-criteria.md
"Verification depth" and decomposition Component 5 / amendments B3 and D1:

  (i)   per-step filtered-vs-unfiltered gradient cosine        (A8)
  (ii)  per-step kept-norm fraction ||g_filtered|| / ||g||     (A8)
  (iii) realized k(t) (kept_rank / proj_k trajectory)          (A2/A8)
  (iv)  basis-rotation rate: principal angles between the realized retained
        basis at step t and step t-Δ                           (B3)
  (v)   VALID-score-vs-step logging hook                       (D1)

Usage (drop-in around the standard filter call):

    filt = SpectralGradientFilter(model, base_opt, rank=k, normalize="none")
    diag = FilterDiagnostics(filt, rotation_interval=10)

    loss.backward()
    rec = diag.filter_grad()      # instead of filt.filter_grad()
    base_opt.step()
    ...
    diag.log_valid_score(step, valid_corr)   # D1 hook, whenever VALID is scored

`diag.records` holds one dict per step; `diag.valid_scores` holds the D1
series; `diag.summary()` gives run-level aggregates including the
distance-from-identity numbers the per-fold report consumes.
"""

import math

import torch


def principal_angles(V1, V2):
    """Principal angles (radians, ascending) between the column spans of V1
    (p, k1) and V2 (p, k2). Computed in fp64 with re-orthonormalization for
    robustness; returns a tensor of min(k1, k2) angles in [0, pi/2]."""
    Q1, _ = torch.linalg.qr(V1.detach().double())
    Q2, _ = torch.linalg.qr(V2.detach().double())
    s = torch.linalg.svdvals(Q1.T @ Q2)
    s = s.clamp(0.0, 1.0)
    return torch.arccos(s).flip(0).sort().values


class FilterDiagnostics:
    """Wraps a SpectralGradientFilter and emits the A8/B3/D1 diagnostics."""

    def __init__(self, filt, rotation_interval=10):
        self.filt = filt
        self.rotation_interval = int(rotation_interval)
        self._prev_basis = None
        self._prev_basis_step = None
        self.records = []       # one dict per filter_grad() call
        self.valid_scores = []  # D1: [{"step": s, "valid_score": v}, ...]

    # ------------------------------------------------------------------ core
    def filter_grad(self):
        """Call between loss.backward() and base_opt.step() in place of
        filt.filter_grad(). Returns the per-step diagnostics record."""
        g_before = self.filt._get_flat_grad().detach().clone()
        rec = dict(self.filt.filter_grad())  # filter's own diagnostics
        g_after = self.filt._get_flat_grad().detach()

        nb = g_before.norm()
        na = g_after.norm()
        denom = (nb * na).clamp_min(1e-30)
        rec["filtered_unfiltered_cosine"] = float(
            (g_before.double() @ g_after.double()
             / denom.double()).item())
        rec["kept_norm_fraction"] = float(
            (na / nb.clamp_min(1e-30)).item())
        # realized k(t): directions actually projected onto this step
        rec["realized_k"] = int(rec.get("kept_rank", 0) or 0)
        rec["eigh_fallback_count"] = int(
            getattr(self.filt, "eigh_fallback_count", 0))

        # (iv) basis-rotation rate every `rotation_interval` steps
        step = int(rec["step"])
        rec["rotation_mean_angle"] = None
        rec["rotation_max_angle"] = None
        rec["rotation_delta_steps"] = None
        basis = self._realized_basis()
        if basis is not None and step % self.rotation_interval == 0:
            if self._prev_basis is not None:
                ang = principal_angles(self._prev_basis, basis)
                rec["rotation_mean_angle"] = float(ang.mean().item())
                rec["rotation_max_angle"] = float(ang.max().item())
                rec["rotation_delta_steps"] = step - self._prev_basis_step
            self._prev_basis = basis.detach().clone()
            self._prev_basis_step = step

        self.records.append(rec)
        return rec

    def _realized_basis(self):
        V = self.filt.V
        if V is None:
            return None
        k = self.filt.proj_k
        return V if k is None else V[:, :k]

    # ------------------------------------------------------------------- D1
    def log_valid_score(self, step, score):
        """D1 hook: VALID-score-vs-step series (feeds the 4th
        under-exploration signature: still-improving-at-cutoff)."""
        self.valid_scores.append(
            {"step": int(step), "valid_score": float(score)})

    # -------------------------------------------------------------- summary
    def summary(self):
        """Run-level aggregates, incl. distance-from-identity (A8)."""
        act = [r for r in self.records if r.get("filtering_active")]
        cos = [r["filtered_unfiltered_cosine"] for r in act]
        knf = [r["kept_norm_fraction"] for r in act]
        ks = [r["realized_k"] for r in act]
        angs = [r["rotation_mean_angle"] for r in self.records
                if r["rotation_mean_angle"] is not None]
        out = {
            "n_steps": len(self.records),
            "n_active_steps": len(act),
            "eigh_fallback_count": int(
                getattr(self.filt, "eigh_fallback_count", 0)),
            "n_valid_scores": len(self.valid_scores),
        }
        if act:
            out.update({
                "mean_cosine": sum(cos) / len(cos),
                "min_cosine": min(cos),
                "mean_kept_norm_fraction": sum(knf) / len(knf),
                "min_kept_norm_fraction": min(knf),
                "max_kept_norm_fraction": max(knf),
                # distance-from-identity: 1 - cosine, averaged over active steps
                "mean_distance_from_identity": 1.0 - sum(cos) / len(cos),
                "realized_k_first": ks[0],
                "realized_k_last": ks[-1],
                "realized_k_min": min(ks),
                "realized_k_max": max(ks),
            })
        if angs:
            out.update({
                "n_rotation_measurements": len(angs),
                "mean_rotation_angle": sum(angs) / len(angs),
                "max_rotation_angle": max(angs),
            })
        return out

    def sanity_check(self):
        """Assert the emitted values are structurally sane (exp-f02 spec):
        cosines finite in [-1, 1]; kept-norm fractions in [0, 1] (+eps);
        principal angles in [0, pi/2]. Returns (ok, problems)."""
        problems = []
        eps = 1e-4
        for r in self.records:
            c = r["filtered_unfiltered_cosine"]
            f = r["kept_norm_fraction"]
            if not math.isfinite(c) or c < -1 - eps or c > 1 + eps:
                problems.append(f"step {r['step']}: cosine {c}")
            if not math.isfinite(f) or f < -eps or f > 1 + eps:
                problems.append(f"step {r['step']}: kept_norm_fraction {f}")
            a = r.get("rotation_mean_angle")
            m = r.get("rotation_max_angle")
            for name, v in (("mean_angle", a), ("max_angle", m)):
                if v is not None and (not math.isfinite(v)
                                      or v < -eps or v > math.pi / 2 + eps):
                    problems.append(f"step {r['step']}: {name} {v}")
        for rec in self.valid_scores:
            if not math.isfinite(rec["valid_score"]):
                problems.append(f"valid score at step {rec['step']} not finite")
        return (len(problems) == 0, problems)
