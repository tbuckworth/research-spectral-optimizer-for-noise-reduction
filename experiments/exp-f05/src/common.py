"""Shared helpers + PRE-DECLARED thresholds for the exp-f05 arm-C design sim.

All pass/fail thresholds are declared HERE, before any sim is run, so the
candidate table is judged against numbers fixed in advance (the plan gives
qualitative criteria; these are their numeric operationalizations, recorded
in out/arm-c-design.md as the proposed binding values for the fold jobs).
"""
import math

import torch

P_NOMINAL = 600_000          # parameter dimension the sim represents
TORCH_THREADS = 6            # leave headroom for concurrent exp-f0x agents

# ---------------------------------------------------------------------------
# Pre-declared thresholds (operationalizing plan criteria 1-7)
# ---------------------------------------------------------------------------
THRESH = {
    # 1. captured-energy fraction: must be far above the k/p noise floor
    "captured_energy_min_abs": 0.05,
    "captured_energy_min_rel": 10.0,      # also >= 10 * (k/p)
    # 2. realized norm-amplification factor to match B's kept-norm ratio
    "amplification_max": 3.0,             # (a)'s predicted 12-160x is the fail zone
    # 3. principal-angle rotation-rate match to B (mean angle per Delta steps)
    "rotation_ratio_lo": 0.5,
    "rotation_ratio_hi": 2.0,
    # 4. descent: norm-matched control loss within ~2x of arm A
    "descent_raw_max": 2.0,               # final loss ratio C/A
    "descent_excess_max": 2.0,            # (final - noise_floor) ratio C/A
    # 5. principal-angle floor between C's and B's realized bases (distinctness)
    "pa_floor_median_deg": 45.0,
    "overlap_cos2_max": 0.5,              # mean cos^2 overlap <= 0.5
    # 6. update-cosine ceiling between C's and B's filtered updates
    "update_cos_max": 0.7,
    # 7. positive-control-for-the-control (planted task)
    "planted_B_min": 0.9,                 # B must recover planted dir (cos)
    "planted_C_ratio_max": 0.6,           # C recovery <= 0.6 * B recovery
}


def qr_orth(V):
    """Orthonormalize columns (thin QR). Returns (d, k) with k = V.shape[1]."""
    Q, _ = torch.linalg.qr(V, mode="reduced")
    return Q


def principal_angles_deg(U, V):
    """Principal angles (degrees, ascending) between column spans of U, V.

    U, V: (d, k1), (d, k2) with orthonormal columns.
    """
    s = torch.linalg.svdvals(U.T @ V).clamp(0.0, 1.0)
    return torch.rad2deg(torch.acos(s)).flip(0)  # ascending svdvals -> descending


def angle_stats(U, V):
    ang = principal_angles_deg(U, V)
    s = torch.cos(torch.deg2rad(ang))
    return {
        "mean_deg": float(ang.mean()),
        "median_deg": float(ang.median()),
        "min_deg": float(ang.min()),
        "mean_cos2": float((s * s).mean()),
    }


def cosine(u, v):
    nu, nv = u.norm(), v.norm()
    if nu < 1e-20 or nv < 1e-20:
        return 0.0
    return float(torch.dot(u, v) / (nu * nv))


class _Dummy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(1))


def make_filter(rank, decay=0.99, warmup=100):
    """Real SpectralGradientFilter instance (hard top-k, normalize='none').

    Model/optimizer are dummies: part-1 drives _update_svd directly with the
    coordinate-space gradient stream (the real streaming-SVD code path).
    """
    from spectral_filter import SpectralGradientFilter
    m = _Dummy()
    opt = torch.optim.SGD(m.parameters(), lr=0.1)
    return SpectralGradientFilter(
        m, opt, rank=rank, decay=decay, warmup=warmup,
        weighting="hard", normalize="none", adaptive="none")


# ---------------------------------------------------------------------------
# SRHT-style exactly-orthogonal implicit rotation on R^pad  (candidate (a))
# R = D1 P1 H D2 P2 : signed perms + block fast Walsh-Hadamard, O(p log p)
# ---------------------------------------------------------------------------
class SRHTRotation:
    BLOCK = 8192

    def __init__(self, p, gen):
        self.p = p
        self.pad = ((p + self.BLOCK - 1) // self.BLOCK) * self.BLOCK
        self.perm1 = torch.randperm(self.pad, generator=gen)
        self.perm2 = torch.randperm(self.pad, generator=gen)
        self.iperm1 = torch.argsort(self.perm1)
        self.iperm2 = torch.argsort(self.perm2)
        self.d1 = (torch.randint(0, 2, (self.pad,), generator=gen) * 2 - 1).float()
        self.d2 = (torch.randint(0, 2, (self.pad,), generator=gen) * 2 - 1).float()

    @staticmethod
    def _fwht(x):
        """Normalized fast Walsh-Hadamard on last dim (power of 2). x: (B, n)."""
        n = x.shape[-1]
        h = 1
        while h < n:
            x = x.reshape(-1, n // (2 * h), 2, h)
            a = x[:, :, 0, :]
            b = x[:, :, 1, :]
            x = torch.stack((a + b, a - b), dim=2).reshape(-1, n)
            h *= 2
        return x / math.sqrt(n)

    def _hadamard(self, v):  # v: (B, pad)
        B = v.shape[0]
        return self._fwht(v.reshape(B * (self.pad // self.BLOCK), self.BLOCK)
                          ).reshape(B, self.pad)

    def _pad(self, v):  # (B, p) -> (B, pad)
        out = torch.zeros(v.shape[0], self.pad)
        out[:, :self.p] = v
        return out

    def apply(self, v):
        """R v for v: (B, p) -> (B, pad). Exactly orthogonal on R^pad."""
        t = self._pad(v)
        t = t[:, self.perm2] * self.d2
        t = self._hadamard(t)
        t = t[:, self.perm1] * self.d1
        return t

    def apply_T(self, v_pad):
        """R^T v for v_pad: (B, pad) -> (B, pad)."""
        t = v_pad * self.d1
        t = t[:, self.iperm1]
        t = self._hadamard(t)
        t = t * self.d2
        return t[:, self.iperm2]
