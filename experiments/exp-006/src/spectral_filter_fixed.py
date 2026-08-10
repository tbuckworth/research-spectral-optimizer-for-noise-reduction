"""Numerically stable local variant of SpectralGradientFilter.

This research copy deliberately does not modify the optimizer repository.
It replaces only the streaming covariance update; projection and public API are
inherited from the repository implementation.
"""
from __future__ import annotations

import math

import torch

from spectral_filter import SpectralGradientFilter as _OriginalFilter


class StableSpectralGradientFilter(_OriginalFilter):
    """Streaming top-k covariance filter with stable symmetric updates.

    The old implementation recovered right singular vectors by division by
    small singular values.  Here C_new is represented directly in the
    orthonormal augmented basis [V, q] and diagonalised there.
    """

    def __init__(self, *args, relative_eig_tol=1e-8, absolute_eig_floor=0.0,
                 stabilize_every=100, **kwargs):
        super().__init__(*args, **kwargs)
        if relative_eig_tol < 0 or absolute_eig_floor < 0:
            raise ValueError("eigenvalue tolerances must be non-negative")
        if stabilize_every is not None and stabilize_every < 1:
            raise ValueError("stabilize_every must be positive or None")
        self.relative_eig_tol = float(relative_eig_tol)
        self.absolute_eig_floor = float(absolute_eig_floor)
        self.stabilize_every = stabilize_every
        self.stabilization_count = 0
        self.max_orthogonality_error = 0.0

    @staticmethod
    def _sym_eigh_desc(matrix_cpu_f64):
        """Eigenpairs of a small symmetric matrix, largest first, in fp64."""
        matrix_cpu_f64 = (matrix_cpu_f64 + matrix_cpu_f64.T) * 0.5
        vals, vecs = torch.linalg.eigh(matrix_cpu_f64)
        return vals.flip(0), vecs.flip(1)

    def _positive_mask(self, eigvals):
        if eigvals.numel() == 0:
            return torch.zeros_like(eigvals, dtype=torch.bool)
        largest = eigvals[0].clamp_min(0.0)
        floor = max(self.absolute_eig_floor,
                    self.relative_eig_tol * float(largest.item()))
        return eigvals > floor

    def _truncate(self, eigvals, eigvecs):
        keep = self._positive_mask(eigvals)
        eigvals = eigvals[keep]
        eigvecs = eigvecs[:, keep]
        if eigvals.numel() == 0:
            return eigvals, eigvecs
        new_k = min(self.rank, eigvals.numel())
        if self.adaptive == "none" and self.energy_threshold is not None:
            frac = torch.cumsum(eigvals, 0) / eigvals.sum()
            k_energy = int(torch.searchsorted(frac, self.energy_threshold).item()) + 1
            new_k = max(1, min(new_k, k_energy))
        return eigvals[:new_k], eigvecs[:, :new_k]

    def _repair_representation(self):
        """Reorthogonalise V while preserving V diag(S^2) V^T."""
        if self.V is None or self.V.shape[1] == 0:
            return
        q, r = torch.linalg.qr(self.V, mode="reduced")
        r64 = r.detach().double().cpu()
        s2 = self.S.detach().double().square()
        small = (r64 * s2.unsqueeze(0)) @ r64.T
        vals, rot = self._sym_eigh_desc(small)
        vals, rot = self._truncate(vals, rot)
        if vals.numel() == 0:
            self.V = None; self.S = None; self.proj_k = None
            return
        self.V = (q @ rot.to(device=q.device, dtype=q.dtype)).contiguous()
        self.S = vals.sqrt().cpu()
        self.stabilization_count += 1
        self._update_proj_k()

    def orthogonality_error(self):
        if self.V is None:
            return 0.0
        gram = self.V.T @ self.V
        eye = torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
        return float(torch.linalg.matrix_norm(gram - eye, ord=2).item())

    def _update_svd(self, g):
        device, dtype = g.device, g.dtype

        if self.grad_mean is None:
            self.grad_mean = g.detach().clone()
        else:
            self.grad_mean.mul_(self.decay).add_(g.detach(), alpha=1-self.decay)
        centered = g.detach() - self.grad_mean

        if self.V is None:
            norm = centered.norm()
            if norm > 0:
                self.V = (centered / norm).unsqueeze(1).contiguous()
                self.S = norm.detach().double().cpu().unsqueeze(0)
            return

        # Periodically repair the factorisation without changing its covariance.
        if self.stabilize_every and self.step_count % self.stabilize_every == 0:
            self._repair_representation()
            if self.V is None:
                return

        # Two-pass orthogonal residual is robust to small accumulated V error.
        c = self.V.T @ centered
        residual = centered - self.V @ c
        correction = self.V.T @ residual
        residual = residual - self.V @ correction
        c = c + correction
        residual_norm = residual.norm()
        has_residual = bool(residual_norm > max(1e-12, 10*torch.finfo(dtype).eps*centered.norm()))

        k = self.V.shape[1]
        c64 = c.detach().double().cpu()
        old_var = self.S.detach().double().square()
        d, w = float(self.decay), float(1-self.decay)
        size = k + int(has_residual)
        small = torch.zeros(size, size, dtype=torch.float64)
        small[:k, :k] = torch.diag(d * old_var) + w * torch.outer(c64, c64)

        q = None
        if has_residual:
            r64 = float(residual_norm.double().item())
            small[:k, k] = w * c64 * r64
            small[k, :k] = small[:k, k]
            small[k, k] = w * r64 * r64
            q = residual / residual_norm

        vals, rot = self._sym_eigh_desc(small)
        vals, rot = self._truncate(vals, rot)
        if vals.numel() == 0:
            self.V = None; self.S = None; self.proj_k = None
            return

        rot_dev = rot.to(device=device, dtype=dtype)
        updated = self.V @ rot_dev[:k]
        if has_residual:
            updated = updated + q.unsqueeze(1) * rot_dev[k].unsqueeze(0)
        self.V = updated.contiguous()
        self.S = vals.sqrt().cpu()
        self._update_proj_k()

        # Cheap exact checks only while rank is modest or at maintenance steps.
        if self.V.shape[1] <= 256 and (self.step_count <= 10 or self.step_count % 50 == 0):
            err = self.orthogonality_error()
            self.max_orthogonality_error = max(self.max_orthogonality_error, err)
            if not math.isfinite(err) or err > 5e-3:
                self._repair_representation()

    def _diagnostics(self):
        out = super()._diagnostics()
        out["stabilization_count"] = self.stabilization_count
        out["max_orthogonality_error"] = self.max_orthogonality_error
        out["relative_eig_tol"] = self.relative_eig_tol
        return out
