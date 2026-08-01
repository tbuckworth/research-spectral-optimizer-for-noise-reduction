#!/usr/bin/env python3
"""Weight-covariance spectral optimizer (v2 — batch-mean streaming).

Tracks a running low-rank estimate of the p×p weight-gradient covariance
using streaming rank-1 SVD updates from batch-mean gradients. Every step:
1. Normal forward/backward → batch-mean gradient g ∈ R^p
2. Rank-1 SVD update with g (trivially cheap)
3. Project g onto top-k eigenspace
4. Pass projected gradient to base optimizer

No per-sample gradients needed. Barely slower than standard Adam.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightCovarianceFilterV2:
    def __init__(self, model, base_optimizer, rank=200, decay=0.99,
                 warmup=100, filter_strength=1.0, energy_threshold=None,
                 adaptive="none", normalize="none",
                 weighting="hard", alpha=1.0, soft_residual=True):
        self.model = model
        self.base_optimizer = base_optimizer
        self.rank = rank            # hard cap on kept directions
        self.decay = decay
        self.warmup = warmup
        self.filter_strength = filter_strength
        # Direction C — graded eigenvalue weighting (soft filter).
        #   weighting="hard"  -> project onto top-k subspace, all kept dirs weight 1
        #                        (the original hard top-k filter)
        #   weighting="soft"  -> reweight each retained eigendirection i by
        #                        w_i = (lambda_i / lambda_max)^alpha, lambda_i = S_i^2,
        #                        then renormalize the result to preserve ||g|| (so
        #                        alpha is a pure "where to point" knob, decoupled from
        #                        the global learning rate).
        # The alpha spectrum (soft):
        #   alpha = 0          -> if soft_residual: identity (no filter, == base opt);
        #                         else: uniform weights == hard top-k filter
        #   alpha = 1          -> update proportional to how agreed-upon a direction is
        #   alpha -> +inf      -> collapses toward the single dominant direction
        #   alpha < 0          -> whitening / natural-gradient (amplify rare dirs)
        # soft_residual=True keeps g's component OUTSIDE the retained basis at weight 1
        # so that alpha=0 is a genuine no-filter control; False drops it (subspace-only).
        self.weighting = weighting
        self.alpha = alpha
        self.soft_residual = soft_residual
        # If set (e.g. 0.99), keep the smallest number of top eigenvectors that
        # capture this fraction of the spectral energy, capped at `rank`. The
        # eigenvalues are already computed each step, so this is ~free.
        self.energy_threshold = energy_threshold
        # Which BASIS to project onto (one categorical knob):
        #   "none"   -> eigenvectors of the covariance C            (current)
        #   "var"    -> eigenvectors of the correlation D^-1/2 C D^-1/2, D=diag(C)
        #               (per-weight variance divided out: structure over magnitude)
        #   "degree" -> eigenvectors of D^-1/2 C D^-1/2, D=row-sums of C
        #               (the normalized-affinity / spectral-clustering basis;
        #                degree of a SIGNED covariance can be <=0, so it is clamped)
        # All three project onto the top-k eigenspace of D^-1/2 C D^-1/2, which is the
        # column space of A = D^-1/2 V diag(S); we project via A (AᵀA)^-1 Aᵀ — no
        # eigendecomposition needed. normalize!="none" ignores the adaptive proj_k.
        self.normalize = normalize
        # Adaptive rank rule (overrides energy_threshold if not "none"):
        #   "none"    -> keep `rank` (fixed), or energy_threshold if set
        #   "effrank" -> keep round(effective rank) = round(exp(entropy of spectrum))
        #   "gap"     -> keep up to the largest multiplicative gap in the log-spectrum
        # Each step the rank-1 update already produces k+1 candidate eigenpairs
        # (the +1 is the new gradient's component orthogonal to the current basis),
        # so the rule only chooses the truncation cutoff; rank can grow by at most
        # 1 per step (a single rank-1 observation adds at most one new direction).
        self.adaptive = adaptive

        self.param_list = list(model.parameters())
        self.n_params = sum(p.numel() for p in self.param_list)

        self.V = None  # (p, k) top eigenvectors
        self.S = None  # (k,) singular values
        self.proj_k = None  # adaptive #directions to project onto (None = all of V)
        self.step_count = 0

        # Running mean for centering
        self.grad_mean = None

    def _get_flat_grad(self):
        return torch.cat([p.grad.reshape(-1) for p in self.param_list])

    def _set_flat_grad(self, flat_grad):
        offset = 0
        for p in self.param_list:
            numel = p.numel()
            p.grad = flat_grad[offset:offset + numel].reshape(p.shape)
            offset += numel

    def _update_svd(self, g):
        """Rank-1 update to the streaming covariance SVD.

        g: (p,) batch-mean gradient vector on the compute device.
        All heavy ops stay on GPU. Only the small (k+1 × k+1) eigh goes to CPU.
        """
        device = g.device

        # Update running mean with same decay
        if self.grad_mean is None:
            self.grad_mean = g.detach().clone()
        else:
            self.grad_mean.mul_(self.decay).add_(g.detach(), alpha=1 - self.decay)

        g_centered = g.detach() - self.grad_mean  # (p,)

        if self.V is None:
            norm = g_centered.norm()
            if norm > 1e-12:
                self.V = (g_centered / norm).unsqueeze(1).contiguous()  # (p, 1)
                self.S = norm.unsqueeze(0).cpu()  # keep S on CPU for eigh
            return

        k = self.V.shape[1]
        sd = math.sqrt(self.decay)
        sn = math.sqrt(1 - self.decay)

        # Project new gradient onto existing basis: c = V^T @ g_centered (k,)
        c = self.V.T @ g_centered  # (k,) on GPU
        # Residual component orthogonal to V
        g_perp = g_centered - self.V @ c  # (p,) on GPU
        g_perp_norm = g_perp.norm()
        has_perp = g_perp_norm > 1e-12

        # Build (k+1 × k+1) Gram matrix on CPU — never materialize (k+1 × p)
        # Row i of "combined" is: sd * S[i] * V[:,i]  for i < k
        # Row k is: sn * g_centered
        # Gram[i,j] = row_i . row_j
        # For i,j < k: sd² * S[i] * S[j] * (V[:,i] . V[:,j]) = sd² * S[i]*S[j] * delta_ij
        # For i < k, j=k: sd * S[i] * sn * (V[:,i] . g_centered) = sd*sn * S[i] * c[i]
        # For i=k, j=k: sn² * (g_centered . g_centered) = sn² * ||g_centered||²

        S_cpu = self.S  # already on CPU
        c_cpu = c.cpu()
        g_norm_sq = g_centered.dot(g_centered).item()

        gram = torch.zeros(k + 1, k + 1)
        # Diagonal block: sd² * S²
        gram[:k, :k] = torch.diag(sd * sd * S_cpu * S_cpu)
        # Off-diagonal: sd * sn * S * c
        cross = sd * sn * S_cpu * c_cpu
        gram[:k, k] = cross
        gram[k, :k] = cross
        # Bottom-right
        gram[k, k] = sn * sn * g_norm_sq

        eigvals, eigvecs = torch.linalg.eigh(gram)
        eigvals = eigvals.flip(0)
        eigvecs = eigvecs.flip(1)
        pos = eigvals > 1e-12
        eigvals = eigvals[pos]
        eigvecs = eigvecs[:, pos]
        # Basis truncation. For the "effrank"/"gap" rules we deliberately KEEP the
        # full basis (up to `rank`) and only narrow the *projection* (self.proj_k,
        # computed below) — decoupling estimation rank from projection rank. This
        # avoids a ratchet: if we truncated the basis to a tiny adaptive count, the
        # covariance could only ever grow +1 direction/step and would get stuck at 1
        # whenever early gradients are near rank-1. Energy-threshold keeps its
        # original basis-truncating semantics (the CIFAR energy sweep used those).
        new_k = min(self.rank, len(eigvals))
        if self.adaptive == "none" and self.energy_threshold is not None and len(eigvals) > 0:
            # smallest #components capturing `energy_threshold` of the energy
            frac = torch.cumsum(eigvals, 0) / eigvals.sum()
            k_energy = int(torch.searchsorted(frac, self.energy_threshold).item()) + 1
            new_k = max(1, min(self.rank, k_energy, len(eigvals)))
        eigvals = eigvals[:new_k]
        eigvecs = eigvecs[:, :new_k]  # (k+1, new_k)
        s_new = eigvals.sqrt()

        # Recover V_new = [V | q] @ eigvecs @ diag(1/s_new)
        # where q = g_perp / ||g_perp|| (the new basis vector)
        # [V | q] is (p, k+1), but we compute V_new without materializing it:
        # V_new[:,j] = sum_i eigvecs[i,j]/s_new[j] * (row_i_direction)
        # For i < k: direction = V[:,i]
        # For i = k: direction = q (or g_centered if no perp component)

        coeffs = eigvecs / s_new.unsqueeze(0)  # (k+1, new_k)
        # V_new = V @ (sd * diag(S_cpu) @ coeffs[:k]) + q @ (sn * coeffs[k:k+1])
        # The "combined" rows are sd*S[i]*V[:,i] and sn*g_centered
        # So V_new = V @ diag(sd*S) @ coeffs[:k] + (sn * g_centered) * coeffs[k]
        #          = V @ (sd * S.unsqueeze(1) * coeffs[:k]).to(device) + ...

        top_coeffs = (sd * S_cpu.unsqueeze(1) * coeffs[:k]).to(device)  # (k, new_k)
        bot_coeffs = (sn * coeffs[k]).to(device)  # (new_k,)

        V_new = self.V @ top_coeffs + g_centered.unsqueeze(1) * bot_coeffs.unsqueeze(0)

        self.V = V_new.contiguous()
        self.S = s_new
        self._update_proj_k()

    def _update_proj_k(self):
        """How many of the (broad) top directions to actually project onto this step.

        Measured on the full retained spectrum self.S, so it can jump to the right
        value immediately rather than ratcheting up one per step.
        """
        if self.adaptive == "none" or self.S is None or len(self.S) == 0:
            self.proj_k = None
            return
        ev = (self.S ** 2)
        ev = ev[ev > 1e-12]
        n = len(ev)
        if n == 0:
            self.proj_k = None
            return
        if self.adaptive == "effrank":
            p = ev / ev.sum()
            eff = float(torch.exp(-(p * (p + 1e-30).log()).sum()).item())
            self.proj_k = max(1, min(self.rank, n, int(round(eff))))
        elif self.adaptive == "gap":
            if n == 1:
                self.proj_k = 1
            else:
                logs = (ev + 1e-30).log()
                drops = logs[:-1] - logs[1:]
                self.proj_k = max(1, min(self.rank, n, int(torch.argmax(drops).item()) + 1))
        else:
            self.proj_k = None

    def _project_gradient(self, g):
        if self.V is None:
            return g
        if self.normalize == "none" and self.weighting == "soft":
            V = self.V if self.proj_k is None else self.V[:, :self.proj_k]
            S = self.S.to(g.device)
            if self.proj_k is not None:
                S = S[:self.proj_k]
            lam = S * S
            lam_max = lam.max().clamp_min(1e-30)
            ratio = (lam / lam_max).clamp_min(1e-12)        # (k,) in (0, 1]
            w = ratio.pow(self.alpha).clamp(max=1e3)        # graded weights
            coeffs = V.T @ g                                # (k,)
            if self.soft_residual:
                # g + V diag(w-1) Vᵀg : retained dirs reweighted, complement untouched
                g_projected = g + V @ ((w - 1.0) * coeffs)
            else:
                g_projected = V @ (w * coeffs)              # subspace-only
            gn = g.norm()
            pn = g_projected.norm().clamp_min(1e-12)
            g_projected = g_projected * (gn / pn)           # preserve ‖g‖
        elif self.normalize == "none":
            V = self.V if self.proj_k is None else self.V[:, :self.proj_k]
            g_projected = V @ (V.T @ g)
        else:
            # Project onto col(A), A = D^{-1/2} V diag(S) — the top-k eigenspace of
            # the diagonally-normalized covariance. No eigendecomposition needed.
            S = self.S.to(g.device)
            Vs = self.V * S.unsqueeze(0)                  # (p, k) = V diag(S)
            if self.normalize == "var":
                D = (Vs * Vs).sum(1)                      # C_ii = per-weight variance
            else:  # "degree": row-sums of C = Vs @ (S * colsum)
                colsum = self.V.sum(0)                    # (k,)
                D = Vs @ (S * colsum)                     # (p,)
                D = D.abs()                               # signed degree -> clamp |.|
            Dinv = (D.clamp_min(1e-12)).rsqrt()           # (p,)
            A = Vs * Dinv.unsqueeze(1)                    # (p, k)
            G = A.T @ A                                   # (k, k)
            G += 1e-6 * torch.eye(G.shape[0], device=G.device, dtype=G.dtype)
            coeffs = torch.linalg.solve(G, A.T @ g)       # (k,)
            g_projected = A @ coeffs
        if self.filter_strength < 1.0:
            return (1 - self.filter_strength) * g + self.filter_strength * g_projected
        return g_projected

    def step(self, inputs, targets):
        """One optimization step. Returns (loss_value, diagnostics_dict)."""
        self.step_count += 1

        # Standard forward/backward
        self.base_optimizer.zero_grad()
        logits = self.model(inputs)
        loss = F.cross_entropy(logits, targets)
        loss.backward()
        loss_val = loss.item()

        flat_grad = self._get_flat_grad()

        # Update covariance estimate every step
        self._update_svd(flat_grad)

        # Apply filter after warmup
        if self.step_count > self.warmup:
            filtered_grad = self._project_gradient(flat_grad)
            self._set_flat_grad(filtered_grad)

        self.base_optimizer.step()
        self.base_optimizer.zero_grad()

        diagnostics = {
            "loss": loss_val,
            "step": self.step_count,
            "filtering_active": self.step_count > self.warmup,
        }
        if self.S is not None:
            diagnostics["top_singular_values"] = self.S[:5].tolist()
            total = (self.S ** 2).sum().item()
            if total > 1e-12:
                diagnostics["variance_in_top5"] = (self.S[:5] ** 2).sum().item() / total
            diagnostics["effective_rank"] = self._effective_rank()
            # basis_rank: directions tracked; kept_rank: directions actually projected
            # onto (== basis_rank unless an adaptive rule narrows the projection).
            basis_rank = self.V.shape[1] if self.V is not None else 0
            diagnostics["basis_rank"] = basis_rank
            diagnostics["kept_rank"] = self.proj_k if self.proj_k is not None else basis_rank

        return loss_val, diagnostics

    def _effective_rank(self):
        if self.S is None:
            return 0.0
        s2 = self.S ** 2
        s2 = s2[s2 > 1e-12]
        if len(s2) == 0:
            return 0.0
        p = s2 / s2.sum()
        entropy = -(p * p.log()).sum().item()
        return math.exp(entropy)

    def reset(self):
        self.V = None
        self.S = None
        self.proj_k = None
        self.step_count = 0
        self.grad_mean = None

    def zero_grad(self):
        self.base_optimizer.zero_grad()
