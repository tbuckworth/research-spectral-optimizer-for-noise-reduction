#!/usr/bin/env python3
"""B1 named eigh variants of SpectralGradientFilter (EU-1 timing only).

These are the two pre-registered engineering variants that could buy back
rank 512/2048 if the per-step CPU eigh (plus the O(p k^2) basis-update
matmul) makes the base configuration unaffordable. Per success-criteria.md
("eigh-every-N-steps / GPU-fp32-eigh variants (named documented variants if
ever used)") they are timed in EU-1 and usable later ONLY as documented
variants with these timings attached.

The canonical filter (spectral_filter.py, RUN-COPY MOD 1) is NOT modified;
variants subclass it.

Variant semantics (documented, binding if ever used):
  EighEveryNFilter   : the full covariance/basis update (_update_svd — the
                       (k+1)x(k+1) eigh AND the O(p k^2) basis rotation) runs
                       only every N-th filter_grad() call; the projection of
                       the gradient onto the current basis still runs EVERY
                       step. Covariance decay semantics therefore change: the
                       EMA sees every N-th gradient only.
  GpuEighFilter      : the (k+1)x(k+1) eigh runs on the GPU in fp32 (cuSOLVER)
                       instead of the CPU; results are moved back to CPU so
                       the rest of the canonical code path is unchanged. On
                       any GPU-eigh failure it falls into the same CPU-fp64
                       fallback as the base filter (counted in
                       eigh_fallback_count) — parent audit Finding 5 showed
                       fp32 cuSOLVER eigh CAN fail under rank collapse, which
                       is why this is a variant, not the default.
  GpuEighEveryNFilter: both combined.
"""

import torch

from spectral_filter import SpectralGradientFilter


class EighEveryNFilter(SpectralGradientFilter):
    def __init__(self, *args, every_n=8, **kwargs):
        super().__init__(*args, **kwargs)
        self.every_n = int(every_n)

    def filter_grad(self):
        self.step_count += 1
        flat_grad = self._get_flat_grad()
        # basis/covariance update only every N steps (or until a basis exists)
        if self.V is None or (self.step_count % self.every_n) == 0:
            self._update_svd(flat_grad)
        if self.step_count > self.warmup:
            filtered = self._project_gradient(flat_grad)
            self._set_flat_grad(filtered)
        return self._diagnostics()


class GpuEighFilter(SpectralGradientFilter):
    def _robust_eigh(self, gram):
        try:
            ev, U = torch.linalg.eigh(gram.cuda())
            return ev.cpu(), U.cpu()
        except Exception as e:
            self.eigh_fallback_count += 1
            print(f"SPECTRAL-FILTER gpu-eigh fallback #{self.eigh_fallback_count} "
                  f"(CPU fp64) at step {self.step_count}: "
                  f"{type(e).__name__}: {e}", flush=True)
            ev64, U64 = torch.linalg.eigh(gram.detach().double().cpu())
            return ev64.to(gram.dtype), U64.to(gram.dtype)


class GpuEighEveryNFilter(GpuEighFilter):
    def __init__(self, *args, every_n=8, **kwargs):
        super().__init__(*args, **kwargs)
        self.every_n = int(every_n)

    filter_grad = EighEveryNFilter.filter_grad
