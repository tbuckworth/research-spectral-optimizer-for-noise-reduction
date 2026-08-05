"""Block-diagonal approximation to a high-rank parameter covariance filter."""
from __future__ import annotations

import torch
import torch.nn as nn

from spectral_filter_fixed import StableSpectralGradientFilter


class _FlatBlock(nn.Module):
    def __init__(self, size, device, dtype):
        super().__init__()
        self.value = nn.Parameter(torch.zeros(size, device=device, dtype=dtype),
                                  requires_grad=True)


class _ComplementStableFilter(StableSpectralGradientFilter):
    def __init__(self, *args, renormalize_complement=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.renormalize_complement = renormalize_complement

    def _project_gradient(self, g):
        if self.V is None:
            return g
        V = self.V if self.proj_k is None else self.V[:, :self.proj_k]
        out = g - V @ (V.T @ g)
        if self.renormalize_complement:
            out = out * (g.norm() / out.norm().clamp_min(1e-12))
        return out


class BlockSpectralGradientFilter:
    """Exact within fixed flat-parameter blocks; cross-block covariance is zero.

    ``total_rank`` is divided as evenly as possible between blocks. Therefore
    the retained block-diagonal subspace has at most ``total_rank`` dimensions,
    while storage and rotation cost depend on rank per block rather than the
    global total rank.
    """
    def __init__(self, model, total_rank, block_size=250_000,
                 projection_mode="top", **filter_kwargs):
        self.param_list = list(model.parameters())
        self.n_params = sum(p.numel() for p in self.param_list)
        first = self.param_list[0]
        sizes = []
        left = self.n_params
        while left:
            sizes.append(min(block_size, left)); left -= sizes[-1]
        if total_rank < len(sizes):
            raise ValueError(f"total_rank={total_rank} must be >= number of blocks={len(sizes)}")
        q, rem = divmod(total_rank, len(sizes))
        ranks = [q + (i < rem) for i in range(len(sizes))]
        self.blocks = []
        if projection_mode not in {"top", "remove", "remove-renorm"}:
            raise ValueError(f"unknown projection_mode={projection_mode}")
        for size, rank in zip(sizes, ranks):
            holder = _FlatBlock(size, first.device, first.dtype)
            cls = StableSpectralGradientFilter if projection_mode == "top" else _ComplementStableFilter
            extra = {} if projection_mode == "top" else {
                "renormalize_complement": projection_mode == "remove-renorm"}
            filt = cls(holder, None, rank=rank, **filter_kwargs, **extra)
            self.blocks.append((holder, filt))
        self.block_sizes = sizes
        self.block_rank_caps = ranks
        self.total_rank = total_rank
        self.projection_mode = projection_mode
        self.step_count = 0

    def _get_flat_grad(self):
        return torch.cat([p.grad.reshape(-1) for p in self.param_list])

    def _set_flat_grad(self, flat):
        offset = 0
        for p in self.param_list:
            n = p.numel()
            p.grad = flat[offset:offset+n].reshape_as(p)
            offset += n

    def filter_grad(self):
        flat = self._get_flat_grad()
        filtered = []
        offset = 0
        diagnostics = []
        for size, (holder, filt) in zip(self.block_sizes, self.blocks):
            holder.value.grad = flat[offset:offset+size]
            diagnostics.append(filt.filter_grad())
            filtered.append(holder.value.grad)
            offset += size
        self._set_flat_grad(torch.cat(filtered))
        self.step_count += 1
        return {"step": self.step_count,
                "filtering_active": diagnostics[0]["filtering_active"],
                "block_count": len(self.blocks),
                "requested_total_rank": self.total_rank,
                "projection_mode": self.projection_mode,
                "realized_total_basis_rank": sum(d.get("basis_rank", 0) for d in diagnostics),
                "max_block_rank": max(self.block_rank_caps),
                "mean_effective_rank": sum(d.get("effective_rank", 0.0) for d in diagnostics)/len(diagnostics)}
