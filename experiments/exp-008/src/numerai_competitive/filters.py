"""Stable global gradient-covariance filtering and mechanism controls.

Operation order is deliberately part of the contract::

    loss.backward() -> optional shared gradient clipping -> filter_grad()
    -> Adam/AdamW moment update -> AdamW decoupled weight decay

Thus the covariance sees exactly the post-loss, pre-Adam gradient supplied to
``filter_grad``.  Clipping is outside this class so the shared experimental
choice can be frozen explicitly.  Projection always precedes Adam moments.

The covariance is never materialised: ``V diag(eigenvalues) V.T`` is retained
with ``V`` of shape ``(p, rank)`` and updated through a small symmetric
``(rank + 1)`` eigendecomposition in float64 on CPU.
"""
from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import torch

OPERATION_ORDER = (
    "backward -> optional shared clipping -> covariance update -> projection "
    "-> Adam moments -> decoupled weight decay"
)
CONTROL_MODES = {"learned", "random", "shuffled", "norm_matched"}


class StreamingSpectralFilter:
    """Low-rank EMA gradient-covariance filter over all model parameters.

    Controls use the same realised rank, warm-up and update cadence as the
    learned filter. ``random`` uses one seeded, persistent random orientation;
    ``shuffled`` updates the EMA from a seeded permutation of bounded gradient
    history; ``norm_matched`` keeps the raw direction but gives it the norm of
    the learned spectral projection.  ``strength=0`` is an exact identity path
    (while state estimation and diagnostics continue).
    """

    def __init__(
        self,
        model: torch.nn.Module | Iterable[torch.nn.Parameter],
        *,
        rank: int = 32,
        decay: float = 0.99,
        warmup: int = 100,
        strength: float = 1.0,
        mode: str = "learned",
        update_every: int = 1,
        seed: int = 0,
        history_size: int = 32,
        relative_eig_tol: float = 1e-10,
    ) -> None:
        if isinstance(model, torch.nn.Module):
            params = list(model.parameters())
        else:
            params = list(model)
        if not params:
            raise ValueError("filter requires at least one parameter")
        if rank < 1:
            raise ValueError("rank must be positive")
        if not 0.0 <= decay < 1.0:
            raise ValueError("decay must be in [0, 1)")
        if warmup < 0 or update_every < 1 or history_size < 1:
            raise ValueError("warmup must be non-negative; cadence/history must be positive")
        if not 0.0 <= strength <= 1.0:
            raise ValueError("strength must be in [0, 1]")
        if mode not in CONTROL_MODES:
            raise ValueError(f"unknown mode {mode!r}; expected one of {sorted(CONTROL_MODES)}")
        if relative_eig_tol < 0:
            raise ValueError("relative_eig_tol must be non-negative")

        self.params = params
        self.n_params = sum(p.numel() for p in params)
        self.rank = min(int(rank), self.n_params)
        self.decay = float(decay)
        self.warmup = int(warmup)
        self.strength = float(strength)
        self.mode = mode
        self.update_every = int(update_every)
        self.seed = int(seed)
        self.history_size = int(history_size)
        self.relative_eig_tol = float(relative_eig_tol)

        self.V: torch.Tensor | None = None
        self.eigenvalues: torch.Tensor | None = None  # CPU float64
        self.grad_mean: torch.Tensor | None = None
        self.random_basis: torch.Tensor | None = None
        self.history: list[torch.Tensor] = []
        self.step_count = 0
        self.update_count = 0
        self._basis_turnover = 0.0
        self._generator = torch.Generator(device="cpu")
        self._generator.manual_seed(self.seed)

    def _flat_grad(self) -> torch.Tensor:
        missing = [i for i, p in enumerate(self.params) if p.grad is None]
        if missing:
            raise RuntimeError(f"parameters without gradients at indices {missing[:8]}")
        return torch.cat([p.grad.detach().reshape(-1) for p in self.params])

    def _set_flat_grad(self, gradient: torch.Tensor) -> None:
        offset = 0
        for param in self.params:
            size = param.numel()
            # Preserve the existing grad tensor identity where possible.
            param.grad.copy_(gradient[offset : offset + size].reshape_as(param))
            offset += size

    @staticmethod
    def _orthogonality_error(basis: torch.Tensor | None) -> float:
        if basis is None or basis.shape[1] == 0:
            return 0.0
        gram = basis.T @ basis
        eye = torch.eye(gram.shape[0], dtype=gram.dtype, device=gram.device)
        return float(torch.linalg.matrix_norm(gram - eye, ord=2).item())

    def _history_gradient(self, raw: torch.Tensor) -> torch.Tensor:
        if self.mode != "shuffled":
            return raw
        self.history.append(raw.detach().cpu().clone())
        if len(self.history) > self.history_size:
            self.history.pop(0)
        index = int(torch.randint(len(self.history), (1,), generator=self._generator).item())
        return self.history[index].to(device=raw.device, dtype=raw.dtype)

    def _update_covariance(self, observation: torch.Tensor) -> None:
        observation = observation.detach()
        if self.grad_mean is None:
            self.grad_mean = observation.clone()
            return
        centered = observation - self.grad_mean
        self.grad_mean.mul_(self.decay).add_(observation, alpha=1.0 - self.decay)
        centered_norm = centered.norm()
        eps = 10.0 * torch.finfo(centered.dtype).eps * max(1.0, float(centered_norm))
        old_basis = self.V

        if self.V is None:
            if float(centered_norm) <= eps:
                return
            self.V = (centered / centered_norm).unsqueeze(1).contiguous()
            self.eigenvalues = torch.tensor(
                [(1.0 - self.decay) * float(centered_norm.double().square())],
                dtype=torch.float64,
            )
            self.update_count += 1
            return

        basis = self.V
        coefficients = basis.T @ centered
        residual = centered - basis @ coefficients
        # A second pass protects orthogonality after many low-precision updates.
        correction = basis.T @ residual
        residual = residual - basis @ correction
        coefficients = coefficients + correction
        residual_norm = residual.norm()
        has_residual = float(residual_norm) > eps
        old_rank = basis.shape[1]
        size = old_rank + int(has_residual)

        small = torch.zeros((size, size), dtype=torch.float64)
        old_values = self.eigenvalues.double()
        c64 = coefficients.double().cpu()
        weight = 1.0 - self.decay
        small[:old_rank, :old_rank] = torch.diag(self.decay * old_values)
        small[:old_rank, :old_rank].add_(torch.outer(c64, c64), alpha=weight)
        residual_direction = None
        if has_residual:
            r64 = float(residual_norm.double())
            small[:old_rank, old_rank] = weight * c64 * r64
            small[old_rank, :old_rank] = small[:old_rank, old_rank]
            small[old_rank, old_rank] = weight * r64 * r64
            residual_direction = residual / residual_norm

        small = (small + small.T) * 0.5
        values, rotations = torch.linalg.eigh(small)
        values, rotations = values.flip(0), rotations.flip(1)
        floor = self.relative_eig_tol * max(0.0, float(values[0]))
        keep = values > floor
        values, rotations = values[keep][: self.rank], rotations[:, keep][:, : self.rank]
        if values.numel() == 0:
            self.V, self.eigenvalues = None, None
            return

        rotate = rotations.to(device=basis.device, dtype=basis.dtype)
        updated = basis @ rotate[:old_rank]
        if has_residual:
            updated = updated + residual_direction.unsqueeze(1) * rotate[old_rank].unsqueeze(0)
        # QR plus a small covariance rotation repairs round-off without changing
        # the represented covariance.
        q, r = torch.linalg.qr(updated, mode="reduced")
        represented = (r.double().cpu() * values.unsqueeze(0)) @ r.double().cpu().T
        represented = (represented + represented.T) * 0.5
        values, rotation = torch.linalg.eigh(represented)
        values, rotation = values.flip(0), rotation.flip(1)
        positive = values > self.relative_eig_tol * max(0.0, float(values[0]))
        values, rotation = values[positive][: self.rank], rotation[:, positive][:, : self.rank]
        self.V = (q @ rotation.to(q)).contiguous()
        self.eigenvalues = values.contiguous()
        overlap_rank = min(old_basis.shape[1], self.V.shape[1])
        if overlap_rank:
            overlap = torch.linalg.matrix_norm(old_basis.T @ self.V, ord="fro").square()
            self._basis_turnover = 1.0 - float(overlap) / overlap_rank
            self._basis_turnover = min(1.0, max(0.0, self._basis_turnover))
        self.update_count += 1

    def _persistent_random_basis(self, rank: int, reference: torch.Tensor) -> torch.Tensor:
        if self.random_basis is None:
            random = torch.randn(
                (self.n_params, self.rank), generator=self._generator, dtype=torch.float64
            )
            random, _ = torch.linalg.qr(random, mode="reduced")
            self.random_basis = random.to(device=reference.device, dtype=reference.dtype)
        elif self.random_basis.device != reference.device or self.random_basis.dtype != reference.dtype:
            self.random_basis = self.random_basis.to(device=reference.device, dtype=reference.dtype)
        return self.random_basis[:, :rank]

    def _projection(self, raw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.V is None:
            return raw, None
        learned = self.V @ (self.V.T @ raw)
        if self.mode == "random":
            basis = self._persistent_random_basis(self.V.shape[1], raw)
            return basis @ (basis.T @ raw), basis
        if self.mode == "norm_matched":
            scale = learned.norm() / raw.norm().clamp_min(torch.finfo(raw.dtype).tiny)
            return raw * scale, self.V
        return learned, self.V

    def filter_flat(self, raw_gradient: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
        """Update state and return a filtered flat gradient plus diagnostics."""
        if raw_gradient.ndim != 1 or raw_gradient.numel() != self.n_params:
            raise ValueError(f"expected flat gradient of length {self.n_params}")
        if not torch.isfinite(raw_gradient).all():
            raise FloatingPointError("non-finite input gradient")
        self.step_count += 1
        observation = self._history_gradient(raw_gradient)
        if (self.step_count - 1) % self.update_every == 0:
            self._update_covariance(observation)

        candidate, projection_basis = self._projection(raw_gradient)
        active = self.step_count > self.warmup and self.V is not None
        if not active or self.strength == 0.0:
            output = raw_gradient  # exact identity; do not form 0*x + 1*g
        elif self.strength == 1.0:
            output = candidate
        else:
            output = torch.lerp(raw_gradient, candidate, self.strength)
        return output, self._diagnostics(
            raw_gradient, candidate, output, projection_basis, active
        )

    def filter_grad(self) -> dict[str, Any]:
        """Filter model ``.grad`` tensors in place; call before optimizer.step()."""
        raw = self._flat_grad()
        filtered, diagnostics = self.filter_flat(raw)
        self._set_flat_grad(filtered)
        return diagnostics

    def _diagnostics(
        self,
        raw: torch.Tensor,
        projected: torch.Tensor,
        output: torch.Tensor,
        projection_basis: torch.Tensor | None,
        active: bool,
    ) -> dict[str, Any]:
        raw_norm = float(raw.norm())
        projected_norm = float(projected.norm())
        denominator = max(raw_norm * projected_norm, torch.finfo(raw.dtype).tiny)
        cosine = float(torch.dot(raw, projected)) / denominator if projected_norm else 0.0
        retained = (projected_norm / raw_norm) ** 2 if raw_norm else 1.0
        values = self.eigenvalues if self.eigenvalues is not None else torch.empty(0)
        if values.numel():
            probabilities = values / values.sum()
            effective_rank = math.exp(float(-(probabilities * probabilities.log()).sum()))
        else:
            effective_rank = 0.0
        diagnostics = {
            "step": self.step_count,
            "update_count": self.update_count,
            "filtering_active": bool(active and self.strength > 0.0),
            "basis_rank": 0 if self.V is None else self.V.shape[1],
            "raw_norm": raw_norm,
            "projected_norm": projected_norm,
            "filtered_norm": float(output.norm()),
            "cosine_raw_projected": cosine,
            "retained_energy": retained,
            "effective_rank": effective_rank,
            "basis_turnover": self._basis_turnover,
            "orthogonality_error": self._orthogonality_error(projection_basis),
        }
        numeric = [v for v in diagnostics.values() if isinstance(v, (int, float))]
        if not all(math.isfinite(float(v)) for v in numeric):
            raise FloatingPointError("non-finite filter diagnostics")
        return diagnostics

    def state_dict(self) -> dict[str, Any]:
        """Return a CPU checkpoint including controls' stochastic/history state."""
        cpu = lambda x: None if x is None else x.detach().cpu().clone()
        return {
            "config": {
                "n_params": self.n_params,
                "rank": self.rank,
                "decay": self.decay,
                "warmup": self.warmup,
                "strength": self.strength,
                "mode": self.mode,
                "update_every": self.update_every,
                "history_size": self.history_size,
                "relative_eig_tol": self.relative_eig_tol,
            },
            "V": cpu(self.V),
            "eigenvalues": cpu(self.eigenvalues),
            "grad_mean": cpu(self.grad_mean),
            "random_basis": cpu(self.random_basis),
            "history": [x.detach().cpu().clone() for x in self.history],
            "step_count": self.step_count,
            "update_count": self.update_count,
            "basis_turnover": self._basis_turnover,
            "generator_state": self._generator.get_state().clone(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        expected = self.state_dict()["config"]
        if state.get("config") != expected:
            raise ValueError("spectral filter checkpoint configuration mismatch")
        reference = self.params[0]
        move = lambda x: None if x is None else x.to(reference.device, reference.dtype)
        self.V = move(state["V"])
        self.eigenvalues = None if state["eigenvalues"] is None else state["eigenvalues"].double().cpu()
        self.grad_mean = move(state["grad_mean"])
        self.random_basis = move(state["random_basis"])
        self.history = [x.detach().cpu().clone() for x in state["history"]]
        self.step_count = int(state["step_count"])
        self.update_count = int(state["update_count"])
        self._basis_turnover = float(state["basis_turnover"])
        self._generator.set_state(state["generator_state"].cpu())


# A concise name for experiment configuration and compatibility with earlier work.
SpectralGradientFilter = StreamingSpectralFilter
