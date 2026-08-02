import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import vmap, grad_and_value, functional_call


class SpectralConsensusFilter:
    """Filters gradients using the eigenspectrum of the inter-sample gradient
    similarity matrix. Only gradient directions that many samples agree on
    survive the projection.

    Wraps a base optimizer: computes per-sample gradients, projects onto
    consensus subspace, sets .grad on parameters, then calls base_optimizer.step().
    """

    def __init__(self, model, base_optimizer, loss_fn=None,
                 variance_threshold=None, mp_factor=2.0,
                 soft=False, soft_temp=0.5):
        self.model = model
        self.base_optimizer = base_optimizer
        self.loss_fn = loss_fn or nn.CrossEntropyLoss()
        self.variance_threshold = variance_threshold
        self.mp_factor = mp_factor
        self.soft = soft
        self.soft_temp = soft_temp

        self.params = {k: v for k, v in model.named_parameters()}
        self.buffers = {k: v for k, v in model.named_buffers()}
        self.param_keys = list(self.params.keys())

        self._setup_vmap()

    def _setup_vmap(self):
        def compute_loss(params, buffers, x, y):
            out = functional_call(self.model, (params, buffers), (x.unsqueeze(0),))
            return self.loss_fn(out, y.unsqueeze(0))

        self._grad_and_value_fn = vmap(
            grad_and_value(compute_loss, has_aux=False),
            in_dims=(None, None, 0, 0),
        )

    def step(self, inputs, targets):
        """One optimization step. Returns (loss, diagnostics_dict)."""
        params = {k: v for k, v in self.model.named_parameters()}
        buffers = {k: v for k, v in self.model.named_buffers()}

        per_sample_grads, per_sample_losses = self._grad_and_value_fn(
            params, buffers, inputs, targets
        )

        batch_size = inputs.size(0)
        mean_loss = per_sample_losses.mean().item()

        flat_grads = self._flatten_grads(per_sample_grads, batch_size)
        consensus_grad, diagnostics = self._spectral_filter(flat_grads, batch_size)
        self._set_grads(consensus_grad)

        self.base_optimizer.step()
        self.base_optimizer.zero_grad()

        diagnostics["loss"] = mean_loss
        return mean_loss, diagnostics

    def _flatten_grads(self, per_sample_grads, batch_size):
        flat_list = []
        for key in self.param_keys:
            g = per_sample_grads[key]
            flat_list.append(g.reshape(batch_size, -1))
        return torch.cat(flat_list, dim=1)

    def _spectral_filter(self, G, batch_size):
        """G: (B, p) per-sample gradient matrix. Returns filtered gradient (p,)."""
        norms = G.norm(dim=1, keepdim=True).clamp(min=1e-12)
        G_norm = G / norms

        S = G_norm @ G_norm.T

        eigenvalues, U = torch.linalg.eigh(S)
        eigenvalues = eigenvalues.flip(0).clamp(min=0)
        U = U.flip(1)

        total_var = eigenvalues.sum().item()
        if total_var < 1e-12:
            return torch.zeros(G.size(1), device=G.device), {
                "eigenvalues": eigenvalues[:10].tolist(),
                "k": 0,
                "consensus_ratio": 0.0,
            }

        mean_eig = total_var / batch_size
        threshold = self.mp_factor * mean_eig
        uniform = torch.ones(batch_size, device=G.device) / batch_size

        if self.variance_threshold is not None:
            cumvar = eigenvalues.cumsum(0) / total_var
            k = int((cumvar < self.variance_threshold).sum().item()) + 1
            k = max(1, min(k, batch_size))
            U_k = U[:, :k]
            weights = U_k @ (U_k.T @ uniform)
        elif self.soft:
            temperature = max(self.soft_temp * mean_eig, 1e-12)
            alpha = torch.sigmoid((eigenvalues - threshold) / temperature)
            weights = U @ (alpha * (U.T @ uniform))
            k = int((eigenvalues > threshold).sum().item())
        else:
            k = int((eigenvalues > threshold).sum().item())
            k = max(1, min(k, batch_size))
            U_k = U[:, :k]
            weights = U_k @ (U_k.T @ uniform)

        consensus_grad = weights @ G

        mean_grad = G.mean(dim=0)
        mean_norm = mean_grad.norm().item()
        consensus_norm = consensus_grad.norm().item()
        ratio = consensus_norm / mean_norm if mean_norm > 1e-12 else 0.0

        return consensus_grad, {
            "eigenvalues": eigenvalues[:10].tolist(),
            "k": k,
            "consensus_ratio": ratio,
        }

    def _set_grads(self, flat_grad):
        offset = 0
        for key in self.param_keys:
            param = self.params[key]
            numel = param.numel()
            param.grad = flat_grad[offset:offset + numel].reshape(param.shape)
            offset += numel

    def zero_grad(self):
        self.base_optimizer.zero_grad()


def compute_eigenvalue_diagnostics(model, loss_fn, inputs, targets):
    """Compute gradient similarity eigenvalues without filtering.
    Used as a diagnostic for non-spectral optimizers."""
    params = {k: v for k, v in model.named_parameters()}
    buffers = {k: v for k, v in model.named_buffers()}
    param_keys = list(params.keys())

    def compute_loss(params_, buffers_, x, y):
        out = functional_call(model, (params_, buffers_), (x.unsqueeze(0),))
        return loss_fn(out, y.unsqueeze(0))

    per_sample_grads, _ = vmap(
        grad_and_value(compute_loss),
        in_dims=(None, None, 0, 0),
    )(params, buffers, inputs, targets)

    batch_size = inputs.size(0)
    flat_list = []
    for key in param_keys:
        g = per_sample_grads[key]
        flat_list.append(g.reshape(batch_size, -1))
    G = torch.cat(flat_list, dim=1)

    norms = G.norm(dim=1, keepdim=True).clamp(min=1e-12)
    G_norm = G / norms
    S = G_norm @ G_norm.T

    eigenvalues = torch.linalg.eigvalsh(S).flip(0).clamp(min=0)
    return eigenvalues[:10].tolist()
