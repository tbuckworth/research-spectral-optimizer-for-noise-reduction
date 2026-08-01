#!/usr/bin/env python3
"""Regression adapter for the streaming variant (smoke test only, sub-component
#6). The original `weight_cov_optimizer_v2.py` (copied verbatim from
~/pyg/optimizers, read-only reference) hard-codes F.cross_entropy in step().
This subclass overrides ONLY step() to use MSE. The source repo is untouched;
this adapter exists so the streaming variant can be smoke-tested on the
regression loop. Any actual use of the streaming variant downstream is an
F11 scope change and must be reported as such.
"""
import torch.nn.functional as F

from weight_cov_optimizer_v2 import WeightCovarianceFilterV2


class WeightCovarianceFilterV2Reg(WeightCovarianceFilterV2):
    def step(self, inputs, targets):
        self.step_count += 1
        self.base_optimizer.zero_grad()
        preds = self.model(inputs)
        loss = F.mse_loss(preds, targets)
        loss.backward()
        loss_val = loss.item()

        flat_grad = self._get_flat_grad()
        self._update_svd(flat_grad)
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
            basis_rank = self.V.shape[1] if self.V is not None else 0
            diagnostics["basis_rank"] = basis_rank
        return loss_val, diagnostics
