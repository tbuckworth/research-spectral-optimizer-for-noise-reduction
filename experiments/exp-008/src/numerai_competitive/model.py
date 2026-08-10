"""Configurable residual MLP used by Exp-008."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


def _activation(name: str) -> nn.Module:
    choices = {"relu": nn.ReLU, "gelu": nn.GELU, "silu": nn.SiLU}
    try:
        return choices[name]()
    except KeyError as exc:
        raise ValueError(f"unknown activation {name!r}") from exc


def _normalization(name: str, width: int) -> nn.Module:
    if name == "none":
        return nn.Identity()
    if name == "layer":
        return nn.LayerNorm(width)
    if name == "batch":
        return nn.BatchNorm1d(width)
    raise ValueError(f"unknown normalization {name!r}")


@dataclass(frozen=True)
class MLPConfig:
    input_dim: int
    width: int = 256
    depth: int = 3
    residual: bool = True
    normalization: str = "layer"
    activation: str = "gelu"
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.input_dim < 1 or self.width < 1 or self.depth < 1:
            raise ValueError("input_dim, width, and depth must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        _activation(self.activation)
        _normalization(self.normalization, self.width)


class _HiddenLayer(nn.Module):
    def __init__(self, config: MLPConfig) -> None:
        super().__init__()
        self.residual = config.residual
        self.net = nn.Sequential(
            _normalization(config.normalization, config.width),
            nn.Linear(config.width, config.width),
            _activation(config.activation),
            nn.Dropout(config.dropout),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output = self.net(inputs)
        return inputs + output if self.residual else output


class ResidualMLP(nn.Module):
    """A scalar-output MLP with optional equal-width residual hidden layers."""

    def __init__(self, config: MLPConfig) -> None:
        super().__init__()
        self.config = config
        self.input = nn.Sequential(
            nn.Linear(config.input_dim, config.width),
            _activation(config.activation),
            nn.Dropout(config.dropout),
        )
        self.hidden = nn.Sequential(*[_HiddenLayer(config) for _ in range(config.depth - 1)])
        self.output_norm = _normalization(config.normalization, config.width)
        self.output = nn.Linear(config.width, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = self.hidden(self.input(inputs))
        return self.output(self.output_norm(hidden)).squeeze(-1)

