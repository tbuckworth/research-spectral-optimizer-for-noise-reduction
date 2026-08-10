"""Convert a frozen search draw into the executable training schema."""
from __future__ import annotations

from typing import Any


def materialize_config(draw: dict[str, Any], *, input_dim: int, updates: int,
                       seed: int = 0) -> dict[str, Any]:
    """Materialize one search draw without changing any sampled hyperparameter."""
    if draw["feature_set"] not in {"medium", "all"}:
        raise ValueError("unsupported feature set")
    config: dict[str, Any] = {
        "model": {
            "input_dim": input_dim,
            "width": draw["width"],
            "depth": draw["depth"],
            "residual": draw["residual"],
            "normalization": draw["normalization"],
            "activation": draw["activation"],
            "dropout": draw["dropout"],
        },
        "arm": draw["arm"],
        "target": draw["target"],
        "seed": seed,
        "batch_mode": "row" if draw["batch_mode"] == "rows" else draw["batch_mode"],
        "batch_size": draw["batch_size"],
        "updates": updates,
        "examples": updates * draw["batch_size"],
        "log_every": max(50, updates // 200),
        "checkpoint_every": max(250, updates // 2),
        "learning_rate": draw["learning_rate"],
        "weight_decay": draw["weight_decay"],
        "loss": draw["loss"],
        "schedule": draw["schedule"],
        "warmup_updates": round(updates * draw["warmup_fraction"]),
        "clip_norm": None if draw["clip_grad_norm"] == 0 else draw["clip_grad_norm"],
    }
    if draw["arm"] == "spectral":
        config["filter"] = {
            "rank": draw["rank"],
            "decay": draw["decay"],
            "strength": draw["filter_strength"],
            "warmup": draw["filter_warmup"],
            "update_every": draw["filter_update_every"],
            "mode": draw["filter_mode"],
            "seed": seed,
        }
    return config
