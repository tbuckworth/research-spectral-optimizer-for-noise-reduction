"""Frozen symmetric complete-procedure search-space generation."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

SEARCH_SEED = 20260810
N_CONFIGS = 40


def _choice(options, u: float):
    return options[min(int(u * len(options)), len(options) - 1)]


def _log_uniform(low: float, high: float, u: float) -> float:
    return float(math.exp(math.log(low) + u * (math.log(high) - math.log(low))))


def generate_search(seed: int = SEARCH_SEED, n: int = N_CONFIGS) -> list[dict]:
    """Generate paired AdamW/spectral configurations from identical shared draws."""
    rng = np.random.default_rng(seed)
    # Stratification per dimension avoids 40 independent clumps while retaining
    # an auditable, dependency-light generator.
    draws = np.empty((n, 20))
    for column in range(draws.shape[1]):
        draws[:, column] = (rng.permutation(n) + rng.random(n)) / n
    output = []
    for i, u in enumerate(draws):
        shared = {
            "config_id": i,
            "feature_set": _choice(["medium", "all"], u[0]),
            "width": _choice([512, 768, 1024, 1536, 2048], u[1]),
            "depth": _choice([2, 3, 4], u[2]),
            "activation": _choice(["relu", "gelu", "silu"], u[3]),
            "normalization": _choice(["none", "layer"], u[4]),
            "residual": bool(u[5] >= 0.5),
            "dropout": _choice([0.0, 0.02, 0.05, 0.1, 0.2], u[6]),
            "learning_rate": _log_uniform(1e-5, 3e-3, u[7]),
            "weight_decay": _choice([0.0, _log_uniform(1e-8, 1e-2, u[8])], u[9]),
            "batch_size": _choice([1024, 2048, 4096, 8192], u[10]),
            "batch_mode": _choice(["rows", "era"], u[11]),
            "loss": _choice(["mse", "huber"], u[12]),
            "schedule": _choice(["constant", "cosine"], u[13]),
            "warmup_fraction": _choice([0.0, 0.02, 0.05, 0.1], u[14]),
            "clip_grad_norm": _choice([0.0, 1.0, 5.0], u[15]),
            "target": "target",
        }
        spectral = {
            "rank": _choice([16, 32, 64, 128, 256], u[16]),
            "decay": _choice([0.9, 0.95, 0.98, 0.99, 0.995, 0.999], u[17]),
            "filter_strength": _choice([0.25, 0.5, 0.75, 1.0], u[18]),
            "filter_warmup": _choice([100, 500, 1000, 2500], u[19]),
            "filter_update_every": _choice([1, 2, 5, 10], (u[16] + u[19]) % 1),
            "filter_mode": "learned",
        }
        output.append({"arm": "adamw", **shared})
        output.append({"arm": "spectral", **shared, **spectral})
    return output


def write_search(path: Path, seed: int = SEARCH_SEED, n: int = N_CONFIGS) -> None:
    configs = generate_search(seed, n)
    payload = {
        "protocol": "exp-008-main-target-symmetric-complete-procedure-v2",
        "seed": seed,
        "configurations_per_arm": n,
        "primary_target": "target",
        "primary_metric": "standalone_exact_corr",
        "fairness_unit": "completed configs/folds/seeds/examples/updates; runtime descriptive",
        "configs": configs,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=SEARCH_SEED)
    parser.add_argument("--n", type=int, default=N_CONFIGS)
    args = parser.parse_args()
    write_search(args.output, args.seed, args.n)


if __name__ == "__main__":
    main()
