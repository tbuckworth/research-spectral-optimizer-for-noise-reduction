"""Predeclare an offline submission route using nested-outer predictions only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import atomic_json, sha256
from .metrics import per_era_corr, per_era_correlation_contribution, summarize_era_scores

MODEL_WEIGHTS = (0.1, 0.25, 0.5, 0.75, 1.0)


def _rank_within_era(values: np.ndarray, eras: np.ndarray) -> np.ndarray:
    frame = pd.DataFrame({"value": values, "era": eras})
    ranks = frame.groupby("era", sort=False)["value"].rank(method="average")
    counts = frame.groupby("era", sort=False)["value"].transform("size")
    return ((ranks - 0.5) / counts).to_numpy(np.float64)


def _score(values: np.ndarray, target: pd.Series, benchmark: pd.Series,
           eras: pd.Series, splits: np.ndarray) -> dict:
    prediction = pd.Series(values, index=eras.index, name="candidate")
    corr = per_era_corr(prediction, target, eras)["candidate"]
    bmc = per_era_correlation_contribution(prediction, benchmark, target, eras)["candidate"]
    era_split = pd.Series(splits, index=eras.index).groupby(eras, sort=False).first()
    if not era_split.index.equals(corr.index):
        raise ValueError("outer split labels do not align with era scores")
    split_mean = corr.groupby(era_split).mean()
    return {
        "corr": summarize_era_scores(corr).loc["candidate"].to_dict(),
        "bmc": summarize_era_scores(bmc).loc["candidate"].to_dict(),
        "split_corr_mean": {str(key): float(value) for key, value in split_mean.items()},
        "worst_split_corr": float(split_mean.min()),
    }


def select_candidate(oof_path: Path, output: Path,
                     model_weights: tuple[float, ...] = MODEL_WEIGHTS) -> dict:
    if not model_weights or any(weight <= 0 or weight > 1 for weight in model_weights):
        raise ValueError("model weights must be non-empty and in (0, 1]")
    with np.load(oof_path) as saved:
        required = {"row_index", "era", "target", "benchmark", "adamw", "spectral", "split"}
        if not required <= set(saved.files):
            raise ValueError(f"OOF artifact lacks {sorted(required - set(saved.files))}")
        seed_keys = sorted(
            key for key in saved.files if key.startswith(("adamw_seed_", "spectral_seed_"))
        )
        arrays = {key: saved[key].copy() for key in required | set(seed_keys)}
    adam_seeds = {key.removeprefix("adamw_seed_") for key in seed_keys
                  if key.startswith("adamw_seed_")}
    spectral_seeds = {key.removeprefix("spectral_seed_") for key in seed_keys
                      if key.startswith("spectral_seed_")}
    if not adam_seeds or adam_seeds != spectral_seeds:
        raise ValueError("OOF artifact must contain matching per-seed predictions for both arms")
    lengths = {len(value) for value in arrays.values()}
    if lengths != {len(arrays["row_index"])} or len(np.unique(arrays["row_index"])) != len(
            arrays["row_index"]):
        raise ValueError("OOF arrays are unequal or contain duplicate rows")
    ids = pd.Index([f"row_{row}" for row in arrays["row_index"]], name="id")
    eras = pd.Series([f"{int(value):04d}" for value in arrays["era"]], index=ids, name="era")
    target = pd.Series(arrays["target"].astype(float), index=ids, name="target")
    benchmark_values = _rank_within_era(arrays["benchmark"], arrays["era"])
    benchmark = pd.Series(benchmark_values, index=ids, name="benchmark")
    candidates = []
    standalone = {}
    for arm in ("adamw", "spectral"):
        model = _rank_within_era(arrays[arm], arrays["era"])
        standalone[arm] = _score(model, target, benchmark, eras, arrays["split"])
        seed_stability = {}
        for seed in sorted(adam_seeds, key=int):
            seed_values = _rank_within_era(arrays[f"{arm}_seed_{seed}"], arrays["era"])
            seed_metrics = _score(seed_values, target, benchmark, eras, arrays["split"])
            seed_stability[seed] = seed_metrics
        standalone[arm]["seeds"] = seed_stability
        stable = (
            all(value > 0 for value in standalone[arm]["split_corr_mean"].values())
            and all(
                value > 0
                for metrics in seed_stability.values()
                for value in metrics["split_corr_mean"].values()
            )
        )
        for weight in model_weights:
            blended = _rank_within_era(
                weight * model + (1 - weight) * benchmark_values, arrays["era"]
            )
            metrics = _score(blended, target, benchmark, eras, arrays["split"])
            candidates.append({
                "arm": arm, "model_weight": float(weight),
                "benchmark_weight": float(1 - weight), "benchmark": "v53_lgbm_ender20",
                "standalone_stable_positive": stable, "metrics": metrics,
            })
    eligible = [candidate for candidate in candidates if candidate["standalone_stable_positive"]]
    if not eligible:
        raise ValueError("neither optimizer has positive standalone CORR in every outer fold")
    selected = max(
        eligible,
        key=lambda candidate: (
            candidate["metrics"]["corr"]["mean"],
            candidate["metrics"]["worst_split_corr"],
            candidate["metrics"]["bmc"]["mean"],
            candidate["model_weight"],
            candidate["arm"] == "spectral",
        ),
    )
    payload = {
        "status": "frozen_train_only_selection",
        "source_oof_sha256": sha256(oof_path),
        "criterion": (
            "eligible arm and every confirmation seed have positive standalone mean CORR "
            "in every nested outer fold; "
            "then descending blend mean CORR, worst-fold CORR, BMC, model weight, spectral tie"
        ),
        "model_weights": list(model_weights), "standalone": standalone,
        "candidates": candidates, "selected": selected,
    }
    atomic_json(output, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oof", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(select_candidate(args.oof, args.output), sort_keys=True))


if __name__ == "__main__":
    main()
