"""Aggregate nested-outer predictions without touching official validation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .data import atomic_json
from .inference import moving_block_bootstrap
from .metrics import per_era_corr, per_era_correlation_contribution, summarize_era_scores
from .train import _atomic_npz


def _result_path(path: Path) -> Path:
    return path if path.name == "result.json" else path / "result.json"


def _rank_within_era(values: np.ndarray, eras: np.ndarray) -> np.ndarray:
    frame = pd.DataFrame({"value": values, "era": eras})
    ranks = frame.groupby("era", sort=False)["value"].rank(method="average")
    counts = frame.groupby("era", sort=False)["value"].transform("size")
    return ((ranks - 0.5) / counts).to_numpy(np.float64)


def _load_seed(path: Path, expected_arm: str) -> tuple[dict, dict[str, np.ndarray]]:
    result_path = _result_path(path)
    result = json.loads(result_path.read_text())
    if result.get("status") != "complete" or result["config"]["arm"] != expected_arm:
        raise ValueError(f"{result_path}: incomplete result or wrong arm")
    prediction_path = result_path.parent / result["prediction_file"]
    with np.load(prediction_path) as saved:
        arrays = {key: saved[key].copy() for key in
                  ("row_index", "era", "target", "benchmark", "prediction")}
    if not all(len(value) == len(arrays["row_index"]) for value in arrays.values()):
        raise ValueError(f"{prediction_path}: prediction arrays have unequal lengths")
    return result, arrays


def _ensemble_split(paths: list[Path], arm: str, expected_seeds: tuple[int, ...]) -> pd.DataFrame:
    loaded = [_load_seed(path, arm) for path in paths]
    loaded.sort(key=lambda item: int(item[0]["config"]["seed"]))
    seeds = tuple(int(result["config"]["seed"]) for result, _ in loaded)
    if seeds != expected_seeds:
        raise ValueError(f"{arm}: seeds {seeds} differ from expected {expected_seeds}")
    split_names = {result["split"]["name"] for result, _ in loaded}
    config_ids = {result["config"]["search_config_id"] for result, _ in loaded}
    if len(split_names) != 1 or len(config_ids) != 1:
        raise ValueError(f"{arm}: a split ensemble mixes splits or configurations")
    reference = loaded[0][1]
    for _, arrays in loaded[1:]:
        for key in ("row_index", "era", "target", "benchmark"):
            if not np.array_equal(reference[key], arrays[key], equal_nan=True):
                raise ValueError(f"{arm}: seed predictions disagree on {key}")
    ranked = [_rank_within_era(arrays["prediction"], arrays["era"])
              for _, arrays in loaded]
    return pd.DataFrame({
        "row_index": reference["row_index"].astype(np.int64),
        "era": reference["era"].astype(np.int64),
        "target": reference["target"].astype(np.float64),
        "benchmark": reference["benchmark"].astype(np.float64),
        "prediction": np.mean(ranked, axis=0),
        "split": next(iter(split_names)),
        "config_id": next(iter(config_ids)),
    })


def _group_paths(paths: list[Path], arm: str) -> dict[str, list[Path]]:
    grouped: dict[str, list[Path]] = {}
    for path in paths:
        result = json.loads(_result_path(path).read_text())
        if result["config"]["arm"] != arm:
            raise ValueError(f"{path}: expected {arm} result")
        grouped.setdefault(result["split"]["name"], []).append(path)
    return grouped


def _arm_oof(paths: list[Path], arm: str, expected_seeds: tuple[int, ...]) -> pd.DataFrame:
    grouped = _group_paths(paths, arm)
    frame = pd.concat(
        [_ensemble_split(grouped[name], arm, expected_seeds) for name in sorted(grouped)],
        ignore_index=True,
    ).sort_values(["era", "row_index"]).reset_index(drop=True)
    if frame["row_index"].duplicated().any():
        raise ValueError(f"{arm}: outer prediction rows overlap")
    if not frame["era"].is_monotonic_increasing:
        raise ValueError(f"{arm}: outer eras are not ordered")
    return frame


def _score(prediction: pd.Series, target: pd.Series, benchmark: pd.Series,
           eras: pd.Series) -> tuple[dict, pd.DataFrame]:
    corr = per_era_corr(prediction, target, eras)[prediction.name]
    bmc = per_era_correlation_contribution(prediction, benchmark, target, eras)[prediction.name]
    return {
        "corr": summarize_era_scores(corr).loc[prediction.name].to_dict(),
        "bmc": summarize_era_scores(bmc).loc[prediction.name].to_dict(),
    }, pd.DataFrame({"corr": corr, "bmc": bmc})


def _plot(per_era: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    per_era.rolling(26, min_periods=13).mean().plot(ax=axes[0])
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set(title="Nested-outer 26-era rolling exact CORR", ylabel="CORR")
    per_era.cumsum().plot(ax=axes[1])
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set(title="Nested-outer cumulative exact CORR", xlabel="Era", ylabel="Cumulative")
    fig.tight_layout()
    fig.savefig(output / "nested-outer-corr.png", dpi=180, facecolor="white")
    plt.close(fig)


def aggregate(adamw_paths: list[Path], spectral_paths: list[Path], output: Path,
              expected_seeds: tuple[int, ...] = (0, 1, 2)) -> dict:
    adamw = _arm_oof(adamw_paths, "adamw", expected_seeds)
    spectral = _arm_oof(spectral_paths, "spectral", expected_seeds)
    alignment = ["row_index", "era", "target", "benchmark", "split"]
    for column in alignment:
        left, right = adamw[column].to_numpy(), spectral[column].to_numpy()
        equal = (np.array_equal(left, right) if left.dtype.kind in "OUS"
                 else np.array_equal(left, right, equal_nan=True))
        if not equal:
            raise ValueError(f"AdamW and spectral OOF predictions disagree on {column}")
    ids = pd.Index([f"row_{row}" for row in adamw["row_index"]], name="id")
    eras = pd.Series(adamw["era"].map(lambda value: f"{value:04d}").to_numpy(),
                     index=ids, name="era")
    target = pd.Series(adamw["target"].to_numpy(), index=ids, name="target")
    benchmark = pd.Series(adamw["benchmark"].to_numpy(), index=ids, name="benchmark")
    adam_prediction = pd.Series(adamw["prediction"].to_numpy(), index=ids, name="adamw")
    spectral_prediction = pd.Series(
        spectral["prediction"].to_numpy(), index=ids, name="spectral"
    )
    adam_summary, adam_era = _score(adam_prediction, target, benchmark, eras)
    spectral_summary, spectral_era = _score(spectral_prediction, target, benchmark, eras)
    per_era = pd.DataFrame({"adamw": adam_era["corr"], "spectral": spectral_era["corr"]})
    comparison = moving_block_bootstrap(per_era["spectral"], per_era["adamw"])
    output.mkdir(parents=True, exist_ok=True)
    per_era.to_csv(output / "nested-outer-per-era.csv")
    _plot(per_era, output)
    _atomic_npz(
        output / "nested-outer-predictions.npz", row_index=adamw["row_index"].to_numpy(),
        era=adamw["era"].to_numpy(), target=adamw["target"].to_numpy(),
        benchmark=adamw["benchmark"].to_numpy(), adamw=adamw["prediction"].to_numpy(),
        spectral=spectral["prediction"].to_numpy(),
    )
    report = {
        "status": "complete", "rows": len(adamw), "eras": int(adamw["era"].nunique()),
        "outer_splits": sorted(adamw["split"].unique().tolist()),
        "expected_seeds": list(expected_seeds), "adamw": adam_summary,
        "spectral": spectral_summary, "spectral_minus_adamw": comparison.to_dict(),
    }
    atomic_json(output / "nested-outer-report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adamw-result", type=Path, action="append", required=True)
    parser.add_argument("--spectral-result", type=Path, action="append", required=True)
    parser.add_argument("--seed", type=int, action="append", default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    seeds = tuple(args.seed) if args.seed is not None else (0, 1, 2)
    print(json.dumps(aggregate(args.adamw_result, args.spectral_result, args.output, seeds),
                     sort_keys=True))


if __name__ == "__main__":
    main()
