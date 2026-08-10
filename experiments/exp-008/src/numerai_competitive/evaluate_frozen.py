"""One-time official-validation evaluation after procedure and model freeze."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from .data import ValidationShard, atomic_json
from .inference import moving_block_bootstrap
from .metrics import per_era_corr, per_era_correlation_contribution, summarize_era_scores
from .model import MLPConfig, ResidualMLP
from .train import _atomic_npz, _evaluate


def _rank_within_era(values: np.ndarray, eras: np.ndarray) -> np.ndarray:
    frame = pd.DataFrame({"value": values, "era": eras})
    ranks = frame.groupby("era", sort=False)["value"].rank(method="average")
    counts = frame.groupby("era", sort=False)["value"].transform("size")
    return ((ranks - 0.5) / counts).to_numpy(np.float64)


def _predict_artifact(artifact: dict, shard: ValidationShard, device: torch.device,
                      batch_size: int) -> np.ndarray:
    if tuple(artifact["feature_names"]) != tuple(shard.manifest["feature_names"]):
        raise ValueError("model and validation feature schemas differ")
    model = ResidualMLP(MLPConfig(**artifact["model_config"]))
    model.load_state_dict(artifact["model"])
    model.eval().to(device)
    output = []
    with torch.inference_mode():
        for start in range(0, len(shard.X), batch_size):
            values = np.asarray(shard.X[start:start + batch_size], dtype=np.float32) / 4
            output.append(model(torch.from_numpy(values).to(device)).float().cpu().numpy())
    return np.concatenate(output).astype(np.float32)


def _ensemble(paths: list[Path], arm: str, shard: ValidationShard, freeze: dict,
              device: torch.device, batch_size: int) -> np.ndarray:
    expected = freeze["selected"][arm]
    artifacts = [torch.load(path, map_location="cpu", weights_only=False) for path in paths]
    signatures = [artifact["signature"] for artifact in artifacts]
    if signatures != expected["model_signatures"]:
        raise ValueError(f"{arm} model signatures/order differ from frozen manifest")
    if len(paths) != len(expected["seeds"]):
        raise ValueError(f"{arm} model count differs from frozen seeds")
    ranked = [
        _rank_within_era(_predict_artifact(artifact, shard, device, batch_size), shard.eras)
        for artifact in artifacts
    ]
    return np.mean(ranked, axis=0).astype(np.float32)


def _plots(per_era: pd.DataFrame, output: Path) -> None:
    rolling = per_era.rolling(52, min_periods=26).mean()
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    rolling.plot(ax=axes[0])
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set(title="52-era rolling exact CORR", ylabel="CORR")
    per_era.cumsum().plot(ax=axes[1])
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set(title="Cumulative exact CORR", xlabel="Era", ylabel="Cumulative CORR")
    fig.tight_layout()
    fig.savefig(output / "official-validation-corr.png", dpi=180, facecolor="white")
    plt.close(fig)


def _prediction_correlation(candidate: np.ndarray, benchmark: pd.Series,
                            eras: pd.Series) -> dict[str, float | int]:
    frame = pd.DataFrame(
        {"candidate": np.asarray(candidate, dtype=np.float64),
         "benchmark": benchmark.to_numpy(dtype=np.float64), "era": eras.to_numpy()},
        index=benchmark.index,
    )
    per_era = frame.groupby("era", sort=False).apply(
        lambda group: group["candidate"].corr(group["benchmark"]),
        include_groups=False,
    )
    if not np.isfinite(per_era).all():
        raise ValueError("candidate-benchmark prediction correlation is non-finite")
    return {
        "global": float(frame["candidate"].corr(frame["benchmark"])),
        "per_era_mean": float(per_era.mean()),
        "per_era_std": float(per_era.std(ddof=1)),
        "per_era_min": float(per_era.min()),
        "per_era_max": float(per_era.max()),
        "eras": len(per_era),
    }


def _evaluate_arrays(prediction: np.ndarray, target_values: np.ndarray,
                     benchmark_values: np.ndarray, era_values: np.ndarray,
                     row_indices: np.ndarray, name: str) -> tuple[dict, pd.DataFrame]:
    ids = pd.Index([f"row_{int(row)}" for row in row_indices], name="id")
    pred = pd.Series(np.asarray(prediction, dtype=np.float64), index=ids, name=name)
    target = pd.Series(np.asarray(target_values, dtype=np.float64), index=ids, name="target")
    benchmark = pd.Series(
        np.asarray(benchmark_values, dtype=np.float64), index=ids, name="benchmark"
    )
    eras = pd.Series(
        [f"{int(value):04d}" for value in era_values], index=ids, name="era"
    )
    corr = per_era_corr(pred, target, eras)[name]
    bmc = per_era_correlation_contribution(pred, benchmark, target, eras)[name]
    return {
        "rows": len(row_indices), "eras": int(eras.nunique()),
        "corr": summarize_era_scores(corr).loc[name].to_dict(),
        "bmc": summarize_era_scores(bmc).loc[name].to_dict(),
    }, pd.DataFrame({"corr": corr, "bmc": bmc})


def _secondary_analyses(shard: ValidationShard, predictions: dict[str, np.ndarray]) -> tuple[dict, pd.DataFrame]:
    definitions: dict[str, tuple[np.ndarray, str, str]] = {}
    for target_name, benchmark_name in (
        ("target_ender_20", "v53_lgbm_ender20"),
        ("target_teager2b_20", "v53_lgbm_ender20"),
        ("target_ender_60", "v53_lgbm_ender60"),
    ):
        if target_name in shard.manifest["targets"] and benchmark_name in shard.manifest["benchmarks"]:
            definitions[target_name] = (
                np.asarray(shard.targets[:, shard.target_index(target_name)], dtype=np.float64),
                benchmark_name,
                "released historical target; secondary only",
            )
    ensemble_names = ("target_cyrusd_20", "target_ender_20", "target_teager2b_20")
    if all(name in shard.manifest["targets"] for name in ensemble_names):
        values = np.column_stack([
            np.asarray(shard.targets[:, shard.target_index(name)], dtype=np.float64)
            for name in ensemble_names
        ])
        covered = np.isfinite(values).all(axis=1)
        ranked = np.column_stack([
            _rank_within_era(values[covered, column], shard.eras[covered])
            for column in range(values.shape[1])
        ])
        ensemble = np.full(len(shard.eras), np.nan, dtype=np.float64)
        ensemble[covered] = _rank_within_era(ranked.mean(axis=1), shard.eras[covered])
        definitions["target_20_rank_ensemble"] = (
            ensemble, "v53_lgbm_ender20",
            "equal rank average of Cyrusd20, Ender20 and Teager2b20; secondary only",
        )
    reports, per_era_rows = {}, []
    for target_name, (target_values, benchmark_name, definition) in definitions.items():
        covered = np.isfinite(target_values)
        indices = np.flatnonzero(covered)
        benchmark_column = shard.benchmark_index(benchmark_name)
        benchmark_values = np.asarray(
            shard.benchmarks[indices, benchmark_column], dtype=np.float64
        )
        target_report = {"definition": definition, "benchmark": benchmark_name, "models": {}}
        model_eras = {}
        for model_name, prediction in predictions.items():
            summary, era_scores = _evaluate_arrays(
                prediction[covered], target_values[covered], benchmark_values,
                shard.eras[covered], indices, model_name,
            )
            target_report["models"][model_name] = summary
            model_eras[model_name] = era_scores
            for era, row in era_scores.iterrows():
                per_era_rows.append({
                    "target": target_name, "model": model_name, "era": era,
                    "corr": row["corr"], "bmc": row["bmc"],
                })
        target_report["spectral_minus_adamw"] = moving_block_bootstrap(
            model_eras["spectral"]["corr"], model_eras["adamw"]["corr"]
        ).to_dict()
        reports[target_name] = target_report
    return reports, pd.DataFrame(per_era_rows)


def evaluate(shard_root: Path, freeze_path: Path, adamw_models: list[Path],
             spectral_models: list[Path], output: Path, device_name: str = "auto",
             batch_size: int = 4096) -> dict:
    shard = ValidationShard.open(shard_root)
    freeze = json.loads(freeze_path.read_text())
    from .data import sha256

    if shard.manifest["freeze_manifest_sha256"] != sha256(freeze_path):
        raise ValueError("validation shard and supplied freeze manifest differ")
    device = torch.device("cuda" if device_name == "auto" and torch.cuda.is_available()
                          else ("cpu" if device_name == "auto" else device_name))
    adamw = _ensemble(adamw_models, "adamw", shard, freeze, device, batch_size)
    spectral = _ensemble(spectral_models, "spectral", shard, freeze, device, batch_size)
    target_column = shard.target_index("target_cyrusd_20")
    covered = np.isfinite(shard.targets[:, target_column])
    indices = np.flatnonzero(covered)
    benchmark_column = shard.benchmark_index("v53_lgbm_ender20")
    adam_summary, adam_era = _evaluate(
        adamw[covered], shard, indices, target_column, benchmark_column
    )
    spectral_summary, spectral_era = _evaluate(
        spectral[covered], shard, indices, target_column, benchmark_column,
    )
    ids = pd.Index([f"row_{i}" for i in indices])
    benchmark = pd.Series(
        np.asarray(shard.benchmarks[indices, benchmark_column], dtype=np.float64), index=ids
    )
    target = pd.Series(
        np.asarray(shard.targets[indices, target_column], dtype=np.float64),
        index=ids, name="target",
    )
    eras = pd.Series(
        [f"{int(value):04d}" for value in shard.eras[indices]], index=ids, name="era"
    )
    benchmark_era = per_era_corr(benchmark.rename("ender20"), target, eras)["ender20"]
    transform = freeze["candidate_transform"]
    base_candidate = adamw if transform["arm"] == "adamw" else spectral
    benchmark_rank_full = _rank_within_era(
        np.asarray(shard.benchmarks[:, benchmark_column], dtype=np.float64), shard.eras
    )
    candidate_full = _rank_within_era(
        transform["model_weight"] * base_candidate
        + transform["benchmark_weight"] * benchmark_rank_full,
        shard.eras,
    )
    candidate = candidate_full[covered]
    candidate_summary, candidate_era = _evaluate(
        candidate, shard, indices, target_column, benchmark_column
    )
    spectral_minus_adamw = moving_block_bootstrap(spectral_era["corr"], adam_era["corr"])
    adamw_minus_ender20 = moving_block_bootstrap(adam_era["corr"], benchmark_era)
    spectral_minus_ender20 = moving_block_bootstrap(spectral_era["corr"], benchmark_era)
    candidate_minus_ender20 = moving_block_bootstrap(candidate_era["corr"], benchmark_era)
    prediction_correlation = {
        "adamw_vs_ender20": _prediction_correlation(adamw[covered], benchmark, eras),
        "spectral_vs_ender20": _prediction_correlation(spectral[covered], benchmark, eras),
        "spectral_vs_adamw": _prediction_correlation(
            spectral[covered], pd.Series(adamw[covered], index=ids), eras
        ),
        "candidate_vs_ender20": _prediction_correlation(candidate, benchmark, eras),
    }
    per_era = pd.DataFrame({
        "adamw": adam_era["corr"], "spectral": spectral_era["corr"],
        "candidate": candidate_era["corr"], "ender20": benchmark_era,
    })
    output.mkdir(parents=True, exist_ok=True)
    per_era.to_csv(output / "official-validation-per-era.csv")
    _plots(per_era, output)
    secondary, secondary_per_era = _secondary_analyses(
        shard, {"adamw": adamw, "spectral": spectral, "candidate": candidate_full}
    )
    secondary_per_era.to_csv(output / "official-validation-secondary-per-era.csv", index=False)
    _atomic_npz(output / "official-validation-predictions.npz", row_index=indices,
                era=shard.eras[indices], adamw=adamw[covered], spectral=spectral[covered],
                candidate=candidate,
                target=shard.targets[indices, target_column],
                benchmark=shard.benchmarks[indices, benchmark_column])
    report = {
        "status": "complete", "target": "target_cyrusd_20",
        "resolved_rows": int(covered.sum()), "resolved_eras": len(np.unique(shard.eras[covered])),
        "adamw": adam_summary, "spectral": spectral_summary,
        "candidate": candidate_summary, "candidate_transform": transform,
        "ender20": summarize_era_scores(benchmark_era).loc["ender20"].to_dict(),
        "spectral_minus_adamw": spectral_minus_adamw.to_dict(),
        "adamw_minus_ender20": adamw_minus_ender20.to_dict(),
        "spectral_minus_ender20": spectral_minus_ender20.to_dict(),
        "candidate_minus_ender20": candidate_minus_ender20.to_dict(),
        "prediction_correlation": prediction_correlation,
        "secondary": secondary,
        "freeze_manifest_sha256": shard.manifest["freeze_manifest_sha256"],
    }
    atomic_json(output / "official-validation-report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--adamw-model", type=Path, action="append", required=True)
    parser.add_argument("--spectral-model", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=4096)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.shard, args.freeze, args.adamw_model, args.spectral_model,
                              args.output, args.device, args.batch_size), sort_keys=True))


if __name__ == "__main__":
    main()
