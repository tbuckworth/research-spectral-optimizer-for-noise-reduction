import hashlib
import json

import numpy as np
import pandas as pd

from numerai_competitive.audit_outer import audit_outer
from numerai_competitive.materialize import materialize_config
from numerai_competitive.metrics import (
    per_era_corr,
    per_era_correlation_contribution,
    summarize_era_scores,
)
from numerai_competitive.splits import EraSplit


def _draw(arm):
    draw = {
        "arm": arm, "config_id": 7, "feature_set": "medium", "width": 8,
        "depth": 2, "residual": False, "normalization": "none", "activation": "relu",
        "dropout": 0.0, "target": "target_cyrusd_20", "batch_mode": "rows",
        "batch_size": 4, "learning_rate": 1e-3, "weight_decay": 0.0, "loss": "mse",
        "schedule": "constant", "warmup_fraction": 0.0, "clip_grad_norm": 0.0,
    }
    if arm == "spectral":
        draw |= {"rank": 2, "decay": 0.99, "filter_strength": 1.0,
                 "filter_warmup": 1, "filter_update_every": 1, "filter_mode": "learned"}
    return draw


def _summary(values):
    return summarize_era_scores(values).loc["prediction"].to_dict()


def _result(root, draw, split):
    arm = draw["arm"]
    task = root / f"stage-outer_1-u10-s0-{arm}-c7"
    task.mkdir(parents=True)
    config = materialize_config(draw, input_dim=3, updates=10, seed=0)
    eras_raw = np.repeat([2, 3], 5)
    row_index = np.arange(10)
    index = pd.Index(row_index)
    eras = pd.Series([f"{era:04d}" for era in eras_raw], index=index, name="era")
    target = pd.Series(
        [0.1, 0.8, 0.3, 0.9, 0.5, 0.7, 0.2, 0.6, 0.4, 1.0],
        index=index, name="target",
    )
    prediction = pd.Series(
        target.to_numpy() + (0.01 if arm == "spectral" else -0.01) * np.arange(10),
        index=index, name="prediction",
    )
    benchmark = pd.Series(np.linspace(0.9, 0.1, 10), index=index, name="benchmark")
    corr = per_era_corr(prediction, target, eras)["prediction"]
    bmc = per_era_correlation_contribution(prediction, benchmark, target, eras)["prediction"]
    np.savez_compressed(
        task / "validation_predictions.npz", row_index=row_index, era=eras_raw,
        target=target.to_numpy(), benchmark=benchmark.to_numpy(),
        prediction=prediction.to_numpy(), per_era_corr=corr.to_numpy(),
        per_era_bmc=bmc.to_numpy(),
    )
    signature = hashlib.sha256(json.dumps(
        {"config": config, "split": split.to_dict()}, sort_keys=True,
    ).encode()).hexdigest()
    (task / "result.json").write_text(json.dumps({
        "status": "complete", "signature": signature, "split": split.to_dict(),
        "config": config, "updates": 10, "prediction_file": "validation_predictions.npz",
        "validation": {"rows": 10, "corr": _summary(corr), "bmc": _summary(bmc)},
    }))


def test_outer_audit_verifies_exact_selected_cells_and_reproduces_scores(tmp_path):
    split = EraSplit("outer_1", ("0001",), ("0002", "0003"), ())
    draws = [_draw("adamw"), _draw("spectral")]
    for draw in draws:
        _result(tmp_path / "results", draw, split)
    manifest = tmp_path / "outer.tsv"
    manifest.write_text(
        "1\touter_1\t10\t0\tadamw\t7\n2\touter_1\t10\t0\tspectral\t7\n"
    )
    selection = {"selected": {"adamw": [7], "spectral": [7], "paired_union": [7]}}
    report = audit_outer(
        manifest, tmp_path / "results", selection, draws, {"medium": 3}, split, 10,
        (0,), tmp_path / "audit",
    )
    assert report["status"] == "audit_complete" and report["cells"] == 2
    assert (tmp_path / "audit" / "outer-results.csv").is_file()
    assert (tmp_path / "audit" / "outer-audit.json").is_file()
