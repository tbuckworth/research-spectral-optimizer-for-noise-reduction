import hashlib
import json

import pytest
import torch

from numerai_competitive.audit_refits import audit_refits
from numerai_competitive.materialize import materialize_config
from numerai_competitive.model import MLPConfig, ResidualMLP


def _draw(arm, config_id):
    value = {
        "arm": arm, "config_id": config_id, "feature_set": "medium",
        "width": 8, "depth": 2, "residual": False, "normalization": "none",
        "activation": "relu", "dropout": 0.0, "target": "target",
        "batch_mode": "rows", "batch_size": 4, "learning_rate": 1e-3,
        "weight_decay": 0.0, "loss": "mse", "schedule": "constant",
        "warmup_fraction": 0.0, "clip_grad_norm": 0.0,
    }
    if arm == "spectral":
        value |= {
            "rank": 2, "decay": 0.99, "filter_strength": 1.0,
            "filter_warmup": 1, "filter_update_every": 1, "filter_mode": "learned",
        }
    return value


def _refit(root, draw, seed, updates=100000):
    config = materialize_config(draw, input_dim=3, updates=updates, seed=seed)
    split = {
        "name": "sealed_validation_refit_60d",
        "train_eras": [f"{era:04d}" for era in range(1, 559)],
        "valid_eras": [], "purged_eras": [f"{era:04d}" for era in range(559, 575)],
    }
    signature = hashlib.sha256(json.dumps(
        {"config": config, "split": split}, sort_keys=True,
    ).encode()).hexdigest()
    model = ResidualMLP(MLPConfig(**config["model"]))
    directory = root / (
        f"final-refit-u{updates}-s{seed}-{draw['arm']}-c{draw['config_id']}"
    )
    directory.mkdir(parents=True)
    torch.save({
        "signature": signature, "model_config": config["model"],
        "model": model.state_dict(), "train_config": config, "train_split": split,
        "target": "target", "feature_names": ["a", "b", "c"],
        "data_version": "v5.3",
    }, directory / "model.pt")
    result = {
        "status": "complete", "signature": signature, "split": split, "config": config,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "updates": updates, "validation": None, "model_file": "model.pt",
    }
    (directory / "result.json").write_text(json.dumps(result))


def _case(tmp_path):
    draws = [_draw("adamw", 1), _draw("spectral", 2)]
    rows = []
    job = 10
    for draw in draws:
        for seed in range(3):
            _refit(tmp_path / "results", draw, seed)
            rows.append(f"{job}\t{draw['arm']}\t{draw['config_id']}\t100000\t{seed}")
            job += 1
    manifest = tmp_path / "manifest.tsv"
    manifest.write_text("\n".join(rows) + "\n")
    selection = {"selected": {"adamw": [1], "spectral": [2]}}
    return draws, manifest, selection


def test_audit_refits_verifies_six_models_and_provenance(tmp_path):
    draws, manifest, selection = _case(tmp_path)
    report = audit_refits(
        manifest, tmp_path / "results", selection, draws, {"medium": 3},
        tmp_path / "audit",
    )
    assert report["status"] == "audit_complete" and report["cells"] == 6
    assert report["selected"] == {"adamw": 1, "spectral": 2}
    assert (tmp_path / "audit" / "refits.csv").is_file()


def test_audit_refits_rejects_tampered_model_metadata(tmp_path):
    draws, manifest, selection = _case(tmp_path)
    path = tmp_path / "results" / "final-refit-u100000-s0-adamw-c1" / "model.pt"
    artifact = torch.load(path, weights_only=False)
    artifact["data_version"] = "v5.2"
    torch.save(artifact, path)
    with pytest.raises(ValueError, match="differs"):
        audit_refits(
            manifest, tmp_path / "results", selection, draws, {"medium": 3},
            tmp_path / "audit",
        )


def test_audit_refits_accepts_arm_specific_selected_budgets(tmp_path):
    draws = [_draw("adamw", 1), _draw("spectral", 2)]
    budgets = {"adamw": 5000, "spectral": 20000}
    rows = []
    for job, draw in enumerate(draws, 10):
        _refit(tmp_path / "results", draw, 0, budgets[draw["arm"]])
        rows.append(
            f"{job}\t{draw['arm']}\t{draw['config_id']}\t{budgets[draw['arm']]}\t0"
        )
    manifest = tmp_path / "manifest.tsv"
    manifest.write_text("\n".join(rows) + "\n")
    selection = {
        "selected": {"adamw": [1], "spectral": [2]},
        "selected_updates": {"adamw": [5000], "spectral": [20000]},
    }
    report = audit_refits(
        manifest, tmp_path / "results", selection, draws, {"medium": 3},
        tmp_path / "audit", None, (0,),
    )
    assert report["updates"] == budgets
