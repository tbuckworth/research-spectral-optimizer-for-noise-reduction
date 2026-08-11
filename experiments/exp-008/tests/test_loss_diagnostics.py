import json

from numerai_competitive.loss_diagnostics import collect_loss, write_loss_diagnostics


def _result(root, arm, config_id):
    path = root / f"stage-fold-u10-s0-{arm}-c{config_id}"
    path.mkdir()
    multiplier = 0.8 if arm == "spectral" else 0.9
    path.joinpath("result.json").write_text(json.dumps({
        "status": "complete", "updates": 10,
        "config": {"arm": arm, "seed": 0, "loss": "mse"},
        "logs": [{"update": update, "loss": multiplier**index}
                 for index, update in enumerate((1, 5, 10))],
    }))


def test_loss_diagnostics_are_paired_normalized_and_hashed(tmp_path):
    for config_id in range(2):
        for arm in ("adamw", "spectral"):
            _result(tmp_path, arm, config_id)
    frame = collect_loss(tmp_path, split="fold", updates=10, seed=0)
    assert len(frame) == 12
    assert frame.groupby(["arm", "config_id"])["loss_over_initial"].first().eq(1).all()
    output = tmp_path / "plots"
    write_loss_diagnostics(frame, output)
    marker = json.loads((output / "loss-diagnostics-complete.json").read_text())
    assert marker["status"] == "complete" and marker["configs_per_arm"] == 2
    assert (output / "training-loss.png").is_file()
