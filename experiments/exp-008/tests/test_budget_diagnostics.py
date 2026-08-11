from pathlib import Path

import pandas as pd

from numerai_competitive.budget_diagnostics import build_budget_diagnostics
from numerai_competitive.data import sha256


def test_builds_audited_budget_table_and_plot(tmp_path: Path) -> None:
    rows = []
    for arm in ("adamw", "spectral"):
        for updates, base in ((5_000, 0.04), (100_000, 0.01)):
            for split, offset in (("inner_1", 0.001), ("inner_2", -0.001)):
                rows.append({
                    "arm": arm, "config_id": 7, "updates": updates,
                    "split": split, "seed": 0,
                    "corr_mean": base + offset + (0.01 if arm == "spectral" else 0),
                })
    output = tmp_path / "diagnostics"
    report = build_budget_diagnostics(pd.DataFrame(rows), [7], output)

    assert report["status"] == "development_budget_sensitivity_complete"
    assert report["cells"] == 8
    assert report["selected"] == {
        "adamw": [{"config_id": 7, "updates": 5_000}],
        "spectral": [{"config_id": 7, "updates": 5_000}],
    }
    for name in ("budget-sensitivity.csv", "budget-sensitivity.png"):
        path = output / name
        assert path.is_file() and report["artifacts"][name] == sha256(path)
    saved = pd.read_csv(output / "budget-sensitivity.csv")
    assert len(saved) == 4
    assert set(saved["cells"]) == {2}
