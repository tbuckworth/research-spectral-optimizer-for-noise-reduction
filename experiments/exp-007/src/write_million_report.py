#!/usr/bin/env python3
"""Render the audited million-step summary as a concise Markdown report."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
summary = json.loads((ROOT / "out/million-summary.json").read_text())
audit = json.loads((ROOT / "out/million-audit.json").read_text())
assert audit["status"] == "passed"

arms = summary["arms"]
names = ["AdamW", "Top 2048", "Top 3072", "Top 4096"]
best = max(names, key=lambda name: arms[name]["selected_smoothed_valid_corr"])
adamw = arms["AdamW"]["selected_smoothed_valid_corr"]
best_gain = arms[best]["selected_smoothed_valid_corr"] / adamw - 1

rows = []
for name in names:
    arm = arms[name]
    rows.append(
        f"| {name} | {arm['selected_step']:,} | "
        f"{arm['selected_smoothed_valid_corr']:.5f} | "
        f"{arm['selected_raw_valid_corr']:.5f} | "
        f"{arm['test_corr_at_valid_selected_step']:.5f} | "
        f"{arm['train_corr_at_valid_selected_step']:.5f} | "
        f"{arm['terminal_valid_corr']:.5f} | "
        f"{arm['terminal_test_corr']:.5f} |"
    )

report = f"""# Million-step overparameterized Numerai study

## Result

The validation-selected winner was **{best}**, with five-checkpoint rolling
VALID correlation **{arms[best]['selected_smoothed_valid_corr']:.5f}** at step
**{arms[best]['selected_step']:,}**. This was **{best_gain:.1%}** above AdamW's
validation-selected score. This is a paired, single-seed result rather than an
estimate of average optimizer performance.

| Arm | Selected step | Rolling VALID corr. | Raw VALID corr. | Exploratory TEST corr. | TRAIN corr. | Terminal VALID corr. | Terminal TEST corr. |
|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

Checkpoint selection used the maximum five-checkpoint rolling **VALID**
correlation. TEST was not used for selection. Because full TEST was evaluated
at every checkpoint, all TEST values are exploratory trajectories, not an
untouched confirmatory holdout.

## Protocol

- Architecture: 2,376–2,048–1,024–512–1, with 7,491,585 parameters.
- Training data: 4,017,510 rows; the model has 1.86 parameters per row.
- Arms: AdamW and blockwise top-subspace spectral ranks 2,048, 3,072, 4,096.
- Shared seed and paired minibatch stream: 20260805.
- One million updates, with full VALID and exploratory TEST every 10,000 steps
  plus steps 100 and 1,000 (102 measurements per arm).
- Spectral covariance approximation: 30 parameter blocks, maximum block size
  250,000, relative eigenvalue tolerance 1e-12.

## Audit

The machine-readable audit passed for all four artifacts. It verifies the
architecture, parameter and split sizes, seed, learning rate, optimizer/rank
configuration, exact evaluation schedule, finite metrics, spectral filter
metadata at every checkpoint, terminal step, and SHA-256 artifact hashes.

## Interpretation limits

- This is one seed and one fixed temporal split.
- Repeated checkpoints are correlated observations, not independent trials.
- The block-diagonal approximation excludes covariance across parameter blocks.
- The repeated TEST trajectory cannot support a pristine holdout claim.
- The experiment compares optimization/regularization behaviour, not runtime.
"""

(ROOT / "results-million.md").write_text(report)
print(report)
