# F0 paired low-fidelity screen

Status: complete and provenance-audited on 2026-08-10. This is a train-only inner-fold
screen, not sealed validation or leaderboard evidence.

## Design

- Split: `outer_1_inner_1` (train eras 0001–0148, purge 0149–0156, evaluate 0157–0234)
- Budget: 5,000 updates, seed 0
- Search: 40 frozen shared configurations, evaluated once with AdamW and once with the
  spectral filter (80 cells total)
- Primary selection metric: exact standalone Numerai CORR mean
- Provenance: all 80 legacy-schema results were matched to the complete frozen scientific
  configuration and official v5.3 feature-set dimension; each result signature and prediction
  artifact was independently verified and hashed

## Results

| Quantity | Result |
|---|---:|
| Spectral wins / AdamW wins | 18 / 22 |
| Mean paired spectral − AdamW CORR | −0.002653 |
| Median paired spectral − AdamW CORR | −0.000367 |
| Best AdamW CORR | 0.048384 (config 38) |
| Best spectral CORR | 0.050616 (config 38) |

The distribution does not show a broad spectral advantage at F0. The best spectral cell does
beat the best AdamW cell, so the predeclared temporal confirmation stage remains informative.
These maxima are selected on this fold and must not be interpreted as unbiased performance.

The paired loss diagnostic uses the same deterministic minibatch stream within each config and
normalizes each run by its first logged loss, so MSE and Huber scales are not averaged directly.
At update 5,000, spectral had a higher normalized logged minibatch loss in 40/40 pairs. The median
spectral/AdamW final-loss ratio was 1.028 and the mean was 1.111 (largest: 3.304, config 7).
Therefore the filter consistently fit the training objective less at this fidelity. This is a
descriptive mechanism result, not evidence of better or worse generalization.

AdamW top 12: `38, 1, 39, 35, 10, 23, 26, 4, 15, 36, 9, 8`.

Spectral top 12: `38, 1, 9, 18, 10, 23, 30, 26, 19, 4, 8, 22`.

Their paired union has 16 configurations. F1 therefore contains 64 exact cells:
16 configurations × 2 optimizer arms × 2 temporal inner folds, each at 20,000 updates and
seed 0.

## Artifacts

- `scores.csv`: audited result-level metrics, signatures, hashes, and provenance mode
- `paired-corr.csv`: paired CORR values and deltas
- `paired-corr.png`: AdamW-vs-spectral scatter and per-configuration deltas
- `selection-outer_1-f0-top12.json`: deterministic arm-specific ranking and paired union
- `submission-outer_1-f1-u20000-s0.tsv`: exact F1 Slurm submission manifest
- `training-loss-curves.csv`, `training-loss.png`: paired normalized train-loss trajectories and
  final-loss scatter
- `loss-diagnostics-complete.json`: hashes and coverage for the loss artifacts

Raw `result.json` and `validation_predictions.npz` files are archived locally under
`experiments/exp-008/out/mats-f0/` and remain on MATS persistent storage.
