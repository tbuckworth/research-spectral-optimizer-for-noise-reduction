# Corrected validation-selected live refit protocol

This protocol replaces the erroneous fixed-20,000-update live refits. It is frozen before
the new calibration results are inspected. Numerai diagnostics and the outer holdout already
observed in this project are excluded from every decision below.

## Frozen candidates and development data

- AdamW: search config 38, seeds 0/1/2 for the final ensemble.
- Spectral: search config 39, seeds 0/1/2 for the final ensemble.
- Stopping calibration uses only `outer_1_inner_1` and `outer_1_inner_2`, seed 0.
- The 16-era target-horizon purge remains unchanged.
- The final fit uses all 574 resolved training eras. No official validation targets enter
  calibration or fitting.

## Comparable stopping calibration

Each arm follows one fixed cosine-schedule trajectory to 40,960,000 sampled examples. The
common checkpoints are 2,560,000, 5,120,000, 10,240,000, 20,480,000, and 40,960,000 examples.
Thus AdamW (batch 1,024) is evaluated at 2,500/5,000/10,000/20,000/40,000 updates and spectral
(batch 2,048) at 1,250/2,500/5,000/10,000/20,000 updates. Batch size no longer silently gives
one arm twice the exposure at an identically named update budget.

For each arm and checkpoint, calculate mean validation-era Corr on each inner fold, then average
the two fold means equally. Select the checkpoint with the largest average Corr; exact ties choose
the earlier checkpoint. BMC is reported but does not select the stopping point. AdamW and spectral
are selected independently.

## Final train-plus-validation refit

The selected update and full cosine-schedule horizon are multiplied by `574 / 218`, where 218 is
the number of training eras in the later calibration fold and 574 is the number in the final
resolved training set. Round to the nearest integer update. This preserves the selected exposure
per training era and the selected position within the learning-rate schedule. It also preserves
each arm's own batch size and independently selected stopping point.

Train seeds 0, 1, and 2 for each arm and average their predictions. Audit exact configuration,
data identity, update count, schedule horizon, model hashes, feature order, finite predictions,
and official-container runtime before upload.

## Live replacement

Export one callable for `eden_adam` and one for `eden_eve`. Upload both as unstaked replacements
only after all audits pass. Never create or alter a stake. Existing round-1334 submissions remain
historical evidence; corrected bundles apply to subsequent rounds.

## Cluster storage

All downloaded data, generated shards, package caches, and temporary artifacts live below
`/ephemeral/t.buckworth`. Only code, compact logs, final checkpoints, and audited bundles are
written to `/mnt/nw/home/t.buckworth`. No large file is written under `/tmp`.
