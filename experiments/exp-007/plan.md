# exp-007 — deliberate overparameterization and optimizer trajectories

## Question

Does the spectral gradient filter help when the Numerai MLP is made larger than
the refit dataset and trained long enough for validation correlation to peak and
decline?

## Stage 1

- Frozen fold: TRAIN eras 1–791, VALID eras 796–891. TEST is not touched.
- Architecture: 2376–2048–1024–512–1 (7,491,585 parameters).
- Paired initialization and minibatch stream for AdamW and the corrected local
  parameter-space spectral filter.
- Record train/VALID MSE and per-era Numerai correlation throughout training.
- Train for 20,000 steps (~5.1 passes in sampled-row equivalents).
- Initial spectral rank: 16, because exact storage scales as `parameters × rank`.

## Stage 2

On the existing 625,025-parameter MLP, sweep exact top-projection ranks 64, 256,
1024, and 2048. Compare against complementary ablation `g - V(V^T g)` at ranks
1, 16, 64, and 256, both raw and norm-matched. Run AdamW on the identical seed
and batch stream. TEST remains untouched.

## Decision rule

Evidence for the anti-overfitting hypothesis requires AdamW VALID correlation to
decline after its peak while spectral retains materially more of its own peak.
Similar trajectories, or spectral deterioration before AdamW, reject that
mechanism in this regime.
