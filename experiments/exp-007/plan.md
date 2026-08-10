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

## Stage 3 — long-horizon saturation

Rerun the paired AdamW and complete blockwise grid for 100,000 steps, evaluating
every 500 steps. Do not early-stop individual variants. Compare raw peaks,
smoothed peaks, peak timing, late performance, and train/validation divergence.

## Stage 4 — million-step rank saturation with exploratory TEST

Run AdamW and blockwise top ranks 2,048, 3,072, and 4,096 for one million steps
from the paired seed and minibatch stream. Evaluate complete VALID and complete
TEST splits every 10,000 steps. Because TEST is repeatedly observed here, call
it an exploratory trajectory rather than an untouched confirmatory holdout.

Launch four resumable 12-hour allocations with:

```bash
./submit-million-chain.sh 4
```

The chain deliberately uses Slurm `afterany`, because reaching the QOS time
limit is the expected reason for one allocation to end. `afterok` would cancel
all later allocations instead of resuming from the durable checkpoints.
