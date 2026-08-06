# exp-007 results — overparameterization and long-horizon spectral filtering

The model was a 7,491,585-parameter MLP trained on 4,017,510 TRAIN rows. All
arms used the same seed and sampled minibatch stream. Metrics used fixed sampled
TRAIN and VALID monitors; TEST remained untouched. High ranks used a declared
30-block block-diagonal parameter-covariance approximation because the exact
global rank-256 implementation exceeded 24 GB before reaching its rank cap.

## 20k paired result

AdamW peaked near step 9,500 and then overfit: TRAIN correlation continued
rising while VALID correlation and MSE deteriorated. Global rank-16 projection
prevented both memorization and useful learning.

## 100k blockwise result

| Arm | Best 5k-smoothed VALID correlation | Peak step | Final-5k mean |
|---|---:|---:|---:|
| AdamW | 0.01568 | 9,500 | 0.00506 |
| Top 256 | 0.01673 | 41,500 | 0.01368 |
| Top 512 | 0.01681 | 77,000 | 0.01518 |
| Top 1,024 | 0.01822 | 83,000 | 0.01563 |
| Top 2,048 | **0.01913** | 75,500 | **0.01814** |

AdamW ended with TRAIN correlation 0.561, VALID correlation 0.0007, and VALID
MSE 0.0670. Top-2,048 ended with TRAIN correlation 0.057, VALID correlation
0.0190, and VALID MSE 0.04965. Thus broad top-subspace projection learned much
more slowly but prevented runaway memorization and produced the best sustained
VALID ranking performance.

Complement-removal arms were not reliable. Raw remove-1,024 briefly reached a
strong smoothed score but catastrophically diverged by 100k (VALID MSE 58.4).
Norm-matched removal also deteriorated. These are retained as negative results.

## Limits

- One paired seed; checkpoint measurements are temporally correlated.
- Sampled VALID monitor rather than a full VALID pass.
- Block-diagonal covariance omits cross-block interactions.
- No untouched TEST evaluation yet; the next experiment treats repeated TEST
  trajectories as exploratory and must not call TEST an untouched holdout.
