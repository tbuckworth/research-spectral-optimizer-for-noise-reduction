# Common-horizon 300k spectral-rank study

## Main result

On this paired single-seed experiment, every spectral arm materially
outperformed AdamW under the predeclared checkpoint-selection rule.  Performance
largely saturated by rank 1,024–2,048: ranks 1,024, 2,048, 3,072, and 4,096 all
had validation-selected five-checkpoint rolling correlation near 0.020, while
rank 1,536 was modestly lower at 0.01844.  Rank 3,072 had the highest observed
rolling VALID score, 0.02017, but its advantage over rank 1,024 (0.01992) and
rank 2,048 (0.01996) was too small to support a meaningful ordering claim.

| Arm | Hardware | Selected step | Rolling VALID | Raw VALID | Exploratory TEST | TRAIN | VALID at 300k | TEST at 300k |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| AdamW | L40 | 50k | 0.01365 | 0.00955 | 0.01008 | 0.37589 | 0.00675 | 0.00833 |
| Top 1,024 | RTX 3090 | 250k | 0.01992 | 0.01887 | 0.02232 | 0.07157 | **0.02057** | 0.02154 |
| Top 1,536 | RTX 3090 | 150k | 0.01844 | 0.02054 | **0.02393** | 0.06557 | 0.01495 | 0.02150 |
| Top 2,048 | L40 | 160k | 0.01996 | 0.01819 | 0.02384 | 0.07258 | 0.01779 | 0.01758 |
| Top 3,072 | L40 | 150k | **0.02017** | 0.02072 | 0.02392 | 0.07352 | 0.01423 | 0.01644 |
| Top 4,096 | L40 | 130k | 0.02011 | **0.02095** | 0.02147 | 0.07023 | 0.01762 | 0.01911 |

The rule was the maximum five-checkpoint rolling **VALID** correlation through
step 300,000.  TEST was never used for selection.  Because full TEST was
evaluated repeatedly, all TEST values are exploratory and not an untouched
confirmatory holdout.

## What happened

AdamW rapidly fit the training sample: TRAIN correlation reached 0.759 by 300k,
while VALID correlation fell to 0.00675 and VALID MSE rose to 0.0809.  The
spectral variants had TRAIN correlation only 0.076–0.114 at 300k, retained
VALID correlation of 0.0142–0.0206, and kept VALID MSE near 0.050.  This is the
anti-overfitting pattern the experiment was designed to detect.

The higher configured ranks did not produce a monotonic performance gain.
Numerically effective covariance rank per block at 300k increased from 4.78 at
configured rank 1,024 to 22.28 at configured rank 4,096, but validation-selected
performance remained nearly flat.  Additional retained low-energy directions
therefore did not provide a clear benefit in this run.

## Audit and caveats

The machine-readable audit checks all six source artifacts for architecture,
parameter count, temporal split sizes, seed, learning rate, exact 32-point
evaluation schedule, finite full-split metrics, block/filter configuration, and
SHA-256 hashes.  It also verifies the centered-sample rank bound: at step 100,
rank 3,072 and 4,096 can realize only `30 × 99 = 2,970` basis directions before
reaching their configured ranks at later checkpoints.

Ranks 1,024 and 1,536 ran concurrently on an RTX 3090; AdamW and ranks 2,048,
3,072, and 4,096 ran on an L40.  A prior same-checkpoint diagnostic demonstrated
cross-hardware trajectory divergence.  Consequently, small differences between
the 3090 and L40 arms are descriptive rather than clean rank-only causal
estimates.  Other limits are one seed, one fixed temporal split, correlated
checkpoint observations, and a block-diagonal approximation that omits
cross-block parameter covariance.

## Bottom line

The spectral filter clearly improves this deliberately overparameterized
Numerai MLP relative to AdamW by suppressing destructive in-sample fitting.
The useful configured-rank range appears broad, with no compelling benefit above
roughly 1,024–2,048 in these data.  The next scientifically useful replication
would vary seeds or temporal folds, not increase rank or train beyond 300k.
