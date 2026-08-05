# exp-006 — stable spectral filter

## Conclusions

1. **The numerical correction works.** Exact covariance, scale-invariance, resume, fp32 stress, and actual-gradient orthogonality gates all passed.
2. **The corrected hard spectral filter does not help this Numerai MLP.** Rank 16 was selected on VALID, but all spectral ranks remained below AdamW.
3. On held-out TEST, AdamW scored **+0.016863** and corrected spectral scored **+0.000609**; the paired difference was **-0.016254**, 95% block-bootstrap CI **[-0.022137, -0.010983]**.

## VALID rank sweep

| Rank | Mean | Seed SD | Realised k | Retained norm |
|---:|---:|---:|---:|---:|
| 1 | +0.002699 | 0.005017 | 1.0 | 0.212964 |
| 2 | -0.003098 | 0.003334 | 2.0 | 0.999872 |
| 4 | +0.000591 | 0.004971 | 4.0 | 0.999922 |
| 8 | -0.003209 | 0.006803 | 8.0 | 0.796818 |
| 16 | +0.002857 | 0.001074 | 16.0 | 0.999959 |
| 32 | -0.003500 | 0.002236 | 20.3 | 0.999963 |

AdamW selected VALID score: **+0.013102**.

## Paired TEST

| Seed | AdamW | Spectral | Difference |
|---:|---:|---:|---:|
| 0 | +0.015697 | -0.002637 | -0.018333 |
| 1 | +0.007724 | -0.001564 | -0.009288 |
| 2 | +0.017822 | +0.000287 | -0.017535 |
| 3 | +0.020311 | +0.004883 | -0.015429 |
| 4 | +0.022761 | +0.002076 | -0.020685 |

## Scope

This establishes that the original implementation had a serious numerical error and that the local replacement fixes it. It also shows that the intended hard top-k filter, with this MLP and fold, is inferior to AdamW. It does not test soft spectral weighting, alternative decay, or other datasets.

## Figures

- `figures/rank_response_fixed.png`
- `figures/paired_test_fixed.png`
- `figures/cumulative_test_fixed.png`
- `figures/numerical_diagnostics_fixed.png`
